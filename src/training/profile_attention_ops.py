"""
Focused profiler for Chess-Fusion attention paths.

This script benchmarks the custom attention modules used in
``chess_fusion_training`` and highlights:

- SDPA fast path vs. the manual ``need_weights=True`` fallback
- dynamic-mask build overhead vs. cached mask reuse
- ``torch.compile`` cold-start cost and likely recompile spikes when shapes vary

Examples:
    python -m training.profile_attention_ops --device cuda
    python -m training.profile_attention_ops --case manual_mha --device cuda
    python -m training.profile_attention_ops --compile-backend aot_eager --device cuda
    python -m training.profile_attention_ops --device cpu --json-output attention_profile.json
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

# Ensure src/ is on the path when running as a script.
_src = str(Path(__file__).parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    dtype = mapping[name]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 is not supported well enough on CPU for this profiler.")
    return dtype


def _nested_counter_dict() -> Dict[str, Dict[str, int]]:
    try:
        from torch._dynamo import utils as dynamo_utils

        return {
            group: {key: int(value) for key, value in counter.items()}
            for group, counter in dynamo_utils.counters.items()
            if counter
        }
    except Exception:
        return {}


def _reset_dynamo_state() -> None:
    try:
        import torch._dynamo as dynamo

        dynamo.reset()
    except Exception:
        pass
    try:
        from torch._dynamo import utils as dynamo_utils

        dynamo_utils.counters.clear()
    except Exception:
        pass


def _compile_times_summary() -> str:
    try:
        from torch._dynamo import utils as dynamo_utils

        summary = str(dynamo_utils.compile_times()).strip()
        return summary
    except Exception:
        return ""


def _diff_counters(
    before: Dict[str, Dict[str, int]],
    after: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    for group in sorted(set(before) | set(after)):
        keys = set(before.get(group, {})) | set(after.get(group, {}))
        delta: Dict[str, int] = {}
        for key in sorted(keys):
            value = after.get(group, {}).get(key, 0) - before.get(group, {}).get(key, 0)
            if value:
                delta[key] = value
        if delta:
            result[group] = delta
    return result


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * pct) - 1))
    return ordered[idx]


def _summarize_times(
    step_times_ms: List[float],
    labels: List[str],
    warmup: int,
) -> Dict[str, Any]:
    if not step_times_ms:
        return {
            "steps": 0,
            "first_ms": 0.0,
            "median_ms": 0.0,
            "mean_ms": 0.0,
            "steady_mean_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
            "slow_steps": [],
        }

    steady = step_times_ms[warmup:] if len(step_times_ms) > warmup else step_times_ms[1:]
    if not steady:
        steady = step_times_ms

    median_ms = statistics.median(steady)
    steady_mean_ms = statistics.mean(steady)
    slow_steps = []
    threshold = max(median_ms * 1.75, median_ms + 2.0) if median_ms > 0 else 0.0
    for idx, (ms, label) in enumerate(zip(step_times_ms, labels), start=1):
        if idx <= warmup:
            continue
        if threshold and ms >= threshold:
            slow_steps.append(
                {
                    "step": idx,
                    "label": label,
                    "ms": round(ms, 3),
                }
            )

    return {
        "steps": len(step_times_ms),
        "first_ms": step_times_ms[0],
        "median_ms": median_ms,
        "mean_ms": statistics.mean(step_times_ms),
        "steady_mean_ms": steady_mean_ms,
        "p95_ms": _percentile(steady, 0.95),
        "max_ms": max(step_times_ms),
        "slow_steps": slow_steps,
    }


def _format_ms(value: float) -> str:
    return f"{value:8.3f} ms"


def _can_use_compile_backend(
    device: torch.device,
    backend: str,
) -> tuple[bool, str]:
    if not hasattr(torch, "compile"):
        return False, "torch.compile is not available in this PyTorch build."
    if backend == "inductor" and device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        if major < 7:
            return (
                False,
                "Inductor/Triton on CUDA needs compute capability >= 7.0; "
                f"current GPU is {major}.{minor}.",
            )
    return True, ""


def _build_compile_wrapper(
    target: Callable[..., Any],
    backend: str,
    mode: str,
    dynamic: bool,
) -> Callable[..., Any]:
    kwargs: Dict[str, Any] = {"backend": backend, "dynamic": dynamic}
    if mode != "none":
        kwargs["mode"] = mode
    return torch.compile(target, **kwargs)


def _device_summary(device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "torch": torch.__version__,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "cuda": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        summary["cuda_version"] = torch.version.cuda
        summary["gpu_name"] = torch.cuda.get_device_name(device)
        summary["compute_capability"] = f"{major}.{minor}"
        summary["flash_sdp_enabled"] = bool(torch.backends.cuda.flash_sdp_enabled())
        summary["mem_efficient_sdp_enabled"] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
        summary["math_sdp_enabled"] = bool(torch.backends.cuda.math_sdp_enabled())
    return summary


def _print_environment(
    device: torch.device,
    dtype: torch.dtype,
    backend: str,
    mode: str,
    dynamic: bool,
) -> None:
    info = _device_summary(device, dtype)
    print("=" * 78)
    print("Attention Profiler")
    print("=" * 78)
    print(f"torch:              {info['torch']}")
    print(f"device:             {info['device']}")
    print(f"dtype:              {info['dtype']}")
    print(f"compile backend:    {backend}")
    print(f"compile mode:       {mode}")
    print(f"compile dynamic:    {dynamic}")
    if device.type == "cuda":
        print(f"gpu:                {info['gpu_name']}")
        print(f"compute capability: {info['compute_capability']}")
        print(f"CUDA runtime:       {info['cuda_version']}")
        print(
            "SDPA toggles:       "
            f"flash={info['flash_sdp_enabled']}  "
            f"mem_efficient={info['mem_efficient_sdp_enabled']}  "
            f"math={info['math_sdp_enabled']}"
        )
    print()


def _make_manual_inputs(
    batch_size: int,
    seq_lens: Iterable[int],
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for seq_len in seq_lens:
        q = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        attn_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        if seq_len > 8:
            attn_mask[:, int(seq_len * 0.9):] = True
        payloads.append(
            {
                "label": f"seq={seq_len}",
                "query": q,
                "key": k,
                "value": v,
                "attn_mask": attn_mask,
            }
        )
    return payloads


def _make_random_boards(batch_size: int, device: torch.device) -> torch.Tensor:
    boards = torch.zeros(batch_size, 18, 8, 8, device=device, dtype=torch.float32)
    piece_ids = torch.randint(0, 13, (batch_size, 64), device=device)
    occupied = piece_ids > 0
    if occupied.any():
        b_idx, sq_idx = occupied.nonzero(as_tuple=True)
        ch_idx = piece_ids[occupied] - 1
        row_idx = sq_idx // 8
        col_idx = sq_idx % 8
        boards[b_idx, ch_idx, row_idx, col_idx] = 1.0

    # Side-to-move and a couple of rights planes to keep the tensor non-empty.
    boards[:, 12].fill_(1.0)
    boards[:, 13].fill_(1.0)
    boards[:, 15].fill_(1.0)
    return boards


def _make_chess_inputs(
    batch_sizes: Iterable[int],
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 64, embed_dim, device=device, dtype=dtype)
        boards = _make_random_boards(batch_size, device)
        payloads.append(
            {
                "label": f"batch={batch_size}",
                "batch_size": batch_size,
                "x": x,
                "boards": boards,
            }
        )
    return payloads


def _make_csmp_inputs(
    batch_sizes: Iterable[int],
    num_taps: int,
    cnn_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for batch_size in batch_sizes:
        boards = _make_random_boards(batch_size, device)
        cnn_tap_features = [
            torch.randn(batch_size, cnn_dim, 8, 8, device=device, dtype=dtype)
            for _ in range(max(num_taps, 0))
        ]
        payloads.append(
            {
                "label": f"batch={batch_size}",
                "batch_size": batch_size,
                "boards": boards,
                "cnn_tap_features": cnn_tap_features,
            }
        )
    return payloads


def _benchmark(
    label: str,
    payloads: List[Dict[str, Any]],
    fn: Callable[[Dict[str, Any]], Any],
    device: torch.device,
    warmup: int,
    capture_compile: bool = False,
) -> Dict[str, Any]:
    if capture_compile:
        before_counters = _nested_counter_dict()
    else:
        before_counters = {}

    step_times_ms: List[float] = []
    labels = [payload["label"] for payload in payloads]
    error: Optional[str] = None

    with torch.inference_mode():
        for payload in payloads:
            try:
                _synchronize(device)
                t0 = time.perf_counter()
                _ = fn(payload)
                _synchronize(device)
                step_times_ms.append((time.perf_counter() - t0) * 1000.0)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                break

    summary = _summarize_times(step_times_ms, labels[: len(step_times_ms)], warmup)
    result: Dict[str, Any] = {
        "label": label,
        "error": error,
        "times_ms": [round(v, 3) for v in step_times_ms],
        "labels": labels[: len(step_times_ms)],
        "summary": {
            key: (
                value
                if not isinstance(value, float)
                else round(value, 3)
            )
            for key, value in summary.items()
        },
    }

    if capture_compile:
        after_counters = _nested_counter_dict()
        result["dynamo_counters"] = _diff_counters(before_counters, after_counters)
        compile_times = _compile_times_summary()
        if compile_times:
            result["compile_times"] = compile_times

    return result


def _slowdown_factor(
    faster: Optional[Dict[str, Any]],
    slower: Optional[Dict[str, Any]],
) -> Optional[float]:
    if not faster or not slower:
        return None
    fast_ms = faster.get("summary", {}).get("steady_mean_ms", 0.0)
    slow_ms = slower.get("summary", {}).get("steady_mean_ms", 0.0)
    if not fast_ms or not slow_ms:
        return None
    return slow_ms / fast_ms


def _steady_ms(result: Optional[Dict[str, Any]]) -> float:
    if not result:
        return 0.0
    return float(result.get("summary", {}).get("steady_mean_ms", 0.0))


def _print_variant(result: Dict[str, Any]) -> None:
    print(f"  {result['label']}")
    if result.get("error"):
        print(f"    error: {result['error']}")
        return

    summary = result["summary"]
    print(
        "    "
        f"first={_format_ms(summary['first_ms'])}  "
        f"steady={_format_ms(summary['steady_mean_ms'])}  "
        f"p95={_format_ms(summary['p95_ms'])}  "
        f"max={_format_ms(summary['max_ms'])}"
    )

    if summary.get("slow_steps"):
        print("    likely slow / recompile steps:")
        for item in summary["slow_steps"]:
            print(f"      step {item['step']:>2}  {item['label']:<14}  {item['ms']:>8.3f} ms")

    counters = result.get("dynamo_counters", {})
    stats = counters.get("stats", {})
    graph_breaks = counters.get("graph_break", {})
    if stats:
        interesting = []
        for key in ("unique_graphs", "calls_captured"):
            if key in stats:
                interesting.append(f"{key}={stats[key]}")
        if interesting:
            print(f"    dynamo stats: {', '.join(interesting)}")
    if graph_breaks:
        print("    graph breaks:")
        for reason, count in sorted(graph_breaks.items(), key=lambda item: (-item[1], item[0])):
            print(f"      {count}x {reason}")
    compile_times = result.get("compile_times", "")
    if compile_times:
        lines = [line.strip() for line in compile_times.splitlines() if line.strip()]
        preview = lines[:3]
        if preview:
            print("    compile metrics:")
            for line in preview:
                print(f"      {line}")


def _run_manual_mha_case(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    from training.chess_fusion_model import ManualMultiHeadAttention

    seq_schedule = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    if not seq_schedule:
        raise ValueError("seq-lens must contain at least one integer.")
    fixed_schedule = [seq_schedule[0]] * len(seq_schedule)

    eager_module = ManualMultiHeadAttention(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=0.0,
    ).to(device=device, dtype=dtype)
    eager_module.eval()

    fixed_payloads = _make_manual_inputs(
        batch_size=args.batch_size,
        seq_lens=fixed_schedule,
        embed_dim=args.embed_dim,
        device=device,
        dtype=dtype,
    )
    varying_payloads = _make_manual_inputs(
        batch_size=args.batch_size,
        seq_lens=seq_schedule,
        embed_dim=args.embed_dim,
        device=device,
        dtype=dtype,
    )

    results: Dict[str, Any] = {"case": "manual_mha", "variants": []}

    eager_fast = _benchmark(
        label="eager / SDPA fast path",
        payloads=fixed_payloads,
        fn=lambda payload: eager_module(
            query=payload["query"],
            key=payload["key"],
            value=payload["value"],
            attn_mask=payload["attn_mask"],
            need_weights=False,
        ),
        device=device,
        warmup=args.warmup,
    )
    results["variants"].append(eager_fast)

    eager_entropy = _benchmark(
        label="eager / need_weights=True",
        payloads=fixed_payloads,
        fn=lambda payload: eager_module(
            query=payload["query"],
            key=payload["key"],
            value=payload["value"],
            attn_mask=payload["attn_mask"],
            need_weights=True,
            average_attn_weights=False,
        ),
        device=device,
        warmup=args.warmup,
    )
    results["variants"].append(eager_entropy)

    if not args.skip_compile:
        can_compile, reason = _can_use_compile_backend(device, args.compile_backend)
        if can_compile:
            _reset_dynamo_state()
            compiled_fixed = _build_compile_wrapper(
                eager_module,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=args.compile_dynamic,
            )
            results["variants"].append(
                _benchmark(
                    label="compile / fixed shape / need_weights=False",
                    payloads=fixed_payloads,
                    fn=lambda payload: compiled_fixed(
                        query=payload["query"],
                        key=payload["key"],
                        value=payload["value"],
                        attn_mask=payload["attn_mask"],
                        need_weights=False,
                    ),
                    device=device,
                    warmup=args.warmup,
                    capture_compile=True,
                )
            )

            _reset_dynamo_state()
            compiled_varying = _build_compile_wrapper(
                eager_module,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=args.compile_dynamic,
            )
            results["variants"].append(
                _benchmark(
                    label="compile / varying seq shape / need_weights=False",
                    payloads=varying_payloads,
                    fn=lambda payload: compiled_varying(
                        query=payload["query"],
                        key=payload["key"],
                        value=payload["value"],
                        attn_mask=payload["attn_mask"],
                        need_weights=False,
                    ),
                    device=device,
                    warmup=args.warmup,
                    capture_compile=True,
                )
            )
        else:
            results["compile_skipped"] = reason

    entropy_slowdown = _slowdown_factor(eager_fast, eager_entropy)
    if entropy_slowdown is not None:
        results["notes"] = [
            f"need_weights=True steady-state slowdown vs SDPA fast path: {entropy_slowdown:.2f}x"
        ]

    return results


def _run_chess_structure_case(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    from training.chess_structure_mp import (
        ChessStructureAttention,
        ChessStructureMP,
        DynamicMaskBuilder,
    )

    batch_schedule = [int(x) for x in args.batch_schedule.split(",") if x.strip()]
    if not batch_schedule:
        raise ValueError("batch-schedule must contain at least one integer.")
    fixed_schedule = [batch_schedule[0]] * len(batch_schedule)

    attention = ChessStructureAttention(
        dim=args.embed_dim,
        n_heads=max(args.num_heads, 8),
        dropout=0.0,
    ).to(device=device, dtype=dtype)
    attention.eval()
    csmp = ChessStructureMP(
        cnn_dim=args.csmp_cnn_dim,
        num_taps=args.csmp_num_taps,
        output_dim=args.csmp_output_dim if args.csmp_output_dim > 0 else args.embed_dim,
        csmp_dim=args.csmp_dim if args.csmp_dim > 0 else args.embed_dim,
        pos_dim=args.csmp_pos_dim,
        piece_dim=args.csmp_piece_dim,
        cnn_proj_dim=args.csmp_cnn_proj_dim if args.csmp_cnn_proj_dim > 0 else None,
        n_layers=args.csmp_layers,
        n_heads=max(args.num_heads, 8),
        ffn_mult=args.csmp_ffn_mult,
        dropout=0.0,
        use_ray_mask=True,
        use_attack_mask=True,
        use_xy_coords=args.csmp_use_xy_coords,
        ablation_no_mask=False,
    ).to(device=device, dtype=dtype)
    csmp.eval()
    masks = DynamicMaskBuilder().to(device)

    fixed_payloads = _make_chess_inputs(
        batch_sizes=fixed_schedule,
        embed_dim=args.embed_dim,
        device=device,
        dtype=dtype,
    )
    varying_payloads = _make_chess_inputs(
        batch_sizes=batch_schedule,
        embed_dim=args.embed_dim,
        device=device,
        dtype=dtype,
    )
    fixed_csmp_payloads = _make_csmp_inputs(
        batch_sizes=fixed_schedule,
        num_taps=args.csmp_num_taps,
        cnn_dim=args.csmp_cnn_dim,
        device=device,
        dtype=dtype,
    )
    varying_csmp_payloads = _make_csmp_inputs(
        batch_sizes=batch_schedule,
        num_taps=args.csmp_num_taps,
        cnn_dim=args.csmp_cnn_dim,
        device=device,
        dtype=dtype,
    )

    def _build_ray_mask_for_payload(payload: Dict[str, Any]) -> torch.Tensor:
        return masks.compute_ray_mask(payload["boards"])

    def _build_attack_mask_for_payload(payload: Dict[str, Any]) -> torch.Tensor:
        ray_mask = payload.get("ray_mask")
        if ray_mask is None:
            ray_mask = _build_ray_mask_for_payload(payload)
        return masks.compute_attack_mask(payload["boards"], ray_mask=ray_mask)

    def _build_head_masks_for_payload(payload: Dict[str, Any]) -> torch.Tensor:
        ray_mask = payload.get("ray_mask")
        if ray_mask is None:
            ray_mask = _build_ray_mask_for_payload(payload)
        attack_mask = payload.get("attack_mask")
        if attack_mask is None:
            attack_mask = _build_attack_mask_for_payload(
                {"boards": payload["boards"], "ray_mask": ray_mask}
            )
        return attention._build_head_masks(
            payload["batch_size"],
            device,
            ray_mask,
            attack_mask,
        )

    def _attach_mask_artifacts(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for payload in payloads:
            enriched_payload = dict(payload)
            enriched_payload["ray_mask"] = _build_ray_mask_for_payload(payload)
            enriched_payload["attack_mask"] = _build_attack_mask_for_payload(enriched_payload)
            enriched_payload["head_masks"] = _build_head_masks_for_payload(enriched_payload)
            enriched.append(enriched_payload)
        return enriched

    fixed_with_masks = _attach_mask_artifacts(fixed_payloads)
    varying_with_masks = _attach_mask_artifacts(varying_payloads)

    results: Dict[str, Any] = {"case": "chess_structure_attention", "variants": []}

    ray_only = _benchmark(
        label="eager / ray mask only",
        payloads=fixed_payloads,
        fn=_build_ray_mask_for_payload,
        device=device,
        warmup=args.warmup,
    )
    results["variants"].append(ray_only)

    attack_only = _benchmark(
        label="eager / attack mask only (ray cached)",
        payloads=fixed_with_masks,
        fn=lambda payload: masks.compute_attack_mask(
            payload["boards"],
            ray_mask=payload["ray_mask"],
        ),
        device=device,
        warmup=args.warmup,
    )
    results["variants"].append(attack_only)

    head_assembly_only = _benchmark(
        label="eager / head mask assembly only",
        payloads=fixed_with_masks,
        fn=lambda payload: attention._build_head_masks(
            payload["batch_size"],
            device,
            payload["ray_mask"],
            payload["attack_mask"],
        ),
        device=device,
        warmup=args.warmup,
    )
    results["variants"].append(head_assembly_only)

    mask_build_only = _benchmark(
        label="eager / build ray+attack+head masks",
        payloads=fixed_payloads,
        fn=_build_head_masks_for_payload,
        device=device,
        warmup=args.warmup,
    )
    results["variants"].append(mask_build_only)

    cached_attention = _benchmark(
        label="eager / attention with cached head masks",
        payloads=fixed_with_masks,
        fn=lambda payload: attention(payload["x"], head_masks=payload["head_masks"]),
        device=device,
        warmup=args.warmup,
    )
    results["variants"].append(cached_attention)

    rebuild_every_time = _benchmark(
        label="eager / rebuild masks + attention",
        payloads=fixed_payloads,
        fn=lambda payload: attention(
            payload["x"],
            head_masks=_build_head_masks_for_payload(payload),
        ),
        device=device,
        warmup=args.warmup,
    )
    results["variants"].append(rebuild_every_time)

    full_csmp = _benchmark(
        label="eager / full ChessStructureMP.forward",
        payloads=fixed_csmp_payloads,
        fn=lambda payload: csmp(
            payload["cnn_tap_features"],
            payload["boards"],
        ),
        device=device,
        warmup=args.warmup,
    )
    results["variants"].append(full_csmp)

    if not args.skip_compile:
        can_compile, reason = _can_use_compile_backend(device, args.compile_backend)
        if can_compile:
            _reset_dynamo_state()
            compiled_fixed = _build_compile_wrapper(
                attention,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=args.compile_dynamic,
            )
            results["variants"].append(
                _benchmark(
                    label="compile / fixed batch / cached masks",
                    payloads=fixed_with_masks,
                    fn=lambda payload: compiled_fixed(
                        payload["x"],
                        head_masks=payload["head_masks"],
                    ),
                    device=device,
                    warmup=args.warmup,
                    capture_compile=True,
                )
            )

            _reset_dynamo_state()
            compiled_varying = _build_compile_wrapper(
                attention,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=args.compile_dynamic,
            )
            results["variants"].append(
                _benchmark(
                    label="compile / varying batch / cached masks",
                    payloads=varying_with_masks,
                    fn=lambda payload: compiled_varying(
                        payload["x"],
                        head_masks=payload["head_masks"],
                    ),
                    device=device,
                    warmup=args.warmup,
                    capture_compile=True,
                )
            )

            _reset_dynamo_state()
            compiled_csmp_fixed = _build_compile_wrapper(
                csmp,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=args.compile_dynamic,
            )
            results["variants"].append(
                _benchmark(
                    label="compile / fixed batch / full CSMP.forward",
                    payloads=fixed_csmp_payloads,
                    fn=lambda payload: compiled_csmp_fixed(
                        payload["cnn_tap_features"],
                        payload["boards"],
                    ),
                    device=device,
                    warmup=args.warmup,
                    capture_compile=True,
                )
            )

            _reset_dynamo_state()
            compiled_csmp_varying = _build_compile_wrapper(
                csmp,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=args.compile_dynamic,
            )
            results["variants"].append(
                _benchmark(
                    label="compile / varying batch / full CSMP.forward",
                    payloads=varying_csmp_payloads,
                    fn=lambda payload: compiled_csmp_varying(
                        payload["cnn_tap_features"],
                        payload["boards"],
                    ),
                    device=device,
                    warmup=args.warmup,
                    capture_compile=True,
                )
            )
        else:
            results["compile_skipped"] = reason

    rebuild_slowdown = _slowdown_factor(cached_attention, rebuild_every_time)
    notes: List[str] = []
    if rebuild_slowdown is not None:
        notes.append(f"rebuilding masks every call vs cached head masks: {rebuild_slowdown:.2f}x")

    mask_total_ms = _steady_ms(mask_build_only)
    ray_ms = _steady_ms(ray_only)
    attack_ms = _steady_ms(attack_only)
    head_ms = _steady_ms(head_assembly_only)
    if mask_total_ms > 0:
        notes.append(
            "approx steady-state mask breakdown: "
            f"ray={ray_ms / mask_total_ms * 100:.0f}%  "
            f"attack={attack_ms / mask_total_ms * 100:.0f}%  "
            f"head_assembly={head_ms / mask_total_ms * 100:.0f}%"
        )

    full_csmp_ms = _steady_ms(full_csmp)
    if full_csmp_ms > 0 and mask_total_ms > 0:
        notes.append(
            "approx dynamic-mask share of full CSMP.forward steady state: "
            f"{mask_total_ms / full_csmp_ms * 100:.0f}%"
        )

    if notes:
        results["notes"] = notes

    return results


def _print_case(case_result: Dict[str, Any]) -> None:
    title = case_result["case"].replace("_", " ")
    print(f"== {title} ==")
    for variant in case_result.get("variants", []):
        _print_variant(variant)
    for note in case_result.get("notes", []):
        print(f"  note: {note}")
    if case_result.get("compile_skipped"):
        print(f"  compile skipped: {case_result['compile_skipped']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Chess-Fusion attention paths.")
    parser.add_argument(
        "--case",
        choices=("all", "manual_mha", "chess_structure"),
        default="all",
        help="Attention case to benchmark.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to benchmark on.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Tensor dtype used for attention tensors.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for ManualMultiHeadAttention benchmarks.",
    )
    parser.add_argument(
        "--batch-schedule",
        default="8,8,8,16,16,8",
        help="Comma-separated batch sizes for ChessStructureAttention compile tests.",
    )
    parser.add_argument(
        "--seq-lens",
        default="128,128,128,256,256,128",
        help="Comma-separated sequence lengths for ManualMultiHeadAttention compile tests.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=512,
        help="Embedding dimension for benchmark modules.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--csmp-layers",
        type=int,
        default=4,
        help="Number of layers for the full ChessStructureMP benchmark.",
    )
    parser.add_argument(
        "--csmp-dim",
        type=int,
        default=0,
        help="Internal CSMP dimension for the full-module benchmark (0 = reuse --embed-dim).",
    )
    parser.add_argument(
        "--csmp-output-dim",
        type=int,
        default=0,
        help="Output dim for the full-module benchmark (0 = reuse --embed-dim).",
    )
    parser.add_argument(
        "--csmp-pos-dim",
        type=int,
        default=32,
        help="Positional embedding dimension for the full-module benchmark.",
    )
    parser.add_argument(
        "--csmp-piece-dim",
        type=int,
        default=64,
        help="Piece embedding dimension for the full-module benchmark.",
    )
    parser.add_argument(
        "--csmp-ffn-mult",
        type=int,
        default=2,
        help="FFN multiplier for the full-module benchmark.",
    )
    parser.add_argument(
        "--csmp-num-taps",
        type=int,
        default=0,
        help="Number of synthetic CNN taps for the full-module benchmark.",
    )
    parser.add_argument(
        "--csmp-cnn-dim",
        type=int,
        default=256,
        help="Channel dimension for each synthetic CNN tap.",
    )
    parser.add_argument(
        "--csmp-cnn-proj-dim",
        type=int,
        default=0,
        help="Optional CNN taper projection dim for the full-module benchmark (0 = disabled).",
    )
    parser.add_argument(
        "--csmp-use-xy-coords",
        action="store_true",
        help="Enable normalized XY coordinate features in the full-module benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of steps ignored for steady-state statistics.",
    )
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        help="Backend passed to torch.compile.",
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        help="Mode passed to torch.compile. Use 'none' to omit the mode kwarg.",
    )
    parser.add_argument(
        "--compile-dynamic",
        action="store_true",
        help="Pass dynamic=True to torch.compile.",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip torch.compile benchmarks and only run eager timing.",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional path to write the full profiling result as JSON.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    _print_environment(
        device=device,
        dtype=dtype,
        backend=args.compile_backend,
        mode=args.compile_mode,
        dynamic=args.compile_dynamic,
    )

    results: Dict[str, Any] = {
        "environment": _device_summary(device, dtype),
        "compile_backend": args.compile_backend,
        "compile_mode": args.compile_mode,
        "compile_dynamic": args.compile_dynamic,
        "cases": [],
    }

    selected_cases = []
    if args.case in ("all", "manual_mha"):
        selected_cases.append(_run_manual_mha_case)
    if args.case in ("all", "chess_structure"):
        selected_cases.append(_run_chess_structure_case)

    for case_fn in selected_cases:
        case_result = case_fn(args, device, dtype)
        results["cases"].append(case_result)
        _print_case(case_result)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"JSON output written to: {output_path}")


if __name__ == "__main__":
    main()
