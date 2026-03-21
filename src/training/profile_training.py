"""
Standalone profiling script for Chess-LM training.

Builds the model from config, runs a fixed number of forward+backward steps using
synthetic data (no dataset required), and prints a timing breakdown table.
Optionally exports a chrome trace via torch.profiler.

Usage:
    python -m training.profile_training --config configs/pretrain_lm_local.yaml
    python -m training.profile_training --config configs/pretrain_lm_local.yaml \\
        --steps 20 --output profiler_output --no-compile

Output:
    - Timing table printed to stdout
    - Chrome trace in <output>/ (open with chrome://tracing or TensorBoard)
"""

import argparse
import time
import os
import sys
from pathlib import Path
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional

import torch

# Ensure src/ is on the path when running as a script
_src = str(Path(__file__).parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
        return
    yield


def make_synthetic_batch(
    config,
    device: torch.device,
    batch_size: Optional[int] = None,
    seq_len: int = 256,
) -> dict:
    """Build a random batch that matches the shape expected by ChessCommentaryModel."""
    B = batch_size or config.batch_size
    mode = config.model.mode
    is_policy_only = mode == "policy_only"

    batch: dict = {}

    # Maia board features (always needed for chess_fusion / policy_only)
    if mode in ("chess_fusion", "policy_only"):
        batch["maia_features"] = torch.zeros(B, 18, 8, 8, device=device)
        batch["side_to_move"] = torch.ones(B, dtype=torch.long, device=device)

    if not is_policy_only:
        # Text sequence tensors
        vocab_size = 32000  # TinyLlama tokenizer vocab
        batch["input_ids"] = torch.randint(0, vocab_size, (B, seq_len), device=device)
        batch["attention_mask"] = torch.ones(B, seq_len, dtype=torch.long, device=device)
        labels = batch["input_ids"].clone()
        labels[:, :10] = -100  # mask prefix tokens (simulate position prefix)
        batch["labels"] = labels
        batch["fen"] = ["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"] * B
        batch["lc0_hidden_states"] = {}

    return batch


def _fmt_row(name: str, avg_ms: float, total_ms: float, indent: int = 0) -> str:
    pct = (avg_ms / total_ms * 100) if total_ms > 0 else 0.0
    prefix = "  " * indent + ("└─ " if indent > 0 else "")
    return f"  {prefix}{name:<32} {avg_ms:>8.1f} ms    {pct:>5.1f}%"


def print_timing_table(timings: List[Dict], config) -> None:
    """Print a human-readable profiling summary."""
    if not timings:
        print("  (no profiling data collected)")
        return

    keys = list(timings[0].keys())

    def avg(key):
        vals = [t[key] for t in timings if key in t and t[key] > 0]
        return sum(vals) / len(vals) if vals else 0.0

    step_ms   = avg("step_ms")
    data_ms   = avg("data_ms") * 1000
    fwd_ms    = avg("fwd_ms") * 1000
    bwd_ms    = avg("bwd_ms") * 1000
    opt_ms    = avg("opt_ms") * 1000

    ms_multi  = avg("adapter_multiscale_ms")
    ms_csmp   = avg("csmp_ms")
    ms_perc   = avg("adapter_perceiver_ms")
    ms_bsr    = avg("bsr_spp_ms")
    ms_xattn  = avg("llm_xattn_ms")
    ms_orig   = avg("llm_orig_ms")

    total_ms = step_ms if step_ms > 0 else (data_ms + fwd_ms + bwd_ms + opt_ms)

    print("\n" + "=" * 65)
    print("  PROFILING SUMMARY")
    print("=" * 65)
    print(f"  {'Component':<32} {'Avg (ms)':>8}     {'% Step':>6}")
    print("-" * 65)
    print(_fmt_row("data_loading",  data_ms,  total_ms))
    print(_fmt_row("forward_pass",  fwd_ms,   total_ms))
    if ms_multi > 0:
        print(_fmt_row("adapter / multi_scale", ms_multi, total_ms, indent=1))
        if ms_csmp > 0:
            print(_fmt_row("csmp",   ms_csmp,  total_ms, indent=2))
        if ms_perc > 0:
            print(_fmt_row("perceiver", ms_perc, total_ms, indent=1))
        if ms_bsr > 0:
            print(_fmt_row("bsr_spp", ms_bsr, total_ms, indent=1))
    if ms_xattn > 0:
        print(_fmt_row("llm_self_attn", ms_orig,  total_ms, indent=1))
        print(_fmt_row("llm_xattn",     ms_xattn, total_ms, indent=1))
    print(_fmt_row("backward_pass", bwd_ms,   total_ms))
    print(_fmt_row("optimizer",     opt_ms,   total_ms))
    print("-" * 65)
    print(_fmt_row("TOTAL STEP",    total_ms, total_ms))
    print("=" * 65)

    # Memory
    mem_fwd = avg("mem_fwd_mb")
    mem_bwd = avg("mem_bwd_mb")
    if mem_fwd > 0 or mem_bwd > 0:
        print(f"\n  Peak GPU memory:")
        print(f"    Forward:  {mem_fwd:.0f} MB")
        print(f"    Backward: {mem_bwd:.0f} MB")

    # Throughput
    tokens_per_sec = avg("tokens_per_sec")
    mfu = avg("mfu")
    if tokens_per_sec > 0:
        print(f"\n  Throughput:")
        print(f"    {tokens_per_sec:,.0f} real tokens/sec")
        if mfu > 0:
            print(f"    MFU: {mfu*100:.1f}%  (GPU peak TFLOPs = {config.profiling.gpu_peak_tflops})")
    print()


def run_profiling(
    config_path: str,
    n_steps: int = 15,
    output_dir: str = "profiler_output",
    batch_size: Optional[int] = None,
    seq_len: int = 256,
    no_compile: bool = True,
    emit_chrome_trace: bool = True,
) -> None:
    from training.config import load_config
    from training.train import ChessCommentaryModel, PolicyOnlyModel

    config = load_config(config_path)

    use_amp = (config.fp16 or config.bf16) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.fp16 and torch.cuda.is_available()))
    amp_ctx = torch.amp.autocast('cuda', dtype=amp_dtype) if use_amp else nullcontext()

    # Force profiling on
    prof = config.profiling
    prof.enabled = True
    prof.cuda_event_timing = True
    prof.csmp_timing = True
    prof.memory_snapshots = True
    prof.emit_interval = 1  # profile every step
    prof.nvtx_ranges = True
    prof.torch_profiler = emit_chrome_trace
    prof.torch_profiler_wait = 2
    prof.torch_profiler_warmup = 2
    prof.torch_profiler_active = max(n_steps - 4, 1)
    prof.torch_profiler_output_dir = output_dir
    prof.log_to_wandb = False

    # Disable compile for readable traces
    if no_compile:
        config.model.use_torch_compile = False
        print("[Profiler] torch.compile disabled for readable operator-level traces.")

    # Disable wandb
    config.use_wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Profiler] Device: {device}")
    print(f"[Profiler] Mode:   {config.model.mode}")
    print(f"[Profiler] Steps:  {n_steps} ({2} warmup + {n_steps - 2} profiled)")

    # Build model
    print("[Profiler] Building model...")
    is_policy_only = config.model.mode == "policy_only"
    if is_policy_only:
        model = PolicyOnlyModel(config.model, torch_dtype=amp_dtype)
    else:
        model = ChessCommentaryModel(config.model, torch_dtype=amp_dtype)
    model.to(device)
    model.train()
    if config.gradient_checkpointing and hasattr(model, 'llm') and hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()
        print("[Profiler] Gradient checkpointing enabled.")
    print(f"[Profiler] Model built.")

    # Build optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    # Set up torch.profiler
    _torch_prof = None
    if emit_chrome_trace:
        import torch.profiler as tprof
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        _torch_prof = tprof.profile(
            activities=[tprof.ProfilerActivity.CPU, tprof.ProfilerActivity.CUDA],
            schedule=tprof.schedule(wait=2, warmup=2, active=max(n_steps - 4, 1), repeat=1),
            on_trace_ready=tprof.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        _torch_prof.__enter__()

    timings = []
    _n_params_llm = 1.1e9
    _gpu_peak_flops = prof.gpu_peak_tflops * 1e12

    use_cuda_events = device.type == "cuda"
    use_nvtx = bool(prof.nvtx_ranges and device.type == "cuda")
    if use_nvtx:
        print("[Profiler] NVTX ranges enabled (forward, backward, optimizer).")

    for step in range(n_steps):
        batch = make_synthetic_batch(config, device, batch_size=batch_size, seq_len=seq_len)

        # Set profiling flags on model
        _tgt = model
        _do_profile = step >= 2  # skip first 2 warmup steps
        if hasattr(_tgt, 'adapter'):
            _tgt.adapter._profile = _do_profile
            if hasattr(_tgt.adapter, 'multi_scale') and hasattr(_tgt.adapter.multi_scale, 'chess_mp'):
                if _tgt.adapter.multi_scale.chess_mp is not None:
                    _tgt.adapter.multi_scale.chess_mp._profile = _do_profile
        _tgt._profile_forward = _do_profile

        step_start = time.perf_counter()

        # CUDA event timing
        if use_cuda_events and _do_profile:
            torch.cuda.reset_peak_memory_stats()  # reset before step so we measure step-local peaks
            ev_fwd_s = torch.cuda.Event(enable_timing=True)
            ev_fwd_e = torch.cuda.Event(enable_timing=True)
            ev_bwd_s = torch.cuda.Event(enable_timing=True)
            ev_bwd_e = torch.cuda.Event(enable_timing=True)
            ev_fwd_s.record()

        # Forward
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        with _nvtx_range("forward", use_nvtx):
            with amp_ctx:
                if is_policy_only:
                    outputs = model(
                        maia_features=batch["maia_features"],
                        side_to_move=batch["side_to_move"],
                    )
                else:
                    outputs = model(
                        lc0_hidden_states=batch.get("lc0_hidden_states", {}),
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        side_to_move=batch.get("side_to_move"),
                        fen=batch.get("fen"),
                        maia_features=batch.get("maia_features"),
                    )
                loss = outputs.loss

        if use_cuda_events and _do_profile:
            ev_fwd_e.record()
            torch.cuda.synchronize()
            fwd_ms = ev_fwd_s.elapsed_time(ev_fwd_e) / 1000.0
            ev_bwd_s.record()
        else:
            fwd_ms = time.perf_counter() - t0

        mem_fwd_mb = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda_events else 0.0
        if use_cuda_events:
            torch.cuda.reset_peak_memory_stats()

        # Backward
        t0 = time.perf_counter()
        with _nvtx_range("backward", use_nvtx):
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if use_cuda_events and _do_profile:
            ev_bwd_e.record()
            torch.cuda.synchronize()
            bwd_ms = ev_bwd_s.elapsed_time(ev_bwd_e) / 1000.0
        else:
            bwd_ms = time.perf_counter() - t0

        mem_bwd_mb = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda_events else 0.0

        t0 = time.perf_counter()
        with _nvtx_range("optimizer", use_nvtx):
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        opt_ms = time.perf_counter() - t0

        step_ms = (time.perf_counter() - step_start) * 1000

        tokens_per_sec = seq_len * (batch_size or config.batch_size) / max(fwd_ms + bwd_ms, 1e-6)
        mfu = (6 * _n_params_llm * tokens_per_sec) / max(_gpu_peak_flops, 1.0) if _gpu_peak_flops > 0 else 0.0

        if _do_profile:
            entry = {
                "step": step,
                "step_ms": step_ms,
                "data_ms": 0.0,
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "opt_ms": opt_ms,  # seconds, consistent with fwd_ms/bwd_ms
                "mem_fwd_mb": mem_fwd_mb,
                "mem_bwd_mb": mem_bwd_mb,
                "tokens_per_sec": tokens_per_sec,
                "mfu": mfu,
                "adapter_multiscale_ms": 0.0,
                "csmp_ms": 0.0,
                "adapter_perceiver_ms": 0.0,
                "bsr_spp_ms": 0.0,
                "llm_xattn_ms": 0.0,
                "llm_orig_ms": 0.0,
            }
            # Read adapter/CSMP/xAttn timings
            _adapter = getattr(model, 'adapter', None)
            if _adapter is not None and hasattr(_adapter, '_last_profile_ms'):
                pm = _adapter._last_profile_ms
                entry["adapter_multiscale_ms"] = pm.get("multi_scale", 0.0)
                entry["csmp_ms"]               = pm.get("csmp", 0.0)
                entry["adapter_perceiver_ms"]  = pm.get("perceiver", 0.0)
                entry["bsr_spp_ms"]            = pm.get("bsr_spp", 0.0)
            try:
                from training.chess_fusion_model import FusionDecoderLayer as _FDL
                _xl = _FDL._profile_xattn_ms
                _ol = _FDL._profile_original_ms
                entry["llm_xattn_ms"] = sum(_xl) if isinstance(_xl, list) else _xl
                entry["llm_orig_ms"]  = sum(_ol) if isinstance(_ol, list) else _ol
            except Exception:
                pass
            timings.append(entry)

        if _torch_prof is not None:
            _torch_prof.step()

        status = f"  step {step+1:>3}/{n_steps}"
        if _do_profile:
            status += f"  fwd={fwd_ms*1000:.0f}ms  bwd={bwd_ms*1000:.0f}ms  loss={loss.item():.4f}"
        print(status)

    if _torch_prof is not None:
        _torch_prof.__exit__(None, None, None)
        print(f"\n[Profiler] Chrome trace written to: {output_dir}/")
        print(f"[Profiler] View with: tensorboard --logdir={output_dir}")

    print_timing_table(timings, config)


def main():
    parser = argparse.ArgumentParser(description="Profile Chess-LM training (no dataset required)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--steps", type=int, default=15,
                        help="Total steps to run (first 2 are warmup, rest are profiled)")
    parser.add_argument("--output", default="profiler_output",
                        help="Output directory for chrome trace (default: profiler_output)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Synthetic sequence length (default: 256)")
    parser.add_argument("--no-compile", action="store_true", default=True,
                        help="Disable torch.compile for readable operator traces (default: True)")
    parser.add_argument("--no-trace", action="store_true", default=False,
                        help="Skip chrome trace export (timing table only)")
    args = parser.parse_args()

    run_profiling(
        config_path=args.config,
        n_steps=args.steps,
        output_dir=args.output,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        no_compile=args.no_compile,
        emit_chrome_trace=not args.no_trace,
    )


if __name__ == "__main__":
    main()

