#!/usr/bin/env python3
"""
Download the latest model checkpoint from a W&B run.

Usage:
    python download_checkpoint.py <entity/project/run_id_or_name>
    python download_checkpoint.py <entity/project/run_id_or_name> --output ./checkpoints
    python src\inference\download_checkpoint.py theswizzles123/chess-commentary/owtjxe4y --output ./cp_chess_fusion1   
    python download_checkpoint.py theswizzles123/chess-commentary/dbo0q37l --output ./cp_maia-perceiver-finetune
    
Examples:
    python download_checkpoint.py myuser/chess-commentary/abc123
    python download_checkpoint.py theswizzles123/chess-commentary-cloud/maia-perceiver-finetune --output ./cp_maia-perceiver-finetune
    python download_checkpoint.py theswizzles123/chess-commentary-cloud_engineered_v0/generous-eon-107 --output ./cp_generous-eon-107
    python download_checkpoint.py myuser/chess-commentary/abc123 -o ./my_checkpoint
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

try:
    import wandb
except ImportError:
    print("Error: wandb is not installed. Install it with: pip install wandb")
    sys.exit(1)


def resolve_run(api, entity: str, project: str, run_identifier: str):
    """
    Resolve a run by ID or display name.
    
    Args:
        api: wandb.Api instance
        entity: wandb entity (username or team)
        project: wandb project name
        run_identifier: Either run ID or display name
    
    Returns:
        wandb.Run object
    """
    # First, try to fetch by ID directly
    try:
        run = api.run(f"{entity}/{project}/{run_identifier}")
        return run
    except wandb.errors.CommError:
        pass  # Not found by ID, try by name
    
    # Search for run by display name
    print(f"Run ID not found, searching by name: {run_identifier}")
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": run_identifier}
    )
    runs_list = list(runs)
    
    if not runs_list:
        # Also try partial match
        all_runs = api.runs(f"{entity}/{project}")
        matching = [r for r in all_runs if run_identifier in r.name]
        if matching:
            print(f"Found {len(matching)} run(s) matching '{run_identifier}':")
            for r in matching[:10]:
                print(f"  - {r.name} (ID: {r.id})")
            if len(matching) == 1:
                return matching[0]
            else:
                raise ValueError(
                    f"Multiple runs match '{run_identifier}'. "
                    "Please use the exact run ID."
                )
        raise ValueError(
            f"Could not find run with name or ID: {run_identifier}\n"
            f"Use 'wandb runs list {entity}/{project}' to see available runs."
        )
    
    if len(runs_list) > 1:
        print(f"Found {len(runs_list)} runs with name '{run_identifier}':")
        for r in runs_list:
            print(f"  - {r.name} (ID: {r.id}, state: {r.state})")
        # Return the most recent one
        return runs_list[0]
    
    return runs_list[0]


def get_run_artifacts(run, artifact_type: str = "model") -> list:
    """Get all artifacts of a specific type for a run object."""
    artifacts = []
    for artifact in run.logged_artifacts():
        if artifact.type == artifact_type:
            # Extract the base name (without version suffix that wandb adds)
            # artifact.name might be like "epoch-1-checkpoint:v2"
            base_name = artifact.name.split(":")[0] if ":" in artifact.name else artifact.name
            
            artifacts.append({
                "name": base_name,
                "version": f"v{artifact.version}" if not str(artifact.version).startswith("v") else artifact.version,
                "created_at": artifact.created_at,
                "description": artifact.description,
                "metadata": artifact.metadata,
                # Use the artifact's qualified_name property for downloading
                "qualified_name": artifact.qualified_name,
            })
    
    return artifacts


def _sanitize_run_config(config: dict) -> dict:
    """Remove wandb-internal keys and return plain config values."""
    return {k: v for k, v in config.items() if not str(k).startswith("_")}


def save_run_config(run, output_path: Path) -> None:
    """Save the run config to config.yaml next to the checkpoint."""
    config_path = output_path / "config.yaml"
    config_data = _sanitize_run_config(dict(run.config))
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, sort_keys=False)
    print(f"Saved run config to: {config_path}")


def download_latest_checkpoint(
    run_path: str,
    output_dir: str = "./checkpoints",
    artifact_name: str = None,
    artifact_type: str = "model",
    exclude_merged_base: bool = False,
    skip_merged_base_download: bool = False,
) -> Path:
    """
    Download the latest model checkpoint from a W&B run.
    
    Args:
        run_path: W&B run path in format "entity/project/run_id_or_name"
        output_dir: Directory to download checkpoint to
        artifact_name: Specific artifact name to download (optional)
        artifact_type: Type of artifact to look for (default: "model")
    
    Returns:
        Path to downloaded checkpoint directory
    """
    api = wandb.Api()
    
    # Parse run path
    parts = run_path.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid run path: {run_path}\n"
            "Expected format: entity/project/run_id_or_name"
        )
    
    entity, project, run_identifier = parts
    
    print(f"Fetching run: {run_path}")
    run = resolve_run(api, entity, project, run_identifier)
    print(f"Run name: {run.name}")
    print(f"Run ID: {run.id}")
    print(f"Run state: {run.state}")
    
    # Get artifacts logged by this run (use the resolved run object directly)
    artifacts = get_run_artifacts(run, artifact_type)
    
    if not artifacts:
        print(f"\nNo artifacts of type '{artifact_type}' found for this run.")
        print("Available artifact types in this run:")
        seen_types = set()
        for artifact in run.logged_artifacts():
            if artifact.type not in seen_types:
                print(f"  - {artifact.type}")
                seen_types.add(artifact.type)
        sys.exit(1)
    
    print(f"\nFound {len(artifacts)} checkpoint(s):")
    for i, art in enumerate(artifacts):
        metadata_str = ""
        if art["metadata"]:
            epoch = art["metadata"].get("epoch", "?")
            loss = art["metadata"].get("loss", "?")
            if isinstance(loss, float):
                loss = f"{loss:.4f}"
            metadata_str = f" (epoch: {epoch}, loss: {loss})"
        print(f"  [{i+1}] {art['name']}:{art['version']}{metadata_str}")
    
    # Select artifact to download
    if artifact_name:
        # Find specific artifact by name
        matching = [a for a in artifacts if artifact_name in a["name"]]
        if not matching:
            print(f"\nNo artifact matching '{artifact_name}' found.")
            sys.exit(1)
        selected = matching[-1]  # Take latest version if multiple
    else:
        # Take the latest (last logged) artifact
        selected = artifacts[-1]
    
    print(f"\nDownloading: {selected['name']}:{selected['version']}")
    
    # Download the artifact using qualified_name
    artifact = api.artifact(selected['qualified_name'])
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download to output directory
    if skip_merged_base_download:
        safe_name = f"{selected['name']}-{selected['version']}"
        download_path = output_path / safe_name
        download_path.mkdir(parents=True, exist_ok=True)
        entries = artifact.manifest.entries
        skip_count = 0
        download_entries = []
        for entry_name in entries.keys():
            if entry_name.startswith("merged_base/"):
                skip_count += 1
                continue
            download_entries.append(entry_name)
        print(f"Skipping {skip_count} merged_base files, downloading {len(download_entries)} files...")
        for i, entry_name in enumerate(download_entries, 1):
            print(f"  [{i}/{len(download_entries)}] {entry_name}")
            artifact.get_entry(entry_name).download(root=str(download_path))
    else:
        download_path = Path(artifact.download(root=str(output_path)))

    if exclude_merged_base:
        merged_base_path = download_path / "merged_base"
        if merged_base_path.exists():
            print("Removing merged_base from downloaded checkpoint")
            shutil.rmtree(merged_base_path)
    
    print(f"\n✓ Checkpoint downloaded to: {download_path}")
    
    # List downloaded files
    print("\nDownloaded files:")
    for item in sorted(download_path.rglob("*")):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            rel_path = item.relative_to(download_path)
            print(f"  {rel_path} ({size_mb:.2f} MB)")
    
    save_run_config(run, download_path)
    return download_path


def main():
    parser = argparse.ArgumentParser(
        description="Download the latest model checkpoint from a W&B run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s myuser/chess-commentary/abc123
  %(prog)s myuser/chess-commentary/dulcet-spaceship-22
  %(prog)s myuser/chess-commentary/abc123 --output ./my_checkpoint
  %(prog)s myuser/chess-commentary/abc123 --artifact-name epoch-5-checkpoint
  %(prog)s myuser/chess-commentary/abc123 --list
        """
    )
    
    parser.add_argument(
        "run_path",
        help="W&B run path in format: entity/project/run_id_or_name"
    )
    parser.add_argument(
        "-o", "--output",
        default="./downloaded_checkpoint",
        help="Output directory for downloaded checkpoint (default: ./downloaded_checkpoint)"
    )
    parser.add_argument(
        "-a", "--artifact-name",
        help="Specific artifact name to download (default: latest)"
    )
    parser.add_argument(
        "-t", "--artifact-type",
        default="model",
        help="Artifact type to look for (default: model)"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        dest="list_only",
        help="List available checkpoints without downloading"
    )
    parser.add_argument(
        "--exclude-merged-base",
        action="store_true",
        help="Remove merged_base after download to save space"
    )
    parser.add_argument(
        "--skip-merged-base-download",
        action="store_true",
        help="Skip downloading merged_base files to save time"
    )
    
    args = parser.parse_args()
    
    if args.list_only:
        # Resolve run first, then list
        api = wandb.Api()
        parts = args.run_path.split("/")
        if len(parts) != 3:
            print(f"Invalid run path: {args.run_path}")
            print("Expected format: entity/project/run_id_or_name")
            sys.exit(1)
        
        entity, project, run_identifier = parts
        run = resolve_run(api, entity, project, run_identifier)
        print(f"Run: {run.name} (ID: {run.id})")
        
        artifacts = get_run_artifacts(run, args.artifact_type)
        if artifacts:
            print(f"\nAvailable checkpoints:")
            for i, art in enumerate(artifacts):
                metadata_str = ""
                if art["metadata"]:
                    epoch = art["metadata"].get("epoch", "?")
                    loss = art["metadata"].get("loss", "?")
                    if isinstance(loss, float):
                        loss = f"{loss:.4f}"
                    metadata_str = f" (epoch: {epoch}, loss: {loss})"
                print(f"  [{i+1}] {art['name']}:{art['version']}{metadata_str}")
        else:
            print(f"No artifacts of type '{args.artifact_type}' found.")
    else:
        # Download checkpoint
        download_latest_checkpoint(
            run_path=args.run_path,
            output_dir=args.output,
            artifact_name=args.artifact_name,
            artifact_type=args.artifact_type,
            exclude_merged_base=args.exclude_merged_base,
            skip_merged_base_download=args.skip_merged_base_download,
        )


if __name__ == "__main__":
    main()

