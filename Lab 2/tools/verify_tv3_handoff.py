"""Read-only audit: exact historical sources, manifests and 12 LFS checkpoints.

This does not claim the current integrated source was used for historical training.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    reconciliation = json.loads((ROOT / "reports/tv3/source_reconciliation_2026-11-12.json").read_text(encoding="utf-8"))
    sources = 0
    for record in reconciliation["runs"]:
        manifest_path = ROOT / record["manifest"]
        allowed_manifest_hashes = {record["manifest_sha256"], record["published_manifest_sha256"]}
        if digest(manifest_path) not in allowed_manifest_hashes:
            raise ValueError(f"Historical manifest changed: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for source in record["sources"]:
            expected = manifest["sources"][source["original_path"]]
            if expected != source["sha256"] or digest(ROOT / source["snapshot"]) != expected:
                raise ValueError(f"Historical source mismatch: {source['snapshot']}")
            sources += 1
    index = json.loads((ROOT / "reports/tv3/checkpoint_check.json").read_text(encoding="utf-8"))
    count = 0
    for run, marks in index.items():
        for mark, info in marks.items():
            path = ROOT / "checkpoints" / run / (mark + ".pth")
            if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
                raise ValueError(f"Checkpoint missing/LFS pointer/corrupt: {path}")
            count += 1
    if count != 12:
        raise ValueError(f"Expected 12 checkpoints, found {count}")
    return {"status": "PASS", "historical_source_files": sources,
            "unchanged_manifests": len(reconciliation["runs"]), "checkpoints": count}


if __name__ == "__main__":
    result = verify()
    if "--load" in sys.argv:
        import torch
        sys.path.insert(0, str(ROOT / "src"))
        from lab_2.model import load_checkpoint
        result["loaded_best_checkpoints"] = {}
        for name in ("resnet18_baseline", "densenet121", "resnet18_lr1e-2", "resnet18_bs128"):
            model, info = load_checkpoint(ROOT / "checkpoints" / name / "best_val_acc.pth", "cpu")
            model.eval()
            with torch.no_grad():
                output = model(torch.zeros(1, 3, 224, 224))
            if output.shape != (1, 10) or not torch.isfinite(output).all():
                raise ValueError(f"Checkpoint incompatible with integrated model: {name}")
            result["loaded_best_checkpoints"][name] = {"epoch": info["epoch"], "output_shape": list(output.shape)}
    print(json.dumps(result, indent=2))
