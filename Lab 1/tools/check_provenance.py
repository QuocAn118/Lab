"""Kiểm tra lưu vết của Lab 1: nguồn gốc file, header code, manifest các run.

Spec: docs/team_plan_integrity.md §9

Lệnh (chạy từ bất kỳ thư mục nào):
    python "Lab 1/tools/check_provenance.py"                 # kiểm tra, lỗi -> exit 1
    python "Lab 1/tools/check_provenance.py" --strict        # coi cảnh báo "nguồn đã đổi" là lỗi
    python "Lab 1/tools/check_provenance.py" --ack docs/requirements/3_4_train.md --by TenBan --note "đã rà theo §6 mới"
    python "Lab 1/tools/check_provenance.py" --ack all --by TenBan --note "..."
    python "Lab 1/tools/check_provenance.py" --add src/model.py --source "docs/requirements/3_3_build_model.md" \
        --by TenBan --method ai-assisted --note "Sinh bằng <công cụ AI>, TV2 review"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_tracking import (  # noqa: E402
    CHECKPOINTS, LAB_ROOT, MANIFEST, OUTPUTS, REGISTRY, RUN_ID_RE, SIDECAR_SUFFIX,
    _ignored_by_git, now_iso, rel, sha256_file, write_json,
)

# Các file bắt buộc phải có trong sổ đăng ký nguồn gốc.
COVERED_GLOBS = ["README.md", ".gitignore", "docs/**/*.md", "src/*.py", "tools/*.py"]
HEADER_LINES = 15


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def error(self, msg): self.errors.append(msg)
    def warn(self, msg): self.warnings.append(msg)
    def info(self, msg): self.infos.append(msg)


def load_registry() -> dict:
    if not REGISTRY.is_file():
        return {"schema_version": 1, "entries": []}
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def save_registry(registry: dict) -> None:
    registry["entries"].sort(key=lambda e: e["path"])
    write_json(REGISTRY, registry)


# ---------------------------------------------------------------- kiểm tra
def check_registry(registry: dict, rep: Report) -> None:
    entries = {e["path"]: e for e in registry.get("entries", [])}
    for path, entry in entries.items():
        if not (LAB_ROOT / path).is_file():
            rep.error(f"[nguồn gốc] {path}: có trong sổ đăng ký nhưng file không tồn tại")
            continue
        if not entry.get("sources"):
            rep.error(f"[nguồn gốc] {path}: chưa ghi nguồn (sources)")
        for src in entry.get("sources", []):
            if "external" in src:
                continue
            spath = LAB_ROOT / src["path"]
            if not spath.is_file():
                rep.error(f"[nguồn gốc] {path}: nguồn {src['path']} không tồn tại")
            elif src.get("sha256") != sha256_file(spath):
                rep.warn(f"[nguồn đã đổi] {path}: {src['path']} thay đổi sau lần rà gần nhất. "
                         f"Rà lại rồi chạy --ack {path}")

    covered = set()
    for pattern in COVERED_GLOBS:
        covered.update(rel(p) for p in LAB_ROOT.glob(pattern) if p.is_file())
    for path in sorted(covered - set(entries)):
        rep.error(f"[nguồn gốc] {path}: chưa đăng ký. Dùng --add {path} --source ...")

    for path, entry in entries.items():
        if not path.endswith(".py") or not (LAB_ROOT / path).is_file():
            continue
        head = (LAB_ROOT / path).read_text(encoding="utf-8").splitlines()[:HEADER_LINES]
        header_specs = {line.split("Spec:", 1)[1].split("§")[0].strip()
                        for line in head if "Spec:" in line}
        for src in entry.get("sources", []):
            if "path" in src and src["path"] not in header_specs:
                rep.error(f"[header] {path}: thiếu dòng 'Spec: {src['path']}' trong {HEADER_LINES} dòng đầu")


def check_runs(rep: Report) -> None:
    run_dirs = {p.name for p in OUTPUTS.iterdir() if p.is_dir()} if OUTPUTS.is_dir() else set()
    ckpt_dirs = {p.name for p in CHECKPOINTS.iterdir() if p.is_dir()} if CHECKPOINTS.is_dir() else set()
    for name in sorted(ckpt_dirs - run_dirs):
        rep.error(f"[run] checkpoints/{name}: không có outputs/{name}/manifest.json tương ứng")

    for name in sorted(run_dirs):
        out = OUTPUTS / name
        if not RUN_ID_RE.match(name):
            rep.error(f"[run] outputs/{name}: tên thư mục sai mẫu run_xxx / smoke_xxx")
            continue
        mpath = out / MANIFEST
        if not mpath.is_file():
            rep.error(f"[run] {name}: thiếu manifest.json — run không được tạo bằng run_tracking.Run")
            continue
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        status = manifest.get("status")
        if status == "running":
            rep.warn(f"[run] {name}: trạng thái 'running' — run bị dừng đột ngột hoặc đang chạy")
        elif status in ("failed", "interrupted"):
            rep.info(f"[run] {name}: {status} (vẫn được giữ làm vết)")
        if manifest.get("git", {}).get("dirty"):
            rep.info(f"[run] {name}: chạy khi code chưa commit (allow_dirty={manifest.get('allow_dirty')})")
        for code in manifest.get("code", []):
            if not code.get("registered"):
                rep.warn(f"[run] {name}: code {code['path']} lúc chạy chưa đăng ký nguồn gốc")

        listed = {f["path"]: f for f in manifest.get("files", [])}
        for path, f in listed.items():
            p = LAB_ROOT / path
            if not p.is_file():
                if f.get("ignored_by_git"):
                    rep.info(f"[run] {name}: {path} không có trên máy này (không lưu trên git)")
                else:
                    rep.error(f"[run] {name}: {path} có trong manifest nhưng đã mất")
            elif sha256_file(p) != f["sha256"]:
                rep.error(f"[run] {name}: {path} bị sửa sau khi run kết thúc (hash không khớp)")
        if status != "running":
            present = {rel(p) for d in (out, CHECKPOINTS / name) if d.is_dir()
                       for p in d.rglob("*") if p.is_file()}
            for path in sorted(present - set(listed) - {rel(mpath)}):
                rep.error(f"[run] {name}: {path} được thêm sau khi run kết thúc, không có trong manifest")


def check_artifacts(rep: Report) -> None:
    if not OUTPUTS.is_dir():
        return
    for p in sorted(OUTPUTS.iterdir()):
        if not p.is_file() or p.name.endswith(SIDECAR_SUFFIX) or p.name.endswith(".tmp"):
            continue
        sidecar = p.with_name(p.name + SIDECAR_SUFFIX)
        if not sidecar.is_file():
            rep.error(f"[đầu ra] {rel(p)}: thiếu {sidecar.name} — tạo bằng run_tracking.record_artifact")
            continue
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if data.get("sha256") != sha256_file(p):
            rep.error(f"[đầu ra] {rel(p)}: bị sửa sau khi tạo (hash không khớp {sidecar.name})")
    for s in sorted(OUTPUTS.glob("*" + SIDECAR_SUFFIX)):
        if not s.with_name(s.name[: -len(SIDECAR_SUFFIX)]).is_file():
            rep.error(f"[đầu ra] {rel(s)}: file gốc đã bị xóa")


# ---------------------------------------------------------------- cập nhật sổ đăng ký
def parse_source(text: str) -> dict:
    if text.startswith("external:"):
        return {"external": text.split(":", 1)[1].strip()}
    path, _, section = text.partition("#")
    path = path.strip()
    if not (LAB_ROOT / path).is_file():
        sys.exit(f"Nguồn không tồn tại: {path}")
    src = {"path": path, "sha256": sha256_file(LAB_ROOT / path)}
    if section:
        src["section"] = section.strip()
    return src


def cmd_add(registry, args) -> None:
    path = args.add
    if not (LAB_ROOT / path).is_file():
        sys.exit(f"File không tồn tại: {path}")
    if any(e["path"] == path for e in registry["entries"]):
        sys.exit(f"{path} đã đăng ký; dùng --ack để ghi lần rà mới")
    if not args.source:
        sys.exit("Cần ít nhất một --source")
    registry["entries"].append({
        "path": path,
        "sources": [parse_source(s) for s in args.source],
        "method": args.method,
        "created_by": args.by,
        "recorded_at": now_iso(),
        "status": "active",
        "note": args.note or "",
        "reviews": [],
    })
    save_registry(registry)
    print(f"Đã đăng ký {path}")


def cmd_ack(registry, args) -> None:
    targets = [e for e in registry["entries"] if args.ack in ("all", e["path"])]
    if not targets:
        sys.exit(f"Không tìm thấy {args.ack} trong sổ đăng ký")
    for entry in targets:
        changed = []
        for src in entry.get("sources", []):
            if "path" in src and (LAB_ROOT / src["path"]).is_file():
                new = sha256_file(LAB_ROOT / src["path"])
                if src.get("sha256") != new:
                    changed.append(src["path"])
                    src["sha256"] = new
        if changed or args.ack != "all":
            entry["reviews"].append({"at": now_iso(), "by": args.by, "sources_changed": changed,
                                     "note": args.note or ""})
            print(f"Đã ghi lần rà {entry['path']}: {', '.join(changed) or 'nguồn không đổi'}")
    save_registry(registry)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Kiểm tra lưu vết Lab 1")
    ap.add_argument("--strict", action="store_true", help="coi cảnh báo là lỗi")
    ap.add_argument("--add", metavar="PATH", help="đăng ký file mới (đường dẫn từ Lab 1)")
    ap.add_argument("--source", action="append", metavar="PATH[#mục] | external:mô tả")
    ap.add_argument("--method", default="manual", choices=["manual", "ai-assisted", "generated-by-script"])
    ap.add_argument("--ack", metavar="PATH|all", help="ghi nhận đã rà lại sau khi nguồn thay đổi")
    ap.add_argument("--by", help="người thực hiện (bắt buộc với --add/--ack)")
    ap.add_argument("--note")
    args = ap.parse_args()

    registry = load_registry()
    if args.add or args.ack:
        if not args.by:
            sys.exit("Cần --by <tên người thực hiện>")
        cmd_add(registry, args) if args.add else cmd_ack(registry, args)
        return 0

    rep = Report()
    check_registry(registry, rep)
    check_runs(rep)
    check_artifacts(rep)
    for m in rep.infos:
        print("INFO ", m)
    for m in rep.warnings:
        print("WARN ", m)
    for m in rep.errors:
        print("ERROR", m)
    failed = bool(rep.errors) or (args.strict and bool(rep.warnings))
    print(f"\n{'FAIL' if failed else 'PASS'} — {len(rep.errors)} lỗi, "
          f"{len(rep.warnings)} cảnh báo, {len(rep.infos)} thông tin")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
