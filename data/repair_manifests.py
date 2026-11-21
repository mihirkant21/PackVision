"""
PackVision AI - Dataset Manifest Repair Tool
=============================================
Cross-references the on-disk train/ and val/ triplet directories against
metadata.jsonl / train_manifest.jsonl / val_manifest.jsonl and rewrites the
manifests so that every complete on-disk triplet ({sample_id}_sketch.png,
{sample_id}_render.png, {sample_id}_depth.png) is included exactly once.

Policy:
- Known ids (present in the split manifests or metadata.jsonl) keep their
  existing record fields; only split/file-path fields are corrected to match
  the actual on-disk location.
- When the same sample_id has triplets in BOTH split directories (stale
  duplicates left behind by an earlier generator run with different file
  content), the copy in the directory matching the authoritative
  metadata.jsonl split is treated as primary, and the other copy is included
  under a disambiguated id "{sample_id}_stale" with a "stale_duplicate_of"
  provenance field and "seed": null (the original generation seed of the
  stale copy is unknown). This keeps every sample_id in exactly one split
  (no train/val id overlap) while still accounting for every on-disk triplet.
- Ids with no record anywhere are synthesized with "category": "unknown"
  (the pv_pkg_NNNN id scheme embeds no category/material/finish information).

Dry-run by default. Pass --write to actually rewrite the manifest files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUFFIXES = ("_sketch.png", "_render.png", "_depth.png")

FIELD_ORDER = [
    "sample_id", "split", "category", "material", "finish", "lighting",
    "background", "brand_name", "sketch_file", "render_file", "depth_file",
    "positive_prompt", "negative_prompt", "resolution", "seed",
]


def scan_split_dir(split_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Map sample_id -> {kind: path} for complete triplets in a split dir."""
    found: Dict[str, Dict[str, Path]] = {}
    for p in sorted(split_dir.glob("*.png")):
        for suffix in SUFFIXES:
            if p.name.endswith(suffix):
                sid = p.name[: -len(suffix)]
                kind = suffix[1:-4]
                found.setdefault(sid, {})[kind] = p
                break
    return found


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def index_by_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        sid = rec.get("sample_id")
        if sid is None:
            continue
        if sid in out:
            print(f"[warn] duplicate sample_id {sid!r} in source records; keeping first occurrence")
            continue
        out[sid] = rec
    return out


def ordered_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Re-emit a record with the canonical field order first, extras after."""
    out = {k: rec[k] for k in FIELD_ORDER if k in rec}
    for k, v in rec.items():
        if k not in out:
            out[k] = v
    return out


def dir_relative(rec: Dict[str, Any], split: str, sid: str) -> Dict[str, Any]:
    """Point sketch/render/depth fields at the on-disk triplet location."""
    rec = dict(rec)
    rec["sketch_file"] = f"{split}/{sid}_sketch.png"
    rec["render_file"] = f"{split}/{sid}_render.png"
    rec["depth_file"] = f"{split}/{sid}_depth.png"
    return rec


def disambiguate(sid: str, taken: set) -> str:
    if sid + "_stale" not in taken:
        return sid + "_stale"
    n = 2
    while f"{sid}_stale{n}" in taken:
        n += 1
    return f"{sid}_stale{n}"


def synthesize_unknown(sid: str, split: str) -> Dict[str, Any]:
    """Minimal record for a triplet that no manifest/metadata knows about."""
    return {
        "sample_id": sid,
        "split": split,
        "category": "unknown",
        "material": "unknown",
        "finish": "unknown",
        "lighting": "unknown",
        "background": "unknown",
        "brand_name": "",
        "positive_prompt": "",
        "negative_prompt": "",
        "resolution": [512, 512],
        "seed": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair PackVision dataset manifests to cover every on-disk triplet."
    )
    parser.add_argument(
        "--data-root", default=str(PROJECT_ROOT / "data" / "dataset"),
        help="Dataset root containing train/, val/ and the manifest JSONL files.",
    )
    parser.add_argument("--write", action="store_true",
                        help="Actually rewrite the manifest files (default: dry run).")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_dir, val_dir = data_root / "train", data_root / "val"
    meta_path = data_root / "metadata.jsonl"
    train_path = data_root / "train_manifest.jsonl"
    val_path = data_root / "val_manifest.jsonl"

    disk = {"train": scan_split_dir(train_dir), "val": scan_split_dir(val_dir)}
    old_meta = index_by_id(load_jsonl(meta_path))
    old_train = index_by_id(load_jsonl(train_path))
    old_val = index_by_id(load_jsonl(val_path))

    print("=== BEFORE ===")
    print(f"disk triplets:  train={len(disk['train'])}  val={len(disk['val'])}  "
          f"total={len(disk['train']) + len(disk['val'])}")
    print(f"records:        train_manifest={len(old_train)}  val_manifest={len(old_val)}  "
          f"metadata={len(old_meta)}")

    overlap = sorted(set(disk["train"]) & set(disk["val"]))
    print(f"sample_ids present in BOTH directories: {len(overlap)}")

    new_records: Dict[str, Dict[str, Any]] = {"train": [], "val": []}
    stats = {"preserved": 0, "repathed": 0, "stale": 0, "synthesized": 0, "incomplete": 0}
    used_ids: Dict[str, set] = {"train": set(), "val": set()}

    # Authoritative split per id: metadata split, else the split manifest that lists it.
    def authoritative_split(sid: str) -> Optional[str]:
        if sid in old_meta and old_meta[sid].get("split") in ("train", "val"):
            return old_meta[sid]["split"]
        if sid in old_train:
            return "train"
        if sid in old_val:
            return "val"
        return None

    for split in ("train", "val"):
        for sid, files in disk[split].items():
            if set(files) != {"sketch", "render", "depth"}:
                print(f"[warn] incomplete triplet skipped: {split}/{sid} "
                      f"(has {sorted(files)})")
                stats["incomplete"] += 1
                continue

            auth = authoritative_split(sid)
            in_both = sid in disk["train"] and sid in disk["val"]
            is_stale = in_both and auth is not None and auth != split

            if is_stale:
                new_sid = disambiguate(sid, used_ids[split])
                used_ids[split].add(new_sid)
                source = (old_val if split == "val" else old_train).get(new_sid)
                if source and source.get("stale_duplicate_of") == sid:
                    rec = dict(source)  # already repaired previously; keep as-is
                elif sid in old_meta:
                    rec = dict(old_meta[sid])
                else:
                    rec = synthesize_unknown(sid, split)
                rec["sample_id"] = new_sid
                rec["split"] = split
                rec = dir_relative(rec, split, sid)
                rec["stale_duplicate_of"] = sid
                rec["seed"] = None  # generation seed of the stale copy is unknown
                new_records[split].append(ordered_record(rec))
                stats["stale"] += 1
                continue

            # Primary copy (or the only copy).
            new_sid = sid
            used_ids[split].add(new_sid)
            prior = (old_train if split == "train" else old_val).get(sid)
            if prior is not None and prior.get("split") == split:
                rec = dict(prior)
                fixed = dir_relative(rec, split, sid)
                if fixed == rec:
                    stats["preserved"] += 1
                else:
                    stats["repathed"] += 1
                rec = fixed
            elif sid in old_meta:
                rec = dir_relative(dict(old_meta[sid]), split, sid)
                rec["split"] = split
                stats["repathed"] += 1
            else:
                rec = dir_relative(synthesize_unknown(sid, split), split, sid)
                print(f"[warn] unknown triplet synthesized with category=unknown: "
                      f"{split}/{sid}")
                stats["synthesized"] += 1
            new_records[split].append(ordered_record(rec))

    # Union metadata: every record in either new manifest, sorted by sample_id.
    union = sorted(
        new_records["train"] + new_records["val"],
        key=lambda r: r["sample_id"],
    )
    meta_ids = [r["sample_id"] for r in union]
    if len(meta_ids) != len(set(meta_ids)):
        print("[error] sample_id collision in union; aborting")
        sys.exit(1)

    print("\n=== ACTIONS ===")
    print(f"preserved as-is:        {stats['preserved']}")
    print(f"(re)pathed/merged:      {stats['repathed']}")
    print(f"stale duplicates:       {stats['stale']} (disambiguated as *_stale)")
    print(f"synthesized (unknown):  {stats['synthesized']}")
    print(f"incomplete (skipped):   {stats['incomplete']}")

    print("\n=== AFTER ===")
    print(f"train_manifest records: {len(old_train)} -> {len(new_records['train'])}")
    print(f"val_manifest records:   {len(old_val)} -> {len(new_records['val'])}")
    print(f"metadata records:       {len(old_meta)} -> {len(union)}")
    both_manifests = set(r["sample_id"] for r in new_records["train"]) & \
        set(r["sample_id"] for r in new_records["val"])
    print(f"sample_ids in BOTH manifests after repair: {len(both_manifests)}")
    stale_in_train = sum(1 for r in new_records["train"] if "stale_duplicate_of" in r)
    stale_in_val = sum(1 for r in new_records["val"] if "stale_duplicate_of" in r)
    print(f"stale duplicates kept:  train={stale_in_train}  val={stale_in_val}")

    if not args.write:
        print("\nDry run only. Re-run with --write to rewrite the manifest files.")
        return

    for path, records in (
        (train_path, new_records["train"]),
        (val_path, new_records["val"]),
        (meta_path, union),
    ):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"wrote {path} ({len(records)} records)")
    print("Done.")


if __name__ == "__main__":
    main()
