#!/usr/bin/env python3
"""Create a reproducible, source-grouped provisional split from round 11."""

from __future__ import annotations

import collections
import hashlib
import heapq
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "staging/round11/train"
DEST = ROOT / "staging/round12"
REPORTS = ROOT / "reports"
SEED = 20261107
FILES = ("article_en.jsonl", "article_zh.jsonl", "natural_en.jsonl", "natural_zh.jsonl")
SPLITS = ("train", "valid", "test")
TARGET = {"train": 0.8, "valid": 0.1, "test": 0.1}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        x, y = self.find(x), self.find(y)
        if x != y:
            self.parent[max(x, y)] = min(x, y)


def shingles(text: str) -> set[int]:
    # Spacing is ignored for candidate generation and for the Jaccard check.
    normalized = re.sub(r"\s+", "", text.casefold())
    pieces = {normalized[i : i + 5] for i in range(max(1, len(normalized) - 4))}
    return {int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in pieces}


def main() -> None:
    rows = []
    for name in FILES:
        with (SOURCE / name).open(encoding="utf-8") as stream:
            for line in stream:
                rows.append((name, json.loads(line)))
    ids = [row[1]["id"] for row in rows]
    assert len(ids) == len(set(ids)), "IDs must be unique before splitting"
    uf = UnionFind(len(rows))
    prior_group = {}
    for i, (name, row) in enumerate(rows):
        group = row.get("group_id")
        if group:
            key = (name, group)
            if key in prior_group:
                uf.union(i, prior_group[key])
            else:
                prior_group[key] = i

    signatures = []
    buckets = collections.defaultdict(list)
    for i, (_, row) in enumerate(rows):
        sig = shingles(row["text"])
        signatures.append(sig)
        for hash_value in heapq.nsmallest(12, sig):
            buckets[(row["language"], hash_value)].append(i)
    shared = collections.Counter()
    large_candidate_buckets = 0
    for members in buckets.values():
        if len(members) > 200:
            large_candidate_buckets += 1
        for offset, a in enumerate(members):
            for b in members[offset + 1 :]:
                if min(len(signatures[a]), len(signatures[b])) * 10 < max(len(signatures[a]), len(signatures[b])) * 7:
                    continue
                shared[(a, b)] += 1
    near_pairs = []
    for (a, b), count in shared.items():
        if count < 2:
            continue
        left, right = signatures[a], signatures[b]
        if len(left & right) / len(left | right) >= 0.85:
            uf.union(a, b)
            near_pairs.append((rows[a][1]["id"], rows[b][1]["id"]))

    groups = collections.defaultdict(list)
    for i in range(len(rows)):
        groups[uf.find(i)].append(i)
    group_list = sorted(groups.values(), key=lambda members: min(rows[i][1]["id"] for i in members))
    rng = random.Random(SEED)
    rng.shuffle(group_list)
    totals = collections.Counter(name for name, _ in rows)
    counts = {name: collections.Counter() for name in FILES}
    assignments = {}
    manifest_groups = []
    for members in group_list:
        by_file = collections.Counter(rows[i][0] for i in members)
        def score(split: str) -> float:
            return sum(
                (counts[name][split] + n - totals[name] * TARGET[split]) ** 2
                - (counts[name][split] - totals[name] * TARGET[split]) ** 2
                for name, n in by_file.items()
            ) / max(TARGET[split], 0.01)
        split = min(SPLITS, key=score)
        for name, n in by_file.items():
            counts[name][split] += n
        explicit = sorted({rows[i][1]["group_id"] for i in members if rows[i][1].get("group_id")})
        if len(explicit) == 1:
            group_id = explicit[0]
        elif len(members) == 1:
            group_id = "record-" + rows[members[0]][1]["id"]
        else:
            group_id = "near-" + digest("\n".join(sorted(rows[i][1]["id"] for i in members)))[:16]
        for i in members:
            assignments[i] = (split, group_id)
        manifest_groups.append({"group_id": group_id, "split": split, "size": len(members), "files": dict(by_file), "original_group_ids": explicit, "ids": [rows[i][1]["id"] for i in members]})

    # A split is not emitted if the exact and normalized duplicate gates fail.
    for normalizer in (lambda x: x, lambda x: re.sub(r"\s+", "", x.casefold())):
        seen = {}
        for i, (_, row) in enumerate(rows):
            key = normalizer(row["text"])
            if key in seen and assignments[seen[key]][0] != assignments[i][0]:
                raise AssertionError(f"cross-split duplicate: {row['id']}")
            seen[key] = i
    id_to_index = {record_id: i for i, record_id in enumerate(ids)}
    for a, b in ((id_to_index[a], id_to_index[b]) for a, b in near_pairs):
        assert assignments[a][0] == assignments[b][0]

    for split in SPLITS:
        (DEST / split).mkdir(parents=True, exist_ok=True)
        for name in FILES:
            with (DEST / split / name).open("w", encoding="utf-8") as stream:
                for i, (row_name, row) in enumerate(rows):
                    if row_name != name or assignments[i][0] != split:
                        continue
                    out = dict(row)
                    out["split"], out["group_id"] = assignments[i]
                    stream.write(json.dumps(out, ensure_ascii=False) + "\n")
    manifest = {"seed": SEED, "source": str(SOURCE), "status": "provisional_pending_source_rights_review", "counts": {name: dict(counts[name]) for name in FILES}, "group_count": len(group_list), "near_duplicate_pairs": near_pairs, "large_candidate_buckets_checked": large_candidate_buckets, "groups": manifest_groups}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "split_manifest_round12.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("groups", "near_duplicate_pairs")}, ensure_ascii=False))
    print("near_duplicate_pairs", len(near_pairs))


if __name__ == "__main__":
    main()
