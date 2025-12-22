#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_labels_txt(path: Path) -> List[str]:
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            labels.append(s)
    # 去重但保序
    seen = set()
    out = []
    for x in labels:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_cat_maps(categories: List[Dict[str, Any]]) -> Tuple[Dict[int, str], Dict[str, int]]:
    id2name = {}
    name2id = {}
    for c in categories:
        cid = int(c["id"])
        name = str(c["name"])
        id2name[cid] = name
        name2id[name] = cid
    return id2name, name2id


def main():
    parser = argparse.ArgumentParser(
        description="Build k-shot COCO json for given val labels (expand label to label-0..label-(k-1))."
    )
    parser.add_argument("-i", "--input", required=True, help="input coco json path")
    parser.add_argument("-l", "--val_labels", required=True, help="val_labels.txt path (one label name per line)")
    parser.add_argument("-k", "--k", type=int, required=True, help="k-shot per label")
    parser.add_argument("-o", "--output", required=True, help="output coco json path")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="do not shuffle; take first k annotations per label in original order",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    labels_path = Path(args.val_labels)
    outp = Path(args.output)

    if args.k <= 0:
        raise ValueError("k 必须是正整数")

    coco = load_json(inp)

    if "categories" not in coco or not isinstance(coco["categories"], list):
        raise ValueError("输入 json 缺少 categories 或 categories 不是 list")
    if "annotations" not in coco or not isinstance(coco["annotations"], list):
        raise ValueError("输入 json 缺少 annotations 或 annotations 不是 list")
    if "images" not in coco or not isinstance(coco["images"], list):
        raise ValueError("输入 json 缺少 images 或 images 不是 list")

    val_labels = load_labels_txt(labels_path)
    id2name, name2id = build_cat_maps(coco["categories"])

    # 收集每个 label 对应的 annotations
    anns_by_label: Dict[str, List[Dict[str, Any]]] = {lab: [] for lab in val_labels}
    for ann in coco["annotations"]:
        cid = int(ann["category_id"])
        name = id2name.get(cid, None)
        if name in anns_by_label:
            anns_by_label[name].append(ann)

    rng = random.Random(args.seed)

    # 1) 构建新的 categories：每个 label 扩展成 k 个 label-i
    new_categories: List[Dict[str, Any]] = []
    new_name_to_id: Dict[str, int] = {}
    next_cat_id = 1
    for lab in val_labels:
        for i in range(args.k):
            new_name = f"{lab}-{i}"
            new_name_to_id[new_name] = next_cat_id
            new_categories.append({
                "id": next_cat_id,
                "name": new_name,
                "supercategory": "map_symbol",
            })
            next_cat_id += 1

    # 2) 选择 annotations，并把它们的 category_id 映射到扩展后的新类别
    new_annotations: List[Dict[str, Any]] = []
    used_image_ids: Set[int] = set()
    missing_labels: List[str] = []
    not_enough: List[Tuple[str, int]] = []

    next_ann_id = 1

    for lab in val_labels:
        candidates = anns_by_label.get(lab, [])
        if len(candidates) == 0:
            missing_labels.append(lab)
            continue

        if not args.no_shuffle:
            candidates = candidates[:]  # copy
            rng.shuffle(candidates)

        chosen = candidates[:args.k]
        if len(chosen) < args.k:
            not_enough.append((lab, len(chosen)))

        # 依序分配到 lab-0, lab-1, ...
        for idx, ann in enumerate(chosen):
            new_label = f"{lab}-{idx}"
            new_cid = new_name_to_id[new_label]

            ann2 = dict(ann)
            ann2["id"] = next_ann_id
            next_ann_id += 1
            ann2["category_id"] = new_cid

            new_annotations.append(ann2)
            used_image_ids.add(int(ann2["image_id"]))

    # 3) 只保留用到的 images
    new_images = [img for img in coco["images"] if int(img["id"]) in used_image_ids]

    # 4) 输出 coco
    coco_out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories,
    }

    save_json(coco_out, outp)

    # 打印统计
    print(f"[OK] Saved: {outp}")
    print(f"     labels in txt: {len(val_labels)}")
    print(f"     categories(out): {len(new_categories)} (each label expanded to k={args.k})")
    print(f"     annotations(out): {len(new_annotations)}")
    print(f"     images(out): {len(new_images)}")
    if missing_labels:
        print(f"[WARN] labels with 0 samples in input annotations: {len(missing_labels)}")
        for x in missing_labels[:20]:
            print(f"       - {x}")
        if len(missing_labels) > 20:
            print("       ...")
    if not_enough:
        print(f"[WARN] labels with <k samples: {len(not_enough)} (kept all available)")
        for lab, n in not_enough[:20]:
            print(f"       - {lab}: {n}/{args.k}")
        if len(not_enough) > 20:
            print("       ...")


if __name__ == "__main__":
    main()
