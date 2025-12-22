#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Set


EXACT_REMOVE_NAMES = {
    "08.02.01-黄色",
    "五边形 3-橙色",
    "27.01-蓝色",
    "菱形 3-紫红",
    "19.05.027-黄色",
    "十字形 3-黄色",
    "19.03.38-蓝绿",
    "集群-黄色",
    "08.03.52-朱红",
    "08.01.01-黄色",
    "08.02.01-朱红",
    "19.03.04-黑色",
    "08.03.46-橙色",
    "19.04.08-黑色",
    "08.03.52-天蓝",
    "19.03.02-黄色",
    "08.02.02-天蓝",
    "08.03.28-朱红",
    "18.63-橙色",
    "十字形 3-橙色",
}

CONTAINS_REMOVE_KEYS = [
    "08.02.01",
    "18.63",
    "18.64",
    "集群",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_category_names(categories: List[Dict[str, Any]], path: Path) -> None:
    """NEW: 保存最终 category 名字到 txt"""
    with path.open("w", encoding="utf-8") as f:
        for cat in categories:
            f.write(f"{cat['name']}\n")


def should_remove_category(name: str) -> bool:
    if name in EXACT_REMOVE_NAMES:
        return True
    return any(k in name for k in CONTAINS_REMOVE_KEYS)


def process_coco(coco: Dict[str, Any]):
    if "categories" not in coco or not isinstance(coco["categories"], list):
        raise ValueError("输入 json 缺少 categories 或 categories 不是 list")
    if "annotations" not in coco or not isinstance(coco["annotations"], list):
        raise ValueError("输入 json 缺少 annotations 或 annotations 不是 list")

    old_categories = coco["categories"]

    kept_categories = []
    removed_old_cat_ids: Set[int] = set()

    # 1) 过滤 categories
    for cat in old_categories:
        name = str(cat.get("name", ""))
        cid = int(cat["id"])
        if should_remove_category(name):
            removed_old_cat_ids.add(cid)
        else:
            kept_categories.append(cat)

    # 2) 重新编号
    old_to_new: Dict[int, int] = {}
    new_categories = []
    next_id = 1
    for cat in kept_categories:
        old_id = int(cat["id"])
        old_to_new[old_id] = next_id
        new_cat = dict(cat)
        new_cat["id"] = next_id
        new_categories.append(new_cat)
        next_id += 1

    # 3) 处理 annotations
    new_annotations = []
    removed_ann = 0
    for ann in coco["annotations"]:
        cid = int(ann["category_id"])
        if cid not in old_to_new:
            removed_ann += 1
            continue
        new_ann = dict(ann)
        new_ann["category_id"] = old_to_new[cid]
        new_annotations.append(new_ann)

    coco_out = dict(coco)
    coco_out["categories"] = new_categories
    coco_out["annotations"] = new_annotations

    return coco_out, new_categories, removed_ann


def main():
    parser = argparse.ArgumentParser(
        description="Filter COCO categories, remap ids, and export category names."
    )
    parser.add_argument("-i", "--input", required=True, help="input coco json path")
    parser.add_argument("-o", "--output", required=True, help="output coco json path")
    parser.add_argument(
        "-c", "--cat_txt",
        default=None,
        help="output txt path for final category names (one per line)",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)

    # 默认 txt 路径：和 output.json 同名
    cat_txt = (
        Path(args.cat_txt)
        if args.cat_txt
        else outp.with_suffix(".categories.txt")
    )

    coco = load_json(inp)
    coco2, final_categories, removed_ann = process_coco(coco)

    save_json(coco2, outp)
    save_category_names(final_categories, cat_txt)

    print(f"[OK] COCO saved to: {outp}")
    print(f"[OK] Category list saved to: {cat_txt}")
    print(f"     categories: {len(coco['categories'])} -> {len(final_categories)}")
    print(f"     annotations removed: {removed_ann}")


if __name__ == "__main__":
    main()
