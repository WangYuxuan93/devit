#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple


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
    with path.open("w", encoding="utf-8") as f:
        for cat in categories:
            f.write(f"{cat['name']}\n")


def contains_should_delete(name: str) -> bool:
    """只要包含这些关键词就删除（优先级最高）"""
    return any(k in name for k in CONTAINS_REMOVE_KEYS)


def process_coco(coco: Dict[str, Any]):
    if "categories" not in coco or not isinstance(coco["categories"], list):
        raise ValueError("输入 json 缺少 categories 或 categories 不是 list")
    if "annotations" not in coco or not isinstance(coco["annotations"], list):
        raise ValueError("输入 json 缺少 annotations 或 annotations 不是 list")

    old_categories = coco["categories"]

    # 1) 先按 CONTAINS_REMOVE_KEYS 删除
    kept_front: List[Dict[str, Any]] = []   # 不在 EXACT_REMOVE_NAMES 的，保持原顺序
    kept_tail: List[Dict[str, Any]] = []    # 在 EXACT_REMOVE_NAMES 的，放到最后
    deleted_old_cat_ids: Set[int] = set()

    for cat in old_categories:
        name = str(cat.get("name", ""))
        old_id = int(cat["id"])

        if contains_should_delete(name):
            deleted_old_cat_ids.add(old_id)
            continue

        # 不删 exact 列表，而是移动到最后
        if name in EXACT_REMOVE_NAMES:
            kept_tail.append(cat)
        else:
            kept_front.append(cat)

    # 2) 拼接：前面 + 后面（exact那批）
    kept_categories = kept_front + kept_tail

    # 3) 重新编号 categories，从 1 开始，并建立 old_id -> new_id 映射
    old_to_new: Dict[int, int] = {}
    new_categories: List[Dict[str, Any]] = []

    next_id = 1
    for cat in kept_categories:
        old_id = int(cat["id"])
        old_to_new[old_id] = next_id
        new_cat = dict(cat)
        new_cat["id"] = next_id
        new_categories.append(new_cat)
        next_id += 1

    # 4) 更新 annotations：被删除类别引用的 annotation 删掉，否则更新 category_id
    new_annotations: List[Dict[str, Any]] = []
    removed_ann = 0

    for ann in coco["annotations"]:
        old_cid = int(ann["category_id"])

        # 类别被 CONTAINS 删除 或 不存在于映射（安全兜底）=> 删除 annotation
        if old_cid in deleted_old_cat_ids or old_cid not in old_to_new:
            removed_ann += 1
            continue

        new_ann = dict(ann)
        new_ann["category_id"] = old_to_new[old_cid]
        new_annotations.append(new_ann)

    coco_out = dict(coco)
    coco_out["categories"] = new_categories
    coco_out["annotations"] = new_annotations

    return coco_out, new_categories, removed_ann, len(deleted_old_cat_ids), (len(kept_front), len(kept_tail))


def main():
    parser = argparse.ArgumentParser(
        description="Move EXACT_REMOVE_NAMES categories to the end, delete CONTAINS_REMOVE_KEYS categories, and remap ids."
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
    cat_txt = Path(args.cat_txt) if args.cat_txt else outp.with_suffix(".categories.txt")

    coco = load_json(inp)
    coco2, final_categories, removed_ann, deleted_cat_cnt, (front_cnt, tail_cnt) = process_coco(coco)

    save_json(coco2, outp)
    save_category_names(final_categories, cat_txt)

    print(f"[OK] COCO saved to: {outp}")
    print(f"[OK] Category list saved to: {cat_txt}")
    print(f"     categories: {len(coco['categories'])} -> {len(final_categories)}")
    print(f"     deleted categories (contains keys): {deleted_cat_cnt}")
    print(f"     kept categories: front={front_cnt}, moved_to_end(exact_list)={tail_cnt}")
    print(f"     annotations removed: {removed_ann}")


if __name__ == "__main__":
    main()
