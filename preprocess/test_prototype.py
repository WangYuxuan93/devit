import torch
import json

#"""
#json_file = "datasets/geomap/annotations_train.json"
#json_file = "datasets/geomap/new/annotations_train.json"
#json_file = "datasets/geomap/annotations_val.json"
#json_file = "datasets/coco/annotations_origin/coco_2017_val_oneshot_s1.json"
#json_file = "datasets/coco/annotations_origin/coco_2017_train_oneshot_s1.json"
#json_file = "datasets/coco/annotations/coco_2017_novel_oneshot_s4_r50.json"
#json_file = "datasets/coco/annotations/instances_val2017.json"

json_file = "datasets/geomap/annotations/geomap_point_train_oneshot_s1.json"
json_file = "datasets/geomap/annotations/geomap_point_valid.json"
json_file = "datasets/geomap/annotations/geomap_point_novel_oneshot_s1_r50.json"

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)
print (json_file)
print (data.keys())
print (data["categories"])
print ("images:{}, annotations: {}".format(len(data["images"]), len(data["annotations"])))

cat_ids = []
for ann in data["annotations"]:
    cat_id = ann["category_id"]
    if cat_id not in cat_ids:
        cat_ids.append(cat_id)

print ("number:{}, cat_ids:{}".format(len(cat_ids), sorted(cat_ids)))

cat_names = []
for cat in data["categories"]:
    cat_name = cat["name"]#.split("-")[0]
    if cat_name not in cat_names:
        cat_names.append(cat_name)

print ("number:{}, cat_names:{}".format(len(cat_names), cat_names))

exit()
#"""

#pkl_path = "weights\initial\oneshot\prototypes_for_train\coco_2017_train_oneshot_s1.vitl14.pkl"
#pkl_path = "weights\initial\oneshot\prototypes_for_train\coco_2017_novel_oneshot_s1.vitl14.pkl"
#pkl_path = "weights\initial\oneshot\prototypes_for_train\coco_2017_novel_oneshot_s4.vitl14.pkl"
pkl_path = "weights/initial/oneshot/ref_coco17.vitl14.pth"

#pkl_path = "weights\geomap\geomap_train_oneshot_s1.vitl14.bbox.pkl"
#pkl_path = "weights\geomap\geomap_val_oneshot_s1.vitl14.bbox.p10.sk.pkl"
#pkl_path = "weights\geomap\geomap_val_oneshot_s1.vitl14.bbox.pkl"

data = torch.load(pkl_path, map_location="cpu")

print (pkl_path)
print (len(data.keys()), data.keys())
print (len(data["labels"]))

if "origin_label_names" in data:
    print (len(data["origin_label_names"]))
    print ("origin_label_names:", data["origin_label_names"])

print (len(data["label_names"]))
print ("label_names:", data["label_names"])

cat_names = []
for cat in data["label_names"]:
    cat_name = cat.split(".")[0]
    if cat_name not in cat_names:
        cat_names.append(cat_name)
print ("number:{}, cat_names:{}".format(len(cat_names), cat_names))    

print (data["prototypes"].shape)

#exit()
p = data["prototypes"]
norm = torch.norm(p, dim=-1)
print(norm.mean(), norm.min(), norm.max())

protos = data["prototypes"]
print(type(protos))

A = protos[0][0]  # [10,1024]
B = protos[0][1]  # [10,1024]

print (A)

print (B)

print("A==B (exact):", torch.equal(A, B))
print("max|A-B|:", (A - B).abs().max().item())
print("mean|A-B|:", (A - B).abs().mean().item())
print("L2(A,B):", torch.norm(A - B).item())

#print ("labels:", data["label_names"])
if isinstance(protos, torch.Tensor):
    print("prototypes shape:", protos.shape)
elif isinstance(protos, dict):
    print("prototype keys:", protos.keys())