import os
import logging
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

logger = logging.getLogger(__name__)

def load_custom_dataset(root_dir):
    """
    加载自定义数据集
    
    参数:
        root_dir: 数据集根目录，包含images和labels子目录
    返回:
        符合Detectron2格式的数据集列表
    """
    # 定义路径
    images_dir = os.path.join(root_dir, "geomap","images")
    labels_path = os.path.join(root_dir, "geomap","labels", "alllabel.txt")
    
    # 检查路径是否存在
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"标签文件不存在: {labels_path}")
    
    # 读取所有标签
    dataset_dicts = []
    classes = set()  # 用于收集所有类别
    
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # 分割行数据
            parts = line.split()
            if len(parts) < 6:
                logger.warning(f"无效的标签格式: {line}")
                continue
                
            # 解析基本信息
            filename = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            label = " ".join(parts[5:])  # 标签可能包含空格
            
            # 收集类别
            classes.add(label)
            
            # 构建图像路径
            image_path = os.path.join(images_dir, f"{filename}.jpg")  # 假设是jpg格式
            if not os.path.isfile(image_path):
                # 尝试其他常见格式
                for ext in [".png", ".jpeg", ".bmp"]:
                    image_path = os.path.join(images_dir, f"{filename}{ext}")
                    if os.path.isfile(image_path):
                        break
                else:
                    logger.warning(f"找不到图像文件: {filename}")
                    continue
            
            # 获取图像ID (从文件名提取数字部分)
            try:
                image_id = int(filename.split("_")[1])
            except (IndexError, ValueError):
                image_id = hash(filename)  # 作为备选方案
            
            # 查找当前图像是否已在数据集列表中
            record = next((d for d in dataset_dicts if d["image_id"] == image_id), None)
            
            from PIL import Image
            img = Image.open(image_path)
            width,height = img.size
            if not record:
                # 创建新的图像记录
                record = {
                    "file_name": image_path,
                    "image_id": image_id,
                    "height": height,  # 可以留空，Detectron2会自动读取
                    "width": width,   # 可以留空，Detectron2会自动读取
                    "annotations": []
                }
                dataset_dicts.append(record)
            
            # 添加标注信息
            record["annotations"].append({
                "bbox": [x, y, w, h],  # [x, y, width, height]
                "bbox_mode": BoxMode.XYWH_ABS,  # 绝对坐标的XYWH格式
                "category_id": label,  # 临时使用标签字符串，后续会转换为ID
                "iscrowd": 0  # 默认为0，表示不是人群
            })
    
    # 将类别名称映射为整数ID
    class_list = sorted(classes)
    class_id_map = {name: i for i, name in enumerate(class_list)}
    
    # 更新标注中的category_id为整数ID
    for record in dataset_dicts:
        for ann in record["annotations"]:
            ann["category_id"] = class_id_map[ann["category_id"]]
    
    logger.info(f"加载完成: {len(dataset_dicts)} 张图像，{len(class_list)} 个类别")
    return dataset_dicts, class_list

def register_geomap_dataset(name, root_dir):
    """
    注册自定义数据集到Detectron2
    
    参数:
        name: 数据集名称 (如 "custom_train")
        root_dir: 数据集根目录
    """
    # 注册数据集
    def load_fn():
        dataset_dicts, class_list = load_custom_dataset(root_dir)
        return dataset_dicts
    
    DatasetCatalog.register(name, load_fn)
    
    # 获取类别列表
    _, class_list = load_custom_dataset(root_dir)
    
    # 注册元数据
    MetadataCatalog.get(name).set(
        thing_classes=class_list,
        root_dir=root_dir
    )
    
    logger.info(f"已注册数据集: {name}，包含 {len(class_list)} 个类别")

# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 数据集根目录
    dataset_root = "data"  # 你的数据根目录
    
    # 注册数据集
    register_geomap_dataset("custom_dataset", dataset_root)
    
    # 测试加载
    from detectron2.data import DatasetCatalog, MetadataCatalog
    
    dataset_dicts = DatasetCatalog.get("custom_dataset")
    metadata = MetadataCatalog.get("custom_dataset")
    
    print(f"数据集包含 {len(dataset_dicts)} 张图像")
    print(f"类别列表: {metadata.thing_classes}")
    
