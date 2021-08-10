import json
import os


if __name__ == "__main__":
    """Build a mapping between image_id and instances"""
    coco_root = "/data/sdb/coco_2017"
    image_dir = os.path.join(coco_root, 'images/train2017')
    json_path = os.path.join(coco_root, 'annotations/instances_train2017.json')
    save_dir = "debug_coco"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(json_path) as f:
        data = json.load(f)
    images_list = data['images']  # length 118287
    anno_list = data['annotations']  # length 860001
    result = {}
    for ann in anno_list:
        img_id = ann["image_id"]
        if img_id not in result:
            result[img_id] = [ann]
        else:
            result[img_id].append(ann)
    """deal with images without instances belonging to the specific 80 classes"""
    """there are 1021 images without instances"""
    for img in images_list:
        id = img["id"]
        if id not in result:
            result[id] = []
    """save results"""
    result_path = os.path.join(coco_root, 'annotations/instances_train2017_image_anno.json')
    with open(result_path, "w") as f_w:
        json.dump(result, f_w)
