from collections import defaultdict
from pathlib import Path
import json
import os

from tqdm import tqdm
import cv2


def _process(images_path: str, label_path: str, save_path: str):
    # read json data
    with open(label_path) as file:
        data = json.load(file)

    # infos
    categories = data["categories"]

    # get all classes
    classes = [categorie["name"] for categorie in categories]

    # create save path
    for name in classes:
        (Path(save_path) / name).mkdir(parents=True, exist_ok=True)

    # sort annotations by id
    annotations = sorted(data["annotations"], key=lambda x: x["image_id"])

    # group annotations by id
    grouped_annotations = defaultdict(list)
    for annotation in annotations:
        key = annotation["image_id"]
        value = {
            "category": categories[annotation["category_id"] - 1]["name"],
            "bbox": [int(x) for x in annotation["bbox"]],
        }
        grouped_annotations[key].append(value)

    # sort images by id
    images = sorted(data["images"], key=lambda x: x["id"])

    # main loop
    for image_info in tqdm(images):
        # get image
        image = cv2.imread(f"{images_path}/{image_info['file_name']}")

        # get label
        annotation = grouped_annotations.pop(image_info["id"])

        for i, obj_info in enumerate(annotation):
            category = obj_info["category"]

            x, y, w, h = obj_info["bbox"]

            obj = image[y : y + h, x : x + w]

            obj_save_path = f"{save_path}/{category}/{image_info['file_name'].replace('.', f'_{i}.')}"

            if not os.path.exists(obj_save_path):
                cv2.imwrite(filename=obj_save_path, img=obj)


def main():
    data_type = ["train", "val", "test"]

    for type in data_type:
        label_path = Path(f"vrpc/data/archive/instances_{type}2019.json")
        images_path = Path(f"vrpc/data/archive/{type}2019")
        save_path = Path("vrpc/data/rpc-classify")

        _process(images_path, label_path, save_path)


if __name__ == "__main__":
    main()
