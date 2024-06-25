from collections import defaultdict
from pathlib import Path
import shutil
import json
import os

from tqdm import tqdm


def xywh_to_yolov8(x, y, w, h, image_width, image_height):
    # Calculate center coordinates
    center_x = x + w / 2
    center_y = y + h / 2

    # Normalize coordinates
    center_x_normalized = center_x / image_width
    center_y_normalized = center_y / image_height

    # Normalize dimensions
    width_normalized = w / image_width
    height_normalized = h / image_height

    return center_x_normalized, center_y_normalized, width_normalized, height_normalized


def process(images_path: str, label_path: str, save_path: str):
    # create save folder
    (save_path / "images").mkdir(parents=True, exist_ok=True)
    (save_path / "labels").mkdir(parents=True, exist_ok=True)

    # read json data
    with open(label_path) as file:
        data = json.load(file)

    # classes
    categories = data["categories"]

    images = sorted(data["images"], key=lambda x: x["id"])

    annotations = sorted(data["annotations"], key=lambda x: x["image_id"])

    # group annotations by id
    grouped_annotations = defaultdict(list)
    for annotation in annotations:
        key = annotation["image_id"]
        value = {
            "category": categories[annotation["category_id"] - 1]["id"] - 1,
            "bbox": [int(x) for x in annotation["bbox"]],
        }
        grouped_annotations[key].append(value)

    # done preparation

    # main loop
    for image in tqdm(images):
        # copy image
        file_name = image["file_name"]

        file_path = f"{images_path}/{file_name}"

        image_save_path = f"{save_path}/images/{file_name}"

        if not os.path.exists(image_save_path):
            shutil.copyfile(src=file_path, dst=image_save_path)

        # create label
        image_id = image["id"]

        annotation = grouped_annotations.pop(image_id)

        label = []
        for item in annotation:
            # convert to yolo format
            yolo_format = xywh_to_yolov8(
                *item["bbox"],
                image_width=image["width"],
                image_height=image["height"],
            )
            label.append(" ".join(str(x) for x in [item["category"], *yolo_format]))

        # save txt
        label_save_path = f"{save_path}/labels/{file_name[:-4]}.txt"
        with open(label_save_path, "w+") as f:
            f.write("\n".join(label))

    # create config yaml
    with open(f"{save_path.parent}/data.yaml", "w") as f:
        f.write(
            "\n".join(
                [
                    f"path: {os.getcwd()}/data/rpc",
                    "train: ../test/images",
                    "val: ../val/images",
                    f"nc: {len(categories)}",
                    f"names: {[x['name'] for x in categories]}",
                ]
            )
        )


def main():
    data_type = ["test", "val"]

    for type in data_type:
        label_path = Path(f"data/archive/instances_{type}2019.json")
        images_path = Path(f"data/archive/{type}2019")
        save_path = Path(f"data/rpc/{type}")

        process(images_path, label_path, save_path)


if __name__ == "__main__":
    main()
