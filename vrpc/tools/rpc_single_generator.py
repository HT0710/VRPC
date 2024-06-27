from pathlib import Path
import shutil

from tqdm import tqdm


def main():
    data_path = Path("data/rpc")
    save_path = Path("data/rpc-single")

    if not data_path.exists():
        raise "Cannot found data path!"

    print("Cloning original data...")
    shutil.copytree(src=data_path, dst=save_path)

    print("Patching...")
    label_paths = save_path.rglob("*.txt")

    for path in tqdm(list(label_paths)):
        with open(str(path), "r") as f:
            labels = f.readlines()

        patched_labels = []
        for label in labels:
            parts = label.split(" ")

            parts[0] = "0"

            new_label = " ".join(parts)

            patched_labels.append(new_label)

        with open(str(path), "w") as f:
            f.write("".join(patched_labels))

    with open(f"{save_path}/data.yaml", "r") as f:
        infos = f.readlines()

    for i, line in enumerate(infos):
        if line.startswith("path:"):
            infos[i] = line.replace("rpc", "rpc-single")

        elif line.startswith("nc:"):
            infos[i] = "nc: 1\n"

        elif line.startswith("names:"):
            infos[i] = "names: ['object']\n"

    with open(f"{save_path}/data.yaml", "w") as f:
        f.write("".join(infos))


if __name__ == "__main__":
    main()
