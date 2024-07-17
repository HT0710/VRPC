import rootutils

rootutils.autosetup()

from modules.data.processing import ImagePreparation  # noqa: E402


def main():
    IP = ImagePreparation(save_folder="vrpc/data")

    IP.auto(data_path="vrpc/data/rpc-classify", remake=False)


if __name__ == "__main__":
    main()
