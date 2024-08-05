import glob

from torch.utils.data import Dataset
from PIL import Image


id_dict = {}
for i, line in enumerate(open("vrpc/data/tiny-imagenet-200/wnids.txt", "r")):
    id_dict[line.replace("\n", "")] = i


class TrainTinyImageNetDataset(Dataset):
    def __init__(self, transform=None):
        self.filenames = glob.glob("vrpc/data/tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id_dict

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.id_dict[img_path.split("/")[4]]
        if self.transform:
            image = self.transform(image)
        return image, label


class TestTinyImageNetDataset(Dataset):
    def __init__(self, transform=None):
        self.filenames = glob.glob("vrpc/data/tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id_dict
        self.cls_dic = {}
        for line in open("vrpc/data/tiny-imagenet-200/val/val_annotations.txt", "r"):
            a = line.split("\t")
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.cls_dic[img_path.split("/")[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label
