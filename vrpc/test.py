import torch.nn as nn
import torch

from modules.data.transform import DataTransformation
import cv2


def main():
    DT = DataTransformation.TOPIL(672)
    AP = nn.AvgPool2d((24, 24))

    conv = nn.Conv2d(3, 64, 3, padding=1)

    image_1 = cv2.imread("vrpc/data/20180824-16-28-47-507_0.jpg")
    # image_2 = cv2.imread("vrpc/data/20180828-16-25-14-994_4.jpg")

    # x1 = cv2.stackBlur(image_1, (17, 17))
    # x2 = cv2.stackBlur(image_2, (17, 17))

    # x1 = AP(DT(x1))
    # x2 = AP(DT(x2))

    # x1 = x1.permute(1, 2, 0).detach().numpy()
    # x2 = x2.permute(1, 2, 0).detach().numpy()

    x0 = cv2.resize(image_1, (500, 500))
    x1 = cv2.resize(image_1, (50, 50))
    x2 = cv2.resize(x1, (500, 500))

    # cv2.imshow("0", x0)
    # cv2.imshow("1", x1)
    # cv2.imshow("2", x2)

    out = conv(DT(x0 - x2)).permute(1, 2, 0).detach().numpy()
    for i in range(64):
        x = out[:, :, i]
        print(x.shape)
        cv2.imshow("3", x)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
