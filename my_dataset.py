import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


def generate_path_list(img_dir):
    file_list = os.listdir(img_dir)
    file_list.sort()
    return [img_dir + '/' + v for v in file_list]


class MyDataSet(Dataset):
    def __init__(self, root_dir, data_set='CHASE', flag_dir="train", img_size=(512, 512)):
        super(MyDataSet, self).__init__()
        self.flag = flag_dir
        data_dir = os.path.join(root_dir, data_set, self.flag)
        self.img_list = generate_path_list(os.path.join(data_dir, 'images'))
        self.manual_list = generate_path_list(os.path.join(data_dir, '1st_manual'))
        self.img_size = img_size

    def __getitem__(self, item):
        img = Image.open(self.img_list[item])
        manual = Image.open(self.manual_list[item])
        img = np.array(img.resize(self.img_size, Image.BILINEAR)).astype(np.float32)
        manual = np.array(manual.resize(self.img_size, Image.BILINEAR)).astype(np.float32)
        print(img.shape, manual.shape)
        return transform(img), transform(manual)

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    dataset = MyDataSet('./data', 'CHASE', 'train')
    img, manual = dataset.__getitem__(2)
    print(img, manual)
    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(manual)
    # plt.show()
