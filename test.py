import torch
import cv2
import numpy
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from my_dataset import MyDataSet
from UNet import *
from torchvision.utils import save_image

data_dir = './data'
img_size = (512, 512)
weight_path = 'unet.pth'
epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dice_evaluate(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    if np.max(output) > 1:
        output = output / 255.
    if np.max(target) > 1:
        target = target / 255

    return (2. * (output * target).sum() + smooth) / \
           (output.sum() + target.sum() + smooth)


def test():
    dice_all = 0
    data_loader = DataLoader(MyDataSet(data_dir, 'CHASE', 'test', img_size), batch_size=1, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
    for v, (img, manual) in tqdm(enumerate(data_loader)):
        img, manual = img.to(device), manual.to(device)
        out_img = net(img)
        dice = dice_evaluate(out_img, manual)
        print('index:' + str(v + 1) + ' dice store:' + str(dice))
        dice_all += dice
        stack_img = torch.stack([manual[0], out_img[0]], dim=0)
        save_image(stack_img, f'exp_image_2/{v + 1}.png')
    print('Average Dice Score: {:.4f}%'.format((dice_all / len(data_loader.dataset)) * 100))


if __name__ == '__main__':
    test()
