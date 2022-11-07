import torch
import cv2
import numpy
import matplotlib.pyplot as plt
import os
from tqdm import trange
from PIL import Image
from torch.utils.data import DataLoader
from torch import nn, optim
from my_dataset import MyDataSet
from UNet import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = './data'
img_size = (512, 512)
weight_path = 'unet.pth'
epoch = 50
save_epoch = 5

if __name__ == '__main__':
    data_loader = DataLoader(MyDataSet(data_dir, 'CHASE', 'train', img_size), batch_size=4, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    for i in trange(epoch):
        for v, (img, manual) in enumerate(data_loader):
            img, manual = img.to(device), manual.to(device)
            print(img.shape)
            out_img = net(img)
            loss = loss_fun(out_img, manual)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f'{i}--loss:{loss.item()}')

            stack_img = torch.stack([manual[0], out_img[0]], dim=0)
            save_image(stack_img, f'exp_image/{i}-{v + 1}.png')
        if i % save_epoch == 0:
            torch.save(net.state_dict(), weight_path)
            print('Save Weight...')
