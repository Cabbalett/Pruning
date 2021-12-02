import torch
from vgg import vgg11_bn
from trainer import test, train
from dataloader import get_dataloader_cifar
import os

if __name__ == '__main__':

    BATCH_SIZE = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloader_cifar(BATCH_SIZE)
    os.makedirs('./weights', exist_ok=True)

    vgg11 = vgg11_bn()
    vgg11.to(device)

    train(vgg11, 20, device, train_loader, val_loader)
    test(vgg11, device, test_loader)