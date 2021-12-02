import torch
import sys 
sys.path.append('../')
import torch.nn as nn
from vgg import vgg11_bn_half, vgg11_bn, vgg11_bn_quarter
from trainer import test, train_prune
from utils import get_new_weights, compare_parameters
from dataloader import get_dataloader_cifar

if __name__ == '__main__':

	BATCH_SIZE = 32
	train_loader, val_loader, test_loader = get_dataloader_cifar(BATCH_SIZE)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	vgg11_bn = vgg11_bn()
	vgg11_bn_quarter = vgg11_bn_quarter()
	model_weights = torch.load("./weights/vgg11.pt", map_location = 'cpu')

	new_model_weights = get_new_weights(model_weights)

	for name, weight in vgg11_bn_quarter.named_parameters():
            weight.data = nn.parameter.Parameter(new_model_weights[name])
            weight.requires_grad = True

	vgg11_bn_quarter.to(device)

	print(compare_parameters(vgg11_bn, vgg11_bn_quarter))

	train_prune(vgg11_bn_quarter, 10, device, train_loader, val_loader)

	test(vgg11_bn_quarter, device, test_loader)