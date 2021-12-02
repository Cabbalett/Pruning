import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloader_cifar(batch_size: int):
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True)
                                            
    val_list = list(range(0, len(test_dataset)//5))

    val_dataset = torch.utils.data.Subset(test_dataset, val_list)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    return train_dataloader, val_dataloader, test_dataloader