import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

def test(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader))
        running_loss = 0.0
        for i, (images, labels) in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            outputs = torch.squeeze(outputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_loss += criterion(outputs, labels).item()
            correct += (predicted == labels).sum().item()
            pbar.update()
            pbar.set_description(
                f" Val: {'':5} Loss: {(running_loss / (i + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
            )

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))


def train_prune(model, EPOCHS, device, train_loader, val_loader):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        total, correct, running_loss = 0, 0, 0.0
        for i, (inputs, labels) in pbar:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            pbar.update()
            pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (i + 1)):.3f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                )
        pbar.close()
        torch.save(model.state_dict(), './weights/vgg11_pruned.pt')
        test(model, device, val_loader)

def train(model, EPOCHS, device, train_loader, val_loader):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        total, correct, running_loss = 0, 0, 0.0
        for i, (inputs, labels) in pbar:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            l1_reg=0
            for name, layer in model.named_parameters():
                if 'bn' in name and 'weight' in name:
                    l1_reg+= torch.norm(layer)

            # import pdb;pdb.set_trace()
            loss = criterion(outputs, labels) + 0.001*l1_reg
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            pbar.update()
            pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (i + 1)):.3f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                )
        pbar.close()
        torch.save(model.state_dict(), './weights/vgg11.pt')
        test(model, device, val_loader)