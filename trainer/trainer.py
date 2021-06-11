import torch.optim as optim
import torch.types


def train(net, loader, criterion, device: torch.types.Device, n_epochs: int, lr: int = 0.001):
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            running_loss += loss.item()

            if i % 50 == 49:
                print(f'[{epoch + 1:3d}, {i + 1:5d}] loss: {running_loss / 200}')
                running_loss = 0.0
