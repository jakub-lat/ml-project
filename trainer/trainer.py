from comet_ml import Experiment
import torch
import torch.optim as optim

from model.metric import calculate_accuracy


def train(net, loader, criterion, device, experiment: Experiment, n_epochs, lr=0.01):
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    example_data, example_labels = next(iter(loader))
    example_data = example_data.to(device)

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
            experiment.log_metric('one_batch_loss', loss.item())

            if i % 50 == 49:
                with torch.no_grad():
                    preds = net(example_data)
                    accuracy = calculate_accuracy(preds, labels)

                    interval_loss = running_loss / 50
                    experiment.log_metric('interval_mean_loss', interval_loss)
                    print(f'[{epoch + 1:3d}, {i + 1:5d}] loss: {interval_loss} accuracy: {accuracy*100:.2f}%')
                    running_loss = 0.0
