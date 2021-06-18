import torch
import torch.optim as optim
from comet_ml import Experiment

from model.metric import calculate_accuracy


def train(net, train_loader, test_loader, criterion, device, experiment: Experiment, n_epochs, lr, check_every,
          test_batches):
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            running_loss += loss.item()
            experiment.log_metric('one_batch_loss', loss.item())

            if i % check_every == 0:
                with torch.no_grad():

                    i = 0
                    running_accuracy = 0.0
                    for data, labels in test_loader:
                        if i == test_batches:
                            break

                        preds = net(data.to(device))
                        running_accuracy += calculate_accuracy(preds, labels.to(device))
                        i += 1

                    experiment.log_metric('interval_mean_loss', running_loss / check_every)
                    experiment.log_metric('interval_accuracy', running_accuracy / i)

                    running_loss = 0.0
