import torch
import torch.optim as optim
from comet_ml import Experiment

from model.metric import get_model_accuracy


def train(net, train_loader, test_loader, criterion, device, experiment: Experiment, n_epochs, lr, check_every,
          test_batches):
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    best_acc = get_model_accuracy(net, test_loader, test_batches, device)
    print(f'starting accuracy: {best_acc*100:.2f}%')

    for epoch in range(n_epochs):
        running_loss = 0.0
        n_batches = 0

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

            if n_batches % check_every == check_every - 1:
                acc = get_model_accuracy(net, test_loader, test_batches, device)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(net.state_dict(), f'saved_models/model.pth')
                    print(f'saved new model with {acc * 100:.2f}% accuracy')

                experiment.log_metric('interval_mean_loss', running_loss / check_every)
                experiment.log_metric('interval_accuracy', acc * 100)

                running_loss = 0.0
                n_batches = 0

            n_batches += 1
