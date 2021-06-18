import torch


def get_model_accuracy(net, test_loader, test_batches, device):
    with torch.no_grad():
        i = 0
        running_accuracy = 0.0
        for data, labels in test_loader:
            if i == test_batches:
                break

            preds = net(data.to(device))
            running_accuracy += calculate_accuracy(preds, labels.to(device))
            i += 1

        return running_accuracy / test_batches


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output: torch.Tensor, target: torch.Tensor, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
