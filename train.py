from comet_ml import Experiment
import torch.cuda
import data_loader as data
import trainer as trainer
from model import model, loss
import json


def train(experiment, data_path, batch_size, n_epochs):
    torch.cuda.empty_cache()
    (train_set, test_set), (train_loader, test_loader) = data.load_data(data_path, batch_size)
    criterion = loss.get_loss()
    net = model.Net(len(train_set.classes)).type(torch.float32)

    trainer.train(net, loader=train_loader, criterion=criterion, device='cuda', experiment=experiment, n_epochs=n_epochs)


if __name__ == '__main__':
    config = json.load('config.json')
    e = Experiment(
        api_key=config['comet_api_key'],
        project_name=config['comet_project_name'],
        workspace=config['comet_workspace'],
    )
    train(e, *config)
