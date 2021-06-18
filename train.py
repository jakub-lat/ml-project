from comet_ml import Experiment
import torch.cuda
import data_loader as data
import trainer as trainer
from model import model, loss
import json


def train(experiment, data_path, batch_size, n_epochs, lr, check_every, test_batches, **kwargs):
    torch.cuda.empty_cache()
    (train_set, test_set), (train_loader, test_loader) = data.load_data(data_path, batch_size)
    criterion = loss.get_loss()
    net = model.Net(len(train_set.classes)).type(torch.float32)

    trainer.train(net,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  criterion=criterion,
                  device='cuda',
                  experiment=experiment,
                  n_epochs=n_epochs,
                  lr=lr,
                  check_every=check_every,
                  test_batches=test_batches,
                  )


if __name__ == '__main__':
    f = open('config.json')
    config = json.load(f)
    e = Experiment(
        api_key=config['comet_api_key'],
        project_name=config['comet_project_name'],
        workspace=config['comet_workspace'],
    )
    train(e, **config)
