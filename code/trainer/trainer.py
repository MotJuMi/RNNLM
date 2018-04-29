import numpy as np
import torch
import math
from base import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else false
        self.log_step = int(np.sqrt(self.batch_size))

    def _to_variable(self, data, target):
        data, target = torch.LongTensor(data), torch.LongTensor(target)
        if self.with_cuda:
            data, target = data.cuda(), target.cuda()
        return data, target

    def _eval_metrics(self, output, target):
        ppl_metrics = np.zeros(len(self.metrics))
        output = output.cpu().numpy()
        target = target.cpu().numpy()
        for i, metric in enumerate(self.metrics):
            ppl_metrics[i] += metric(output, target)
        return ppl_metrics

    def _train_epoch(self, epoch):
        self.model.train()
        if self.with_cuda:
            self.model.cuda()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = self._to_variable(data, target)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.data[0]
            total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: '
                                 '{:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    len(self.data_loader) * self.data_loader.batch_size,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.data[0]))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            data, target = self._to_variable(data, target)

            output = self.model(data)
            loss = self.loss(output, target)

            total_val_loss += loss.data[0]
            #total_val_metrics += self._eval_metrics(output, target)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': math.exp(total_val_loss / len(self.valid_data_loader))
        }












