import time
import torch
from copy import deepcopy
from utils.train_util import init_seed, print_model_parameters
from configs import args
import os


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, args=args, lr_scheduler=None, save_dir=None):
        super(Trainer, self).__init__()
        self.args = args
        self.device = self.args['device']
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.save_dir = save_dir

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_sample = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device).float()
            target = target.to(self.device).float()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            total_loss += loss.item()
            total_sample += len(data)
            if self.args['grad_norm']:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
            self.optimizer.step()
            del loss, output
        if self.args['lr_decay'] and self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return total_loss / total_sample

    def val_epoch(self):
        self.model.eval()
        total_val_loss = 0
        total_sample = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data = data.to(self.device).float()
                target = target.to(self.device).float()
                output = self.model(data)
                loss = self.loss(output, target)
                total_val_loss += loss.item()
                total_sample += len(data)
                del output, loss
        return total_val_loss / total_sample

    def train(self):
        init_seed(1995)
        print_model_parameters(self.model)
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        # start_time = time.time()

        for i in range(1, self.args['n_epochs'] + 1):
            train_epoch_loss = self.train_epoch()
            val_epoch_loss = self.val_epoch()
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            print('Epoch: {}, Train Loss: {}, Valid Score: {}'.format(i, train_epoch_loss, val_epoch_loss))
            if train_epoch_loss > 1e6:
                print('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            if not_improved_count == self.args['early_stop_patience']:
                print("Validation performance didn\'t improve for {} epochs. "  "Training stops.".format(
                    args['early_stop_patience']))

            if best_state and os.path.exists(self.save_dir):
                best_model = deepcopy(self.model.state_dict())
                torch.save(best_model, self.save_dir)
                print('Model saved successfully:', self.save_dir)

    def test(self):
        pass
