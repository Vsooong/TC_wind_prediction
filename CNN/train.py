from Basic_trainer import Trainer
from configs import args
import os
from torch.utils.data import DataLoader, Subset
from data_loader import load_data
import torch
import torch.nn as nn


def train():
    args['model_name'] = 'Conv3D'
    args['batch_size'] = 32
    train_num = 2000
    val_num = 500
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(current_dir, '../experiments', args['model_name'] + '.pth')
    train_set, val_set = load_data(train_num=train_num, val_num=val_num)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'])
    val_loader = DataLoader(val_set, batch_size=args['batch_size'])

    model = args['model_list'][args['model_name']]()
    if args['pretrain'] and os.path.exists(save_dir):
        model.load_state_dict(torch.load(save_dir, map_location=args['device']))
        print('load model from:', save_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'])
    if args['lr_decay']:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in args['lr_decay_step']]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=lr_decay_steps,
                                                            gamma=args['lr_decay_rate'])
    else:
        lr_scheduler = None

    loss_fn = nn.MSELoss().to(args['device'])

    trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader, args, lr_scheduler, save_dir)


if __name__ == "__main__":
    train()
