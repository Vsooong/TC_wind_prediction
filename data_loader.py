import os
import torch
import xarray as xr
from torch.utils.data import Dataset
from configs import args
import numpy as np


# current_dir = os.path.dirname(os.path.realpath(__file__))
# print(current_dir)
def geo_standard_para():
    z_middle = [114400, 57480, 30820, 14750]
    z_range = [13000, 8000, 5000, 3000]
    return np.array(z_middle), np.array(z_range)


def load_data(data_dir=args['1grid_data'], train_num=2000, val_num=500):
    data = xr.open_dataset(data_dir)
    z_middle, z_range = geo_standard_para()
    train_z = data['z'][:train_num].values
    train_z = (train_z - z_middle[:, np.newaxis, np.newaxis]) / z_range[:, np.newaxis, np.newaxis]
    train_u = data['u'][:train_num].values / 100
    train_v = data['v'][:train_num].values / 50
    val_z = data['z'][-val_num:].values
    val_z = (val_z - z_middle[:, np.newaxis, np.newaxis]) / z_range[:, np.newaxis, np.newaxis]
    val_u = data['u'][-val_num:].values / 100
    val_v = data['v'][-val_num:].values / 50

    # print(np.max(val_z), np.min(val_z))
    print('Train samples: {}, Valid samples: {}'.format(train_num - 17, val_num - 17))
    dict_train = {
        'u': train_u,
        'v': train_v,
        'z': train_z,
    }
    dict_valid = {
        'u': val_u,
        'v': val_v,
        'z': val_z,
    }

    train_dataset = WindDataset(dict_train)
    valid_dataset = WindDataset(dict_valid)
    return train_dataset, valid_dataset


class WindDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['u']) - 18

    def __getitem__(self, idx):
        return self.data['u'][idx:idx + 6], self.data['v'][idx:idx + 6], self.data['z'][idx:idx + 6], \
               self.data['u'][idx + 6:idx + 18], self.data['v'][idx + 6:idx + 18], self.data['z'][idx + 6:idx + 18]


if __name__ == '__main__':
    train_set, val_set = load_data()
    from torch.utils.data import DataLoader
    import time

    start_time = time.time()
    # train_dataset = load_train_data('cmip')
    # valid_dataset = load_val_data('soda')
    train_loader = DataLoader(train_set, batch_size=args['batch_size'])
    for step, (u, v, z, u_tar, v_tar, z_tar) in enumerate(train_loader):
        print(len(u))
    #     print(z[0, 0, 0])
