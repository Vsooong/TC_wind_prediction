import torch
import os
from CNN.Conv3D import simple3D
args = {
    'model_list': {
        'Conv3D': simple3D,
        # 'CNN2_4': CNN2_4,
        # 'graphCNN': graphCNN,
        # 'convLSTM': convLSTM,
        # 'AGCRN': AGCRN,
        # 'lstmNN': ANN,
    },
    'pretrain': False,
    'n_epochs': 200,
    'learning_rate': 9e-5,
    'lr_decay': True,
    'lr_decay_rate': 0.5,
    'lr_decay_step': [5, 15, 30, 45],
    'batch_size': 12,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'grad_norm': False,
    'max_grad_norm': 5,
    'early_stop_patience': 50,
    '1grid_data': [i for i in ['/home/dl/GSW/data/wind/1grid_wind_1990_2020.nc', r'D:/data/wind/1grid_wind_1990_2020.nc'] if
                   os.path.exists(i)][0],
    'path_list': [
        './tcdata/enso_final_test_data_B/',
        r'D:\data\enso\test样例_20210207_update\test样例',
        '/home/dl/GSW/data/enso/test样例_20210207_update/test样例',
    ]
}

if __name__ == '__main__':
    print(args['1grid_data'])
