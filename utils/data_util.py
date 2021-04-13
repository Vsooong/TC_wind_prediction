import xarray as xr
import numpy as np
import os
from configs import args
import torch
import matplotlib.pyplot as plt
import datetime
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import animatplot as amp
from matplotlib import cm
from matplotlib.cm import get_cmap


def resample_data(ori_nc=r'D:\data\wind/refactor/extra_2020.nc', save_dir=r'E:\data'):
    latitude = np.arange(45, -6, -1).astype(float)
    longitude = np.arange(90, 171, 1).astype(float)
    data = xr.open_dataset(ori_nc).sel(latitude=latitude, longitude=longitude)
    new_name = '1grid_' + os.path.basename(ori_nc)
    save_path = os.path.join(save_dir, new_name)
    data.to_netcdf(save_path)
    print(data)


def merge_data(save_dir=[i for i in ['/home/dl/GSW/data/wind/extra/', r'D:\data\wind\refactor'] if os.path.exists(i)][0]):
    years = sorted([i for i in os.listdir(save_dir) if i.startswith('1grid')])
    print(years)
    datas = []
    for year in years:
        path = os.path.join(save_dir, year)
        datas.append(xr.open_dataset(path, decode_times=True))
    to_merge = xr.concat(datas, 'time')
    # # data2 = data2.assign_coords(level=850)
    # # data2 = data2.expand_dims('level')
    to_merge.to_netcdf(os.path.join(save_dir, 'extra_1990_2020.nc'))
    print(to_merge)


def data_hist(num=10000):
    data = xr.open_dataset(args['1grid_data'])
    train_z = data['v'][:num, :].values
    print(train_z.shape)
    train_data = torch.as_tensor(train_z)
    print(torch.mean(train_data))
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.histplot(torch.flatten(train_data))
    # plt.xlim(14000, 15000)
    plt.show()


def wind_vis(start_hour=0, length=100):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(current_dir, '../experiment/plot/wind', 'wind.gif')

    data = xr.open_dataset(args['1grid_data'])
    lon = np.array(data['longitude'].values)
    lat = np.array(data['latitude'].values)

    lon_grid, lat_grid = np.meshgrid(lon, lat, )
    u = data['v'][start_hour, 0].values
    v = data['v'][start_hour, 0].values
    # # ax = fig.gca(projection='3d')
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.set_tight_layout(True)
    #
    arrow = ax.quiver(lon_grid, lat_grid, u, v, units='width')

    def update(t):
        label = 'timestep {0}'.format(t)
        now_date = startdate + datetime.timedelta(days=int(t / 4), hours=6 * int(t % 4))
        print(now_date.strftime("%Y-%m-%d, %H:%M:%S"))
        u = data['v'][t, 0].values
        v = data['v'][t, 0].values
        arrow.set_UVC(u, v)
        return arrow,

    anim = FuncAnimation(fig, update, frames=np.arange(start_hour, start_hour + length), interval=50, repeat=False)
    anim.save(save_dir, dpi=100, writer=PillowWriter(fps=10))
    fig.show()

    # # ax = fig.add_subplot(111, projection='3d')
    # fig.savefig('3_quiver_plots.png', dpi=300, bbox_inches='tight')


def pressure_vis(start_hour=0, length=100, height=2):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(current_dir, '../experiment/plot/wind', 'pressure.gif')

    data = xr.open_dataset(args['1grid_data'])
    lon = np.array(data['longitude'].values)
    lat = np.array(data['latitude'].values)

    lon_grid, lat_grid = np.meshgrid(lon, lat, )
    geo = data['z'][start_hour, height].values
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})
    fig.set_tight_layout(True)
    surf = ax.plot_surface(lon_grid, lat_grid, geo, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_zlim(29000, 32000)
    ax.view_init(15, -20)

    def animate(t):
        # Delete the existing contour
        ax.collections = []
        # Plot the ith contour
        ax.plot_surface(lon_grid, lat_grid, data['z'][t, height].values, cmap=cm.coolwarm, linewidth=0,
                        antialiased=False)

    anim = FuncAnimation(fig, animate, frames=np.arange(start_hour, start_hour + length), interval=50, repeat=False)
    anim.save(save_dir, dpi=100, writer=PillowWriter(fps=10))
    fig.show()


startdate = datetime.date(1990, 1, 1)

if __name__ == '__main__':
    enddate = datetime.date(1990, 9, 13)
    delta = 4 * (enddate - startdate).days
    # wind_vis(delta, 36)
    # pressure_vis(delta, 36)
    # resample_data()
    # merge_data()
    data_hist()
    # data = xr.open_dataset(r'D:\data\wind/1grid_wind_1990_2020.nc')
    # print(data)

    # plt.imshow(data['z'][1000, 0].values)
    # plt.show()
