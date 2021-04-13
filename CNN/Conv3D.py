import torch.nn as nn
import torch


class simple3D(nn.Module):
    def __init__(self, hidden_planes=64):
        super(simple3D, self).__init__()
        self.planes = hidden_planes
        self.conv1 = nn.Conv3d(5, 5, kernel_size=(3, 3, 3), stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv3d(5, 5, kernel_size=(3, 3, 3), stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv3d(5, 5, kernel_size=(3, 3, 3), stride=1, padding=1, bias=True)

    def forward(self, u, v, z):
        u = u[:, 1:, ...]
        v = v[:, 1:, ...]
        z = z[:, 1:, ...]
        x = self.conv1(u)
        return x


if __name__ == '__main__':
    devcie = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = simple3D().to(devcie)
    input1 = torch.rand(12, 6, 4, 51, 81).to(devcie)
    input2 = torch.rand(12, 6, 4, 51, 81).to(devcie)
    input3 = torch.rand(12, 6, 4, 51, 81).to(devcie)
    # print_model_parameters(model)
    x = model(input1, input2, input3)
    print(x.shape)
