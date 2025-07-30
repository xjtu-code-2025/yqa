
import torch
import torch.nn as nn
import torch.nn.functional as F


# 考虑参数padding，stride

class MyConv2d(nn.Module):
    #def __init__(self, in_channels, out_channels, kernel_size):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(MyConv2d, self).__init__()
        # W
        # Parameter可以把一个张量变成一个可学习的参数，乘0.01是为了初始化时的数值不太大
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride


    def forward(self, x):
        batch_size, in_channels, H, W = x.shape
        out_channels = self.weight.shape[0]
        k = self.kernel_size
        s = self.stride
        p = self.padding
        # 步长为1，且不padding
        #out_H = H - k + 1
        #out_W = W - k + 1

        # 考虑padding
        x = F.pad(x, (p, p, p, p))

        out_H = (H + 2 * p - k) // s + 1
        out_W = (W + 2 * p - k) // s + 1   

        output = torch.zeros((batch_size, out_channels, out_H, out_W), device=x.device)

        for b in range(batch_size):
            for oc in range(out_channels):
                for ic in range(in_channels):
                    for i in range(out_H):
                        for j in range(out_W):
                            # 没考虑stride 
                            # region = x[b, ic, i:i + k, j:j + k]
                            # 考虑padding

                            region = x[b, ic, i*s:i*s + k, j*s:j*s + k]
                            output[b, oc, i, j] += torch.sum(region * self.weight[oc, ic])
                output[b, oc] += self.bias[oc]
        return output
class MyBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(MyBatchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean([0, 2, 3])
            var = x.var([0, 2, 3], unbiased=False)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv = MyConv2d(1, 8, 3, padding=1, stride=2)  
        self.bn = MyBatchNorm2d(8)
        self.pool = nn.MaxPool2d(2)
        #滑动的次数：输入-核大小/步长向下取整 + 1

        self.fc = nn.Linear(8 * 7 * 7, 10)  # 28 - 3 + 2//2 + 1 = 14 

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



