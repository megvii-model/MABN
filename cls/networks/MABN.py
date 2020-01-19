import torch
import torch.nn as nn
import torch.nn.functional as F

class MABNFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias,
                running_var, eps, momentum,
                sta_matrix, pre_x2, pre_gz, iters
               ):
        ctx.eps = eps
        current_iter = iters.item()
        ctx.iter = current_iter
        N, C, H, W = x.size()

        x = x.view(N//2, 2, C, H, W)
        x2 = (x * x).mean(dim=4).mean(dim=3).mean(dim=1)
        var = torch.cat([pre_x2, x2], dim=0)

        var = torch.mm(sta_matrix, var)
        var = var.view(N//2, 1, C, 1, 1)

        if current_iter == 1:
            var = x2.view(N//2, 1, C, 1, 1)

        z = x /(var + eps).sqrt()
        r = (var + eps).sqrt() / (running_var.view(1, 1, C, 1, 1) + eps).sqrt()
        if current_iter < 100:
            r = torch.clamp(r, 1, 1)
        else:
            r = torch.clamp(r, 1/5, 5)
        y = r * z
        ctx.save_for_backward(z, var, weight, sta_matrix, pre_gz, r)

        if current_iter == 1:
            running_var.copy_(var.mean(dim=0).view(-1,))
        running_var.copy_(momentum*running_var + (1-momentum)*var.mean(dim=0).view(-1,))
        pre_x2.copy_(x2)
        y = weight.view(1,C,1,1) * y.view(N, C, H, W) + bias.view(1,C,1,1)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        current_iter = ctx.iter
        N, C, H, W = grad_output.size()
        z, var, weight, sta_matrix, pre_gz, r  = ctx.saved_variables
        y = r * z
        g = grad_output * weight.view(1, C, 1, 1)
        g = g.view(N//2, 2, C, H, W) * r
        gz = (g * z).mean(dim=4).mean(dim=3).mean(dim=1)

        mean_gz = torch.cat([pre_gz, gz], dim=0)
        mean_gz = torch.mm(sta_matrix, mean_gz)
        mean_gz = mean_gz.view(N//2, 1, C, 1, 1)

        if current_iter == 1:
            mean_gz = gz.view(N//2, 1, C, 1, 1)
        gx = 1. / torch.sqrt(var + eps) * (g - z * mean_gz)
        gx = gx.view(N, C, H, W)
        pre_gz.copy_(gz)

        return gx, (grad_output * y.view(N, C, H, W)).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0),  None, None, None, None, None, None, None

class MABN2d(nn.Module):

    def __init__(self, channels, eps=1e-5, momentum=0.98, buffer_size=16):
        """
            buffer_size: Moving Average Batch Size / Normalization Batch Size
            running_var: EMA statistics of x^2
            buffer_x2: batch statistics of x^2 from last several iters
            buffer_gz: batch statistics of phi from last several iters
            iters: current iter
        """
        super(MABN2d, self).__init__()
        self.B = buffer_size
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.register_buffer('running_var', torch.ones(channels))
        self.register_buffer('sta_matrix', torch.ones(self.B, 2 *self.B)/self.B)
        self.register_buffer('pre_x2', torch.ones(self.B, channels))
        self.register_buffer('pre_gz', torch.zeros(self.B, channels))
        self.register_buffer('iters', torch.zeros(1,))
        self.eps = eps
        self.momentum = momentum
        self.init()

    def init(self):
        for i in range(self.sta_matrix.size(0)):
            self.sta_matrix[i][:i+1] = 0
            self.sta_matrix[i][self.B+i+1:] = 0

    def forward(self, x):
        if self.training:
            self.iters.copy_(self.iters + 1)
            x = MABNFunction.apply(x, self.weight, self.bias,
                                   self.running_var, self.eps, 
                                   self.momentum, self.sta_matrix, 
                                   self.pre_x2, self.pre_gz, 
                                   self.iters)
            return x
        else:
            N, C, H, W = x.size()
            var = self.running_var.view(1, C, 1, 1)
            x = x / (var + self.eps).sqrt()

        return self.weight.view(1,C,1,1) * x + self.bias.view(1,C,1,1)
