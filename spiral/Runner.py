import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import matplotlib.pyplot as plt

def create_visualizer():
    # create image dir container
    if not os.path.exists('imgs'):
        os.makedirs('imgs')
    
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2))
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        def forward(self, t, y):
            return self.net(y**3)

class RunningAverageMeter(object):
    def __init__( self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

if __name__ == '__main__':
    i = 0
    device = 'cpu' # running on macbook rn
    true_y = torch.tensor([[2., 0.]])
    t = torch.linspace(0., 25., 1000) # data size 1000 
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]) 
    
    # 1. create visualizer
    create_visualizer()

    # 2. create ode net
    func = ODEFunc()
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)   
    end = time.time()
    
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)







