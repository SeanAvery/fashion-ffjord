import torch
import torch.nn as nn
import torch.optim as optim
import os
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

if __name__ == '__main__':
    i = 0

    # 1. create visualizer
    create_visualizer()

    # 2. create ode net
    func = ODEFunc()
    
