import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import os
import matplotlib.pyplot as plt

from torch.profiler import profile, record_function, ProfilerActivity

class MonteCarloSimulation():
    def __init__(self, s0, sigma, r, dividend, correlation, strike, t_mat, n, m, d):
        self.s0 = s0
        self.sigma = sigma
        self.r = r
        self.dividend = dividend
        self.correlation = correlation
        self.strike = strike
        self.t_mat = t_mat
        self.n = n
        self.m = m
        self.d = d

    def euler_scheme(self):
        """ Simulates the stock price that follows a GBM with Euler scheme """
        s0 = self.s0
        r = self.r
        dividend = self.dividend
        sigma = self.sigma
        d = self.d
        m = self.m
        m = 8192
        n = self.n
        correlation = self.correlation
        t_mat = self.t_mat
        dt = t_mat/n
        q = np.ones([d, d], dtype=np.float32) * correlation
        np.fill_diagonal(q, 1.)
        q = torch.tensor(q, dtype = torch.float32).transpose(0,1)
        l = torch.linalg.cholesky(q)
        w = torch.matmul(torch.randn(m * n, d, dtype=torch.float32) * np.sqrt(t_mat / n), l)
        w = w.view(m, n, d).permute(0, 2, 1)
        w = torch.cumsum(w, dim=2)
        t = torch.tensor(np.linspace(start=t_mat / n, stop=t_mat, num=n, endpoint=True), dtype=torch.float32)
        s = torch.exp((r - dividend - sigma ** 2 / 2) * t + sigma * w) * s0
        return s

class NeuralNetwork(nn.Module):
    """ L1 and L2 d + 50 """
    """ Input layer d + 1 """
    def __init__(self, i, l1, l2, epsilon = 1e-6):
        super().__init__()
        self.l1 = nn.Linear(i, l1, dtype=torch.float32)
        self.bn1 = nn.BatchNorm1d(l1, epsilon, dtype=torch.float32)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(l1, l2, dtype=torch.float32)
        self.bn2 = nn.BatchNorm1d(l2, epsilon, dtype=torch.float32)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(l2, 1, dtype=torch.float32)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = x.permute(2,0,1)
        y = self.l1(x).permute(1, 2, 0)
        y = self.bn1(y)
        y = self.relu1(y).permute(2, 0, 1)
        y = self.l2(y).permute(1, 2, 0)
        y = self.bn2(y)
        y = self.relu2(y).permute(2, 0, 1)
        y = self.l3(y).permute(1, 2, 0)
        y = self.sigmoid(y)
        
        return y

def iteration(model, underlying_process, simulation, payoffs, device):
    Z = torch.cat([simulation[:,:,:-1], payoffs[:,:,:-1]], dim=1).to(device)
    payoffs = payoffs.to(device)

    lr_values = [0.001, 0.0001]
    lr_boundaries = [400, 800]
    epsilon = 1e-8
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_values[0], eps=epsilon)
    scheduler = MultiStepLR(optimizer, milestones=lr_boundaries, gamma=0.1)
    epochs = 500
    epoch_times = []
    losses = []
    
    for epoch in range(epochs):
        start_time = time.time()
        forward_pass = model.forward(Z)
        u_list = [forward_pass[:, :, 0]]
        u_sum = u_list[-1]
        for k in range(1, underlying_process.n - 1):
            u_list.append(forward_pass[:, :, k] * (1. - u_sum))
            u_sum = u_sum + u_list[-1]

        u_list.append(1. - u_sum)
        u_stack = torch.cat(u_list, dim=1)

        batch_size = Z.size(0)
        batch_payoffs = payoffs.view(batch_size, -1)
        loss = torch.mean(torch.sum(-u_stack * batch_payoffs, dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        end_time = time.time()

        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)
        losses.append(-loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Option Price: {-loss.item()}, Average Epoch Time {np.mean(epoch_times)}")
            epoch_times = []

    plt.figure()
    plt.plot(range(epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Option Price')
    plt.title('Option Price over Epochs')
    plt.show()

    return forward_pass, model

def g(x, k, k2=105):
    return torch.maximum(torch.max(x, dim=1, keepdim=True).values - k, torch.tensor(0.0, dtype=torch.float32)) - 2 * torch.maximum(torch.max(x, dim = 1, keepdim=True).values -k2, torch.tensor(0.0, dtype=torch.float32))
    return torch.maximum(k - torch.max(x, dim=1, keepdim=True).values, torch.tensor(0.0, dtype=torch.float32))

def solve(underlying_process, device):
    underlying_process_simulation = underlying_process.euler_scheme()
    payoffs = g(underlying_process_simulation, underlying_process.strike)
    torch.set_default_device(device)
    model = NeuralNetwork(underlying_process.d + 1,
                         underlying_process.d + 50,
                         underlying_process.d + 50)

    simulation = underlying_process_simulation
    forward_pass, model = iteration(model, underlying_process, simulation, payoffs, device)
    
    # Evaluate model
    evaluate_model(model, underlying_process, device)

def evaluate_model(model, underlying_process, device):
    with torch.no_grad():
        torch.set_default_device("cpu")
        evaluation_simulation = underlying_process.euler_scheme()
        evaluation_payoffs = g(evaluation_simulation, underlying_process.strike)
        torch.set_default_device(device)
        
        Z_eval = torch.cat([evaluation_simulation[:,:,:-1], evaluation_payoffs[:,:,:-1]], dim=1).to(device)
        
        model.eval()
        predictions = model.forward(Z_eval)
        
        u_list = [predictions[:, :, 0]]
        u_sum = u_list[-1]
        for k in range(1, underlying_process.n - 1):
            u_list.append(predictions[:, :, k] * (1. - u_sum))
            u_sum = u_sum + u_list[-1]

        u_list.append(1. - u_sum)
        u_stack = torch.cat(u_list, dim=1)
        
        batch_size = Z_eval.size(0)
        batch_payoffs = evaluation_payoffs.view(batch_size, -1).to(device)
        loss = torch.mean(torch.sum(-u_stack * batch_payoffs, dim=1))
        
        print(f"Option Price: {-loss.item()}")

        plt.figure()
        plt.scatter(u_stack.cpu().numpy().flatten(), evaluation_payoffs.cpu().numpy().flatten(), alpha=0.5)
        plt.xlabel('Model Predictions')
        plt.ylabel('Actual Payoffs')
        plt.title('Model Predictions vs Actual Payoffs')
        plt.show()

def main():
    start_time = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    s0 = 100
    sigma = 0.2
    r = 0.01
    dividend = 0
    correlation = 0.5
    strike = 100
    t_mat = 0.25
    n = 50
    m = 8192
    d = 2
    print("Solving an American put with values:\n"
        f"S: {s0}\n"
        f"Sigma: {sigma}\n"
        f"r: {r}\n"
        f"Dividend: {dividend}\n"
        f"Correlation: {correlation}\n"
        f"Strike: {strike}\n"
        f"T: {t_mat}\n"
        f"N: {n}\n"
        f"M: {m}\n"
        f"Dims: {d}\n"
        )

    underlying_process = MonteCarloSimulation(s0, sigma, r,
                                             dividend, correlation, strike,
                                            t_mat, n, m, d)

    solve(underlying_process, device)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"It took {execution_time} seconds")
    

if __name__ == '__main__':
    main()

