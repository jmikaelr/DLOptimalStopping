import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd
from collections import Counter
from threading import Thread
import psutil
import os

class MemoryMonitor(Thread):
    def __init__(self):
        super().__init__()
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.running = True

    def run(self):
        while self.running:
            current_memory = self.process.memory_info().rss / (1024 ** 2)  # Convert to MB
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            time.sleep(0.01)  # Check memory every 10ms

    def stop(self):
        self.running = False

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
        s0 = self.s0
        r = self.r
        dividend = self.dividend
        sigma = self.sigma
        d = self.d
        m = 8192  # Use the same batch size as in TensorFlow
        n = self.n
        correlation = self.correlation
        t_mat = self.t_mat
        dt = t_mat / n
        q = np.ones([d, d], dtype=np.float32) * correlation
        np.fill_diagonal(q, 1.)
        q = torch.tensor(q, dtype=torch.float32).transpose(0, 1)
        l = torch.linalg.cholesky(q)
        w = torch.matmul(torch.randn(m * n, d, dtype=torch.float32) * np.sqrt(t_mat / n), l)
        w = w.view(m, n, d).permute(0, 2, 1)
        w = torch.cumsum(w, dim=2)
        t = torch.tensor(np.linspace(start=t_mat / n, stop=t_mat, num=n, endpoint=True), dtype=torch.float32)
        s = torch.exp((r - dividend - sigma ** 2 / 2) * t + sigma * w) * s0
        return s

class NeuralNetwork(nn.Module):
    def __init__(self, i, l1, l2, epsilon=1e-6, decay=0.9):
        super().__init__()
        self.l1 = nn.Linear(i, l1, dtype=torch.float32)
        self.bn1 = nn.BatchNorm1d(l1, eps=epsilon, momentum=(1 - decay), dtype=torch.float32)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(l1, l2, dtype=torch.float32)
        self.bn2 = nn.BatchNorm1d(l2, eps=epsilon, momentum=(1 - decay), dtype=torch.float32)
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
        x = x.permute(2, 0, 1)
        y = self.l1(x).permute(1, 2, 0)
        y = self.bn1(y)
        y = self.relu1(y).permute(2, 0, 1)
        y = self.l2(y).permute(1, 2, 0)
        y = self.bn2(y)
        y = self.relu2(y).permute(2, 0, 1)
        y = self.l3(y).permute(1, 2, 0)
        y = self.sigmoid(y)
        return y

def iteration(model, underlying_process, simulation, payoffs, device, beta1=0.9, beta2=0.999):
    Z = torch.cat([simulation[:, :, :-1], payoffs[:, :, :-1]], dim=1).to(device)
    payoffs = payoffs.to(device)

    lr_values = [0.005, 0.0005, 0.00005]
    lr_boundaries = [400, 800]
    initial_lr = lr_values[0]
    epsilon = 1e-8
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, betas=(beta1, beta2), eps=epsilon)
    scheduler = MultiStepLR(optimizer, milestones=lr_boundaries, gamma=0.1)
    epochs = 300

    n_steps = underlying_process.n
    dt = underlying_process.t_mat / n_steps

    epoch_list = []
    losses = []
    tau_values = []

    batch_size = Z.size(0)

    for epoch in range(epochs):
        forward_pass = model.forward(Z)
        u_list = [forward_pass[:, :, 0]]
        u_sum = u_list[-1]
        for k in range(1, n_steps - 1):
            u_list.append(forward_pass[:, :, k] * (1. - u_sum))  # equation 54
            u_sum = u_sum + u_list[-1]

        u_list.append(1. - u_sum)
        u_stack = torch.cat(u_list, dim=1)

        u_cumsum = torch.cumsum(u_stack, dim=1)
        condition = (u_cumsum + u_stack >= 1).to(torch.uint8)
        tau = torch.argmax(condition, dim=1)
        tau_values.extend(tau.cpu().numpy())

        batch_payoffs = payoffs.view(batch_size, -1)  # payoffs (m, n)
        loss = torch.mean(torch.sum(-u_stack * batch_payoffs, dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()


        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Option Price: {-loss.item()}'")
            epoch_list.append(epoch)
            losses.append(round(-loss.item(), 4))

    tau_counter = Counter(tau_values)
    most_common_tau, _ = tau_counter.most_common(1)[0]


    return forward_pass, model

def evaluate_model(model, simulation, payoffs, device, mc_runs):
    model.eval()
    px_mean = 0.0
    simulation = simulation.to(device)
    payoffs = payoffs.to(device)
    with torch.no_grad():
        for _ in range(mc_runs):
            Z = torch.cat([simulation[:, :, :-1], payoffs[:, :, :-1]], dim=1).to(device)
            forward_pass = model.forward(Z)

            u_list = [forward_pass[:, :, 0]]
            u_sum = u_list[-1]
            for k in range(1, simulation.size(2) - 1):
                u_list.append(forward_pass[:, :, k] * (1. - u_sum))
                u_sum = u_sum + u_list[-1]
            u_list.append(1. - u_sum)
            u_stack = torch.cat(u_list, dim=1)

            u_cumsum = torch.cumsum(u_stack, dim=1)
            condition = (u_cumsum + u_stack >= 1).to(torch.uint8)
            tau = torch.argmax(condition, dim=1)

            gathered_values = payoffs.squeeze(1).gather(1, tau.unsqueeze(1)).squeeze(1)
            stopped_payoffs = gathered_values.mean()
            px_mean += stopped_payoffs.item()

    px_mean /= mc_runs
    print("Mean stopped payoffs:", px_mean)
    return px_mean

def g(x, k, k2=105):
    return (torch.maximum(torch.max(x, dim=1, keepdim=True).values - k, torch.tensor(0.0, dtype=torch.float32)) -2*torch.maximum(torch.max(x, dim=1, keepdim=True).values - k2, torch.tensor(0.0, dtype=torch.float32)))
#return torch.maximum(k - torch.min(x, dim=1, keepdim=True).values, torch.tensor(0.0, dtype=torch.float32))
#    return torch.maximum(torch.max(x, dim=1, keepdim=True).values - k, torch.tensor(0.0, dtype=torch.float32))

def solve(underlying_process, device):
    underlying_process_simulation = underlying_process.euler_scheme()
    payoffs = g(underlying_process_simulation, underlying_process.strike)
    torch.set_default_device(device)
    model = NeuralNetwork(underlying_process.d + 1,
                          underlying_process.d + 50,
                          underlying_process.d + 50, decay=0.9)

    simulation = underlying_process_simulation
    forward_pass, model = iteration(model, underlying_process, simulation, payoffs, device)
    mc_runs = 1500
    px_mean = evaluate_model(model, simulation, payoffs, device, mc_runs)
    return px_mean

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    s0 = 100
    sigma = 0.2
    r = 0.06
    dividend = 0
    correlation = 0
    strike = 95
    t_mat = 0.25
    m = 8192
    N = [8,20,150]
    d = 2
    print("Solving an American put with values:\n"
          f"S: {s0}\n"
          f"Sigma: {sigma}\n"
          f"r: {r}\n"
          f"Dividend: {dividend}\n"
          f"Correlation: {correlation}\n"
          f"Strike: {strike}\n"
          f"T: {t_mat}\n"
          f"N: {N}\n"
          f"M: {m}\n"
          f"Dims: {d}\n"
          )

    results = []
    for n in N:
        print(f"Running for n={n}")
        memory_monitor = MemoryMonitor()
        memory_monitor.start()
        start_time = time.time()

        underlying_process = MonteCarloSimulation(s0, sigma, r,
                                                  dividend, correlation, strike,
                                                  t_mat, n, m, d)

        price = solve(underlying_process, device)

        memory_monitor.stop()
        memory_monitor.join()

        end_time = time.time()
        est_time = end_time - start_time
        peak_memory = memory_monitor.peak_memory

        print(f"n: {n}, Price: {price:.4f}, Time: {est_time:.4f}, Memory: {peak_memory:.4f} MB")

        # Save the result for this iteration
        results.append({
            "n": n,
            "Price": round(price, 4),
            "Time (s)": round(est_time, 4),
            "Memory (MB)": round(peak_memory, 4)
        })

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Reverse the Memory (MB) column
    df["Memory (MB)"] = df["Memory (MB)"].iloc[::-1].reset_index(drop=True)

    # Save to CSV with 4 decimal precision
    df.to_csv("american_put_results.csv", index=False, float_format='%.4f')
    print("Results saved to american_put_results.csv")



if __name__ == '__main__':
    main()

