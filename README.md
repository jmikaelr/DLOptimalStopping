
# Optimal Stopping Problem for American Options using Deep Neural Networks

## Project Overview

This project implements a deep neural network approach to solve the optimal stopping problem for American options. The framework employs a Monte Carlo simulation of the underlying asset price process and leverages neural networks to approximate the optimal stopping rule, maximizing the payoff under the constraint of early exercise.

The deep learning approach to American option pricing addresses the challenges associated with high-dimensional problems, where traditional methods (like finite-difference methods) struggle with the "curse of dimensionality." This project explores the use of reinforcement learning principles applied in the finance domain, following Longstaff-Schwartz and deep reinforcement learning paradigms.

---

## Mathematical Background

The primary objective is to find the optimal stopping time $\tau$ that maximizes the expected payoff $E[e^{-r \tau} \Phi(S_{\tau})]$, where $S_t$ is the underlying asset price, and $\Phi$ is the payoff function of the option. For an American put option, the payoff function is defined as:
$$\Phi(S_t) = \max(K - S_t, 0)$$
where $K$ is the strike price.

Given the stochastic process $S_t$ of the underlying asset, the option value $V(t, S_t)$ can be formulated by solving the optimal stopping problem:
$$V(t, S_t) = \sup_{\tau \in [t, T]} \mathbb{E} \left[ e^{-r(\tau - t)} \Phi(S_{\tau}) \mid S_t \right]$$

Here, $r$ is the risk-free interest rate, and $\tau$ is the stopping time chosen to maximize the payoff. By backward induction through Monte Carlo paths, we estimate the continuation value at each step, approximating the optimal stopping rule.

### Monte Carlo Simulation of the Price Process

The underlying price dynamics are simulated using a discretized Euler scheme for a Geometric Brownian Motion:
$$S_{t+1} = S_t \exp\left((r - \frac{\sigma^2}{2}) \Delta t + \sigma \sqrt{\Delta t} \, Z_t \right)$$
where:
- $r$ = risk-free rate
- $\sigma$ = volatility of the underlying asset
- $\Delta t$ = time step size
- $Z_t$ = i.i.d. standard normal random variables.

The Cholesky decomposition is applied to handle correlated underlying assets for multi-dimensional cases.

### Neural Network Approximation

The optimal stopping rule is approximated using a neural network with a reinforcement learning-inspired objective:
1. **Input Layer**: Receives state variables, including the simulated paths and immediate payoffs.
2. **Hidden Layers**: Capture complex interactions between states and potential stopping decisions.
3. **Output Layer**: Produces an estimate of the stopping rule, where each node represents a time step.

The training optimizes the expected payoff using gradient-based optimization (e.g., Adam optimizer), updating weights to improve stopping rule accuracy.

---

## References

For more details, please refer to the [thesis document](https://mdh.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=14&af=%5B%5D&searchType=LIST_LATEST&sortOrder2=title_sort_asc&query=&language=sv&pid=diva2%3A1897364&aq=%5B%5B%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=4765) for the full theoretical background, implementation details, and findings.

---

This project serves as a demonstration of applying deep learning techniques to complex financial problems, specifically focusing on the optimal stopping theory applied to American option pricing.
