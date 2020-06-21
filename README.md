# WAN_PDE

- Implemented [Weak Adversarial Nets](https://arxiv.org/abs/1907.08272) for solving high-dimenstional PDE.
- Formulated time dependent solution as an ODE problem, instead of numerical integration. Backpropagation by adjoint method of neural ODEs.

Results: Faster converge, but at the cost of greater memory consumption during backpropagation by Tracker.jl 


