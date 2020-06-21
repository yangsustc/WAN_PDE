# WAN_PDE

- Implemented [Weak Adversarial Nets](https://arxiv.org/abs/1907.08272) for solving high-dimenstional PDE.
- Formulated time dependent solution as an ODE problem, instead of numerical integration. Backpropagation by adjoint method of neural ODEs.

### Results:

Faster and better convergence, but at the cost of greater memory consumption during backpropagation by Tracker.jl 

### Possible directions:

**Current limitation** of this approach of solving PDE is huge memory consumption while backpropagating

A possible direction is to switch autodiff backend to Zygote.jl instead of Tracker.jl, this is expected to overcome the current limitation.

Zygote.jl uses source code transformation for reverse mode AutoDiff. This leads to faster and much less memory consumption while backpropagating when compared to Tracker.jl's Auto-diff.
