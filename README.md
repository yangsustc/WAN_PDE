# WAN_PDE

- Implemented [Weak Adversarial Nets](https://arxiv.org/abs/1907.08272) for solving high-dimenstional PDE.
- Formulated time dependent solution as an ODE problem, instead of numerical integration. Backpropagation by adjoint method of neural ODEs.

## Results:

Both Time dependent and independent PDEs converged well.

#### Elliptic Dirichlet (20 dimensional) 
Note that graph is shown about x1 and x2.

Approximated Function:

[!Approx Function](https://github.com/Ayushk4/WAN_PDE/blob/master/Elliptic_dirichlet_files/dims%3D20/After_20000_Iters.png)

True Function:

![True Function](https://github.com/Ayushk4/WAN_PDE/blob/master/Elliptic_dirichlet_files/dims%3D20/True_function.png)

Re-formulating as ODE problem serves a better approximation than numerical approximation methods.
The typical method of backpropagation through ODEs is adjoint method as mentioned in [Neural Ordinary Differential Equations paper (NIPS 2018) by Chen et al.](https://arxiv.org/abs/1806.07366)

The intuition We can select from a wide variety of ODE solvers.
Faster and better convergence, but at the cost of greater memory consumption during backpropagation by Tracker.jl 

### Possible directions:

**Current limitation** of this approach of solving PDE is huge memory consumption while backpropagating

**Solution to the current limitation** switch autodiff backend from [Tracker.jl](https://github.com/FluxML/Tracker.jl) to [Zygote.jl](https://github.com/FluxML/Zygote.jl). This can be done when DiffEqFlux.jl and Flux.jl switches over to Zygote.jl. 

Zygote.jl uses source code transformation for reverse mode AutoDiff. This leads to faster and much less memory consumption while backpropagating when compared to Tracker.jl's Auto-diff.
