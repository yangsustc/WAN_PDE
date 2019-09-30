using LinearAlgebra
using Flux

d = 5           # Number of dimensions
Kφ = 1          # Num iters for adversarial network
Kᵤ = 2          # Num iters for solutions
τη = Float32(0.04)       # Learning rate for adversary network
τθ = Float32(0.015)      # Learning rate for primal network
Nr = 4000 * d   # No. of sampled pts in region
Nb = 40 * d * d # No. of sampled point on boundary
α = 10000 * Nb  # Weight paramter on boundary
hlsθ = 20       # Hidden Layer size for primal network
hlsη = 50       # Hidden Layer size of adversarial network

a0(x) = 1 + sum(x .^ 2)
ρ0(x) = (π * x[1] * x[1] + x[2] * x[2]) / 2
ρ1(x) = (π * π * x[1] * x[1] + x[2] * x[2]) / 4

f(x, _a, _ρ0, _ρ1, _cos) = 4 * _ρ1 * _a * sin(_ρ0) - 4 * _ρ0 * _cos -
                            (π + 1) * _a * _cos + 2 * _ρ1 * _cos
# So that we don't recalculate the same terms
f(x, _a, _ρ0, _ρ1) = f(x, _a, _ρ0, _ρ1, cos(_ρ0))
f(x) = f(x, a0(x), ρ0(x), ρ1(x))

∇u() =
∇_dot_a∇u(x) =
lhs(x) = -∇_dot_a∇u(x) + sum(∇u . ^ 2)/2 - f(x)

# Primal networks - weak solution to PDE
uθ = Flux.Chain(Dense(d, hlsθ, tanh),
                Dense(hlsθ, hlsθ, tanh),
                Dense(hlsθ, hlsθ, elu),
                Dense(hlsθ, hlsθ, tanh),
                Dense(hlsθ, hlsθ, elu),
                Dense(hlsθ, hlsθ, tanh),
                Dense(hlsθ, 1)
    )

# Primal networks - weak solution to PDE
φη = Flux.Chain(Dense(d, hlsη, tanh),
                Dense(hlsη, hlsη, tanh), sinc,
                Dense(hlsη, hlsη, softplus), sinc,
                Dense(hlsη, hlsη, tanh),
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, hlsη, tanh),
                Dense(hlsη, hlsη, tanh), sinc,
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, hlsη),
                Dense(hlsη, 1)
    )


# loss_int(x) =
loss_bndry(x) = sum((uθ(x) .- g(x)) .^ 2)) / Nbd = 5           # Number of dimensions
loss(x) = loss_int(x) + α * loss_bndry(x)

optθ = Flux.Optimise.ADAGrad(η = τθ)
optη = Flux.Optimise.ADAGrad(η = τη)

psθ = params(uθ)
psη = params(φη)

function train_step()
    #sample, then make it param

    # Sampling
    xr = sample_region()
    xb = sample_boundary()

    for i in 1:Kᵤ
        # update weak solution network parameter
        l = loss(x)
        gradsθ = Tracker.gradient(() -> l, psθ)
        Tracker.update!(optθ, psθ, gradsθ)
    end

    for i in 1:Kφ
        # update adversarial network parameter
        l = loss(x)
        gradsη  = Tracker.gradient(() -> l, psη)
        Tracker.update!(optη, psη, gradsη)
    end
end

# Final plott
u_true(x) = sin((π * (x[1] ^ 2) + x[2] ^ 2) / 2)
