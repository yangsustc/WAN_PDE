using LinearAlgebra
using Flux
using Flux: softplus, elu
using Random

d = 5                   # Number of dimensions
Kφ = 1                  # Num iters for adversarial network
Kᵤ = 2                  # Num iters for solutions
τη = 0.04               # Learning rate for adversary network
τθ = 0.015              # Learning rate for primal network
Nr = 4000 * d           # No. of sampled pts in region
Nb = 40 * d * d         # No. of sampled point on boundary
α = 10000               # Weight paramter on boundary
hlsθ = 20               # Hidden Layer size for primal network
hlsη = 50               # Hidden Layer size of adversarial network

a0(x) = 1 + sum(x .^ 2)
ρ0(x) = (π * x[1] * x[1] + x[2] * x[2]) / 2
ρ1(x) = (π * π * x[1] * x[1] + x[2] * x[2]) / 4

# So that we don't recalculate the same terms
f(x, _a, _ρ0, _ρ1, _cos) = 4 * _ρ1 * _a * sin(_ρ0) - 4 * _ρ0 * _cos -
                            (π + 1) * _a * _cos + 2 * _ρ1 * _cos
f(x, _a, _ρ0, _ρ1) = f(x, _a, _ρ0, _ρ1, cos(_ρ0))
f(x) = f(x, a0(x), ρ0(x), ρ1(x))

grad_uθ(x) = Tracker.gradient((x) -> sum(uθ(x)), x; nest=true)[1]
# ∇_dot_a∇u(x) = Tracker.jacobian(x-> a0(x) .* ∇u(x), x)
grad_φη(x) = Tracker.gradient((x) -> sum(φη(x)), x; nest=true)[1]

# grad_
I(x, _∇uθ) = - (_∇uθ' * grad_φη(x))[1] * a0(x) + (φη(x)[1]) * (sum(_∇uθ .^ 2)/2 - f(x))
I(x) = I(x, grad_uθ(x))

# Primal networks - weak solution to PDE
uθ = Flux.Chain(Dense(d, hlsθ, tanh),
                # Dense(hlsθ, hlsθ, tanh),
                # Dense(hlsθ, hlsθ, softplus),
                # Dense(hlsθ, hlsθ, tanh),
                # Dense(hlsθ, hlsθ, softplus),
                # Dense(hlsθ, hlsθ, tanh),
                Dense(hlsθ, 1)
    )

# Adversarial network
φη = Flux.Chain(Dense(d, hlsη, tanh),
                Dense(hlsη, hlsη, tanh),
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, hlsη, sinc),
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, hlsη, sinc),
                Dense(hlsη, hlsη, sinc),
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, 1)
    )

u_true(x) = sin.((π .* view(x, 1, :, :) .^ 2 + view(x, 2, :, :) .^ 2) ./ 2)
g(x) = u_true(x)

# loss_int.(x) =
loss_bndry(x) = sum((uθ(x) .- g(x)') .^ 2)
loss(xr, xb) = loss_int(xr) + α * loss_bndry(xb)
loss(xr) = loss_int(xr)

# optθ = Flux.Optimise.ADAGrad(τθ)
# optη = Flux.Optimise.ADAGrad(τη)

optθ = Flux.Optimise.Descent(τθ)
optη = Flux.Optimise.Descent(τη)

psθ = params(uθ)
psη = params(φη)

# boundary_iter = collect(partition(1:Nb, ))

function train_step()
    # sample points
    xr = [2 .* rand(Float32, d, 1) .- 1 for i in 1:Nb] # Sampling in region
    xb = 2 .* rand(Float32, d, Nb) .- 1
    for i in 1:Nb
        (j=rand(1:10)) <= d ? xb[j, i] = 1 : xb[j - 5, i] = -1
    end

    for i in 1:Kᵤ
        # update weak solution network parameter
        # l = loss(xr, xb)
        gradsθ = Flux.Tracker.gradient(() -> loss(xr, xb), psθ)
        Flux.Tracker.update!(optθ, psθ, gradsθ)
    end

    for i in 1:Kφ
        # update adversarial network parameter
        l = loss(xr)
        gradsη  = Tracker.gradient(() -> l, psη)

        do gradient ascent
        Tracker.update!(optη, psη, gradsη)
    end
end

# Final plot
u_true(x) = sin((π * (x[1] ^ 2) + x[2] ^ 2) / 2)
