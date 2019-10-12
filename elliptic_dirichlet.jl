using LinearAlgebra
using Flux
using Flux: softplus
using Random
using Plots
using BSON: @save
using CuArrays

d = 5                   # Number of dimensions
Kφ = 1                  # Num iters for adversarial network
Ku = 2                  # Num iters for solutions
τη = 0.04               # Learning rate for adversary network
τθ = 0.015              # Learning rate for primal network
Nr = 4000 * d           # No. of sampled pts in region
Nb = 40 * d * d         # No. of sampled point on boundary
α = 10000 * Nb          # Weight paramter on boundary
hlsθ = 20               # Hidden Layer size for primal network
hlsη = 50               # Hidden Layer size of adversarial network

a0(x) = 1 .+ sum(x .^ 2, dims= 1)
ρ0(x) = (π .* (view(x, 1, :, :) .^ 2) .+ (view(x, 2, :, :) .^ 2)) ./ 2
ρ1(x) = (Float16(π) .^ 2 .* (view(x, 1, :, :) .^ 2) .+ (view(x, 2, :, :) .^ 2)) ./ 4

# So that we don't recalculate the same terms
f(x, _a, _ρ0, _ρ1, _cos) = 4 .* _ρ1 .* _a .* sin.(_ρ0) .+
                            (2 .* _ρ1 .- 4 .* _ρ0 .- (π + 1) .* _a) .* _cos
f(x, _a, _ρ0, _ρ1) = f(x, _a, _ρ0, _ρ1, cos.(_ρ0))
f(x, _a) = f(x, _a, ρ0(x)', ρ1(x)')

# grad_uθ(x) = Tracker.gradient((x) -> sum(uθ(x)), x; nest=true)[1]
# grad_φη(x) = Tracker.gradient((x) -> sum(φη(x)), x; nest=true)[1]

grad_uθ(x) = Tracker.forward(x -> uθ(x), x)[2](1)[1]
grad_φη(x) = Tracker.forward(x -> φη(x), x)[2](1)[1]

function I(x, _∇uθ, _φη, _a)
    t1 = sum(_∇uθ .* grad_φη(x), dims = 1) .* _a
    t2 = sum(_∇uθ .* _∇uθ, dims = 1)  ./ 2
    (t2  .- f(x, _a)) .* _φη .- t1
end

I(x, _φη) = I(x, grad_uθ(x), _φη, a0(x))

# Primal network - weak solution to PDE

uθ = Flux.Chain(Dense(d, hlsθ), x -> tanh.(x),
                Dense(hlsθ, hlsθ), x -> tanh.(x),
                Dense(hlsθ, hlsθ, softplus),
                Dense(hlsθ, hlsθ), x -> tanh.(x),
                Dense(hlsθ, hlsθ, softplus),
                Dense(hlsθ, hlsθ), x -> tanh.(x),
                Dense(hlsθ, 1)
    ) |> gpu

# Adversarial network
ϵ = Float32(1e-7)

_sinc_custom(x) = @. sin(x) / (x + ϵ) # Speed on GPUs during backprop

φη = Flux.Chain(Dense(d, hlsη), x -> tanh.(x),
                Dense(hlsη, hlsη), x -> tanh.(x),
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, hlsη), x -> _sinc_custom(x),
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, hlsη), x -> _sinc_custom(x),
                Dense(hlsη, hlsη), x -> _sinc_custom(x),
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, 1)
    ) |> gpu

u_true(x) = sin.((π .* view(x, 1, :, :) .^ 2 + view(x, 2, :, :) .^ 2) ./ 2)
g(x) = u_true(x)

function loss_int(xr, _φη)
    t1 = sum(I(xr, _φη))
    log((t1 * t1) / sum(_φη .* _φη))
end

loss_int(xr) = loss_int(xr, φη(xr))

function loss_bndry(xb)
    k = uθ(xb) .- g(xb)'
    sum(( k .* k) ) / Nb
end

loss(xr, xb) = loss_int(xr) + α * Nb * loss_bndry(xb) # To update primal network
loss(xr) = loss_int(xr) # Needed to update adversarial network

optθ = Flux.Optimise.ADAGrad(τθ)
optη = Flux.Optimise.ADAGrad(-τη)

# optθ = Flux.Optimise.Descent(τθ*.1)
# optη = Flux.Optimise.Descent(-τη *.1) # minus coz of gradient ascent
psθ = Flux.params(uθ)
psη = Flux.params(φη)

function train_step()
    # Sample points
    xr = 2 .* rand(Float32, d, Nr) .- 1# Sampling in region

    # Sample along the boundary
    xb = 2 .* rand(Float32, d, Nb) .- 1
    for i in 1:Nb
        (j=rand(1:10)) <= d ? xb[j, i] = 1 : xb[j - 5, i] = -1
    end

    xr = gpu(xr)
    xb = gpu(xb)

    for i in 1:Ku
        # update weak solution network parameter
        gradsθ = Flux.Tracker.gradient(() -> loss(xr, xb), psθ)
        Flux.Tracker.update!(optθ, psθ, gradsθ)
    end

    for i in 1:Kφ
        # update adversarial network parameter
        gradsη = Flux.Tracker.gradient(() -> loss(xr), psη)
        Flux.Tracker.update!(optη, psη, gradsη)
    end
end

NUM_ITERS = 20000

function custom_training_loop(start = 1)
    start < 1 && return

    for i in start:NUM_ITERS
        train_step()
        if i % 100 == 0
            u1 = cpu(uθ)
            adversary1 = cpu(φη)

            @save "primal$(i).bson" u1
            @save "adversary$(i).bson" adversary1
            # @save "primal$(i).bson" uθ
            # @save "adversary$(i).bson" φη
        end
        if i % 100 == 0
            println("$(i) iterations done!")
        end
    end
    println("Training done!")
    @save "primal.bson" uθ
    @save "adversary.bson" φη
end

custom_training_loop()

# Plots
pyplot(leg=true)
plot_x = plot_y = range(-1, stop = 1, length = 20)
l1 = @layout [a{0.7w} b]
l2 = @layout [a{0.7w} b]

u_true_plot(x, y) = u_true(vcat(x, y))[1]
u_theta_plot(x, y) = uθ(vcat(x, y, zeros(Float32, d -2, 1))).data[1]
p_true = plot(plot_x, plot_y, u_true_plot, st = [:surface, :contourf], layout=l1)
p_theta = plot(plot_x, plot_y, u_theta_plot, st = [:surface, :contourf], layout=l2)
