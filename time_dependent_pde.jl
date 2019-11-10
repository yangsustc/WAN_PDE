using LinearAlgebra
using Flux
using Flux: softplus
using Random
using Plots
using BSON: @save
# using CuArrays
using Base.Iterators

t0 = 0                  # Start at t0 = 0
T = 1                   # Let's solve in time interval [0, T]
N = 10                  # Divide time into 10 segments
d = 5                   # Number of dimensions
Kφ = 1                  # Num iters for adversarial network
Ku = 2                  # Num iters for solutions
τη = 0.04               # Learning rate for adversary network
τθ = 0.015              # Learning rate for primal network
Nr = 4000 * d           # No. of sampled pts in region
Nb = 40 * d * d         # No. of sampled point on boundary
Na = Nb
α = 10000 * Nb          # Weight paramter on boundary
γ = α                   # Weight paramter at t=0
hlsθ = 20               # Hidden Layer size for primal network
hlsη = 50               # Hidden Layer size of adversarial network

pi = Float16(π)
pi_by_2 = Float32(π) / 2
pi_sq_by_2_minus_2 = (Float16(π * π) / 2) - 2

f(x, _g, _sin) = pi_sq_by_2_minus_2 .* _sin .* exp.(-1 .* view(x, d+1, :, :)) - _g .* _g
f(x, _sin) = f(x, g(x, h(x, _sin)), _sin)
f(x) = f(x, sin_term(x))

g(x, _h) = _h .* exp.(-1 .* view(x, d+1, :, :))
g(x) = g(x, h(x))

sin_term(x) = sin.(pi_by_2 .* view(x, 1, :, :))
h(x, _sin) = 2 .* _sin
h(x) = h(x, sin_term(x))

grad_uθ(x) = Tracker.forward(x -> uθ(x), x)[2](1)[1]
grad_φη(x) = Tracker.forward(x -> φη(x), x)[2](1)[1]

# Since view can't be done on tracked arrays, instead we multiply the unneeded by 0.
select_grad_x = [ones(d,1);0] |> gpu
select_grad_t = [zeros(d,1);1] |> gpu

function I(xr, xrt0, xrT, _∇uθ, _∇φη, _φη, _uθ)
    t1 = sum(uθ(xrT) .* φη(xrT) .- uθ(xrt0) .* φη(xrt0))
    t2 = -sum(_uθ .* _∇φη .* select_grad_t) / N
    t3_1 = sum(_∇φη .*_∇uθ .* select_grad_x, dims = 1)
    t3_2 = ((_uθ .^ 2) .+ f(xr)') .* _φη
    t3 = sum(t3_1 .- t3_2) * (T - t0) / N

    return t1 + t2 + t3
end

# I(x, _φη) = I(x, grad_uθ(x), _φη)

# Primal network - weak solution to PDE
uθ = Flux.Chain(Dense(d + 1, hlsθ), x -> tanh.(x),
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

φη = Flux.Chain(Dense(d + 1, hlsη), x -> tanh.(x),
                Dense(hlsη, hlsη), x -> tanh.(x),
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, hlsη), x -> _sinc_custom(x),
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, hlsη), x -> _sinc_custom(x),
                Dense(hlsη, hlsη), x -> _sinc_custom(x),
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, 1)
    ) |> gpu

u_true(xs_and_t) = g(xs_and_t)

function loss_int(xr, xrt0, xrT, _∇uθ, _∇φη, _φη, _uθ)
    t1 = I(xr, xrt0, xrT, _∇uθ, _∇φη, _φη, _uθ)
    log((t1 * t1) / sum(_φη .* _φη))
end

loss_int(xr, xrt0, xrT) = loss_int(xr, xrt0, xrT, grad_uθ(xr), grad_φη(xr), φη(xr), uθ(xr))

function loss_bndry(xb)
    k = uθ(xb) .- g(xb)'
    sum((k .* k)) / Nb
end

function loss_init(xa)
    k = uθ(xa) .- h(xa)'
    sum((k .* k)) / Na
end

loss_primal(xr, xrt0, xrT, xb, xa) = loss_int(xr, xrt0, xrT) + γ * loss_init(xa) + α * loss_bndry(xb)  # To update primal network
loss_adversarial(xr, xrt0, xrT) = loss_int(xr, xrt0, xrT) # Needed to update adversarial network

optθ = Flux.Optimise.ADAGrad(τθ)
optη = Flux.Optimise.ADAGrad(-τη)# minus coz of gradient ascent

psθ = Flux.params(uθ)
psη = Flux.params(φη)

times = range(Float32(t0), length=N, stop=Float32(T)) # Needed for numerical integration over T
times_tr = collect(flatten(repeat(collect(times)', Nr)))'

tr_t0 = fill(Float32(t0), 1, Nr) |> gpu
tr_T = fill(Float32(T), 1, Nr) |> gpu

function train_step()
    # Sample points
    xr = 2 .* rand(Float32, d, Nr) .- 1 # Sampling in region
    xr = gpu(xr)
    xrt0 = vcat(xr, tr_t0)
    xrT = vcat(xr, tr_T)
    xr = hcat(collect(repeated(xr, N))...)
    xr = vcat(xr, times_tr)

    # Sample along the boundary
    xb = vcat(2 .* rand(Float32, d, Nb) .- 1, rand(Float32, 1, Nb))
    for i in 1:Nb
        (j=rand(1:2*d)) <= d ? xb[j, i] = 1 : xb[j - 5, i] = -1
    end

    xa = 2 .* rand(Float32, d + 1, Na) .- 1
    xa = xa .* [ones(d);0]

    # Sample in the region in t=0
    # xr = gpu(xr)
    # xrt0 = gpu(xr0)
    # xrT = gpu(xrT)
    xb = gpu(xb)
    xa = gpu(xa)

    for i in 1:Ku
        # update weak solution network parameter
        gradsθ = Flux.Tracker.gradient(() -> loss_primal(xr, xrt0, xrT, xb, xa), psθ)
        Flux.Tracker.update!(optθ, psθ, gradsθ)
    end

    for i in 1:Kφ
        # update adversarial network parameter
        gradsη = Flux.Tracker.gradient(() -> loss_adversarial(xr, xrt0, xrT), psη)
        Flux.Tracker.update!(optη, psη, gradsη)
    end
end

NUM_ITERS = 20000

function custom_training_loop(start = 1)
    start < 1 && return

    for i in start:NUM_ITERS
        train_step()
        if i % 100 == 1
            u1 = cpu(uθ)
            adversary1 = cpu(φη)

            @save "primal$(i).bson" u1
            @save "adversary$(i).bson" adversary1

        end
        if i % 100 == 0
            println("$(i) iterations done!")
        end
    end

    u1 = cpu(uθ)
    adversary1 = cpu(φη)

    @save "primal.bson" u1
    @save "adversary.bson" adversary1

    println("Training done!")
end

custom_training_loop()

# Plots
pyplot(leg=true)
plot_x = range(-1, stop = 1, length = 20)
plot_t = range(0, stop = 1, length = 20)
l1 = @layout [a{0.7w} b]
l2 = @layout [a{0.7w} b]
l3 = @layout [a{0.7w} b]

u_true_plot(t, x) = u_true(vcat(x, zeros(Float32, d - 1, 1), t))[1]
u_theta_plot(t, x) = uθ(vcat(x, zeros(Float32, d -1, 1), t)).data[1]
u_diff(t, x) = abs(u_true_plot(t, x) - u_theta_plot(t, x))
p_true = plot(plot_t, plot_x, u_true_plot, st = [:surface, :contourf], layout=l1)
p_theta = plot(plot_t, plot_x, u_theta_plot, st = [:surface, :contourf], layout=l2)
p_diff = plot(plot_t, plot_x, u_diff, st = [:surface, :contourf], layout=l3)
