# Check if starting pt could be changed.

using LinearAlgebra
using Flux
using Flux: softplus
using Random
using Plots
using BSON: @save
using CuArrays
using Base.Iterators
# using DifferentialEquations
using DiffEqFlux
using CUDAnative: sinc


t0 = 0                  # Start at t0 = 0
T = 1.0f0               # Let's solve in time interval [0, T]
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

f(x, _g, _sin) = pi_sq_by_2_minus_2 .* _sin ./ exp.(view(x, d+1, :, :)) .- _g .* _g
f(x, _sin) = f(x, g(x, _sin), _sin)
f(x) = f(x, sin_term(x))

g(x, _sin) = 2 .* _sin ./ exp.(view(x, d+1, :, :))
g(x) = 2 .* sin_term(x) ./ exp.(view(x, d+1, :, :))

sin_term(x) = sin.(pi_by_2 .* view(x, 1, :, :))
h(x, _sin) = 2 .* _sin
h(x) = 2 .* sin_term(x)

grad_uθ(x) = Tracker.forward(x -> uθ(x), x)[2](1)[1]
grad_φη(x) = Tracker.forward(x -> φη(x), x)[2](1)[1]
grad_(x, f) = Tracker.forward(x -> f(x), x)[2](1)[1]

# Since view can't be done on tracked arrays, instead we multiply the unneeded by 0 and add.
select_grad_x = [ones(Float32, d,1);0] |> gpu
select_grad_t = [zeros(Float32, d,1);1] |> gpu

tspan = (0.0, T)
tspan = Float32.(tspan)

function dudt(xt, p)
    (uθ_, φη_) = DiffEqFlux.restructure(Chain(uθ, φη), p)
    # xt = vcat(xr, CuArrays.fill(t, (1, Nr)))
    # global _∇uθ, _∇φη, _φη, _uθ
    _∇uθ_ = grad_(xt, uθ_)
    _∇φη_ = grad_(xt, φη_)
    _φη_ = φη_(xt)
    _uθ_ = uθ_(xt)
    _f = f(xt)'
    t2 = -sum(_uθ_ .* _∇φη_ .* select_grad_t )#./ 16)
    t3 = (sum(_∇φη_ .*_∇uθ_ .* select_grad_x )#./ 16)
             - sum(((_uθ_ .^ 2) .+ _f) .* _φη_ )#./ 16)
         ) * (T - t0)
    return t2 + t3
end

# function n_ode
function n_ode(x, xr)

    p = DiffEqFlux.destructure(Chain(uθ, φη))

    # Note: the following custom neural_ode uses u as `Scalar` than `Vector`
    dudt_(u::Tracker.TrackedReal, p, t) = dudt(vcat(xr, CuArrays.fill(t, (1,Nr))), p)# =# fill(t, (1,Nr))))
    dudt_(u::Real, p, t) = Flux.data(dudt(vcat(xr, CuArrays.fill(t, (1,Nr))), p)) # =# fill(t, (1,Nr)))))
    prob = ODEProblem(dudt_, x, tspan, p)
    return diffeq_adjoint(p, prob, save_start=false, save_everystep=false, u0=x)
end

# dudt(u, p, t, xt) = dudt(u, p, t, xt, grad_uθ(xt), grad_φη(xt), φη(xt), uθ(xt))
# dudt(u, p, t) = dudt(u, p, t, vcat(xr, CuArrays.fill(t, (1, Nr))))

# n_ode(x) = neural_ode(dudt_, x, tspan, Tsit5(), save_everystep=false, save_start=false)

function I(xr, xrt0, xrT)
    t1 = sum(uθ(xrT) .* φη(xrT) .- uθ(xrt0) .* φη(xrt0)) #/ 16
    return n_ode(t1, xr) #* 16
end

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
ϵ = Float32(1e-4)

# _sinc_custom(x) = @. sin(x) / (x + ϵ) # Speed on GPUs during backprop

φη = Flux.Chain(Dense(d + 1, hlsη), x -> tanh.(x),
                Dense(hlsη, hlsη), x -> tanh.(x),
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, hlsη), x -> sinc.(x),
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, hlsη), x -> sinc.(x),
                Dense(hlsη, hlsη), x -> sinc.(x),
                Dense(hlsη, hlsη, softplus),
                Dense(hlsη, 1)
    ) |> gpu

m = Chain(uθ, φη)

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
optη = Flux.Optimise.ADAGrad(-τη) # minus coz of gradient ascent

psθ = Flux.params(uθ)
psη = Flux.params(φη)

times = range(Float32(t0), length=N, stop=Float32(T)) # Needed for numerical integration over T
# times_tr = collect(flatten(repeat(collect(times)', Nr)))'

tr_t0 = fill(Float32(t0), 1, Nr) |> gpu
tr_T = fill(Float32(T), 1, Nr) |> gpu
xr = gpu(2 .* rand(Float32, d, Nr) .- 1)# Sampling in region
xrt0 = vcat(xr, tr_t0)
xrT = vcat(xr, tr_T)
_∇uθ = grad_uθ(xrt0)
_∇φη = grad_φη(xrt0)
_φη = φη(xrt0)
_uθ = uθ(xrt0)

function train_step()
    # Sample points
    xr = gpu(2 .* rand(Float32, d, Nr) .- 1) # Sampling in region
    xrt0 = vcat(xr, tr_t0)
    xrT = vcat(xr, tr_T) 
    # Sample along the boundary
    xb = vcat(2 .* rand(Float32, d, Nb) .- 1, rand(Float32, 1, Nb))
    for i in 1:Nb
        (j=rand(1:2*d)) <= d ? xb[j, i] = 1 : xb[j - 5, i] = -1
        if (rand() > 0.99)
            (j=rand(1:2*d)) <= d ? xb[j, i] = 1 : xb[j - 5, i] = -1
        end
    end
    xb = gpu(xb)

    # Sample in the region in t=0
    xa = vcat(2 .* CuArrays.rand(Float32, d, Na) .- 1, CuArrays.zeros(Float32, 1, Na))

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
