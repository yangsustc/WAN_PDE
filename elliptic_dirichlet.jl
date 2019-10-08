using LinearAlgebra
using Flux
using Flux: softplus, elu
using Random
using Plots
using BSON: @save

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
ρ1(x) = (π .^ 2 .* (view(x, 1, :, :) .^ 2) .+ (view(x, 2, :, :) .^ 2)) ./ 4

# So that we don't recalculate the same terms
f(x, _a, _ρ0, _ρ1, _cos) = 4 .* _ρ1 .* _a .* sin.(_ρ0) .- 4 .* _ρ0 .* _cos .-
                            (π + 1) .* _a .* _cos .+ 2 .* _ρ1 .* _cos
f(x, _a, _ρ0, _ρ1) = f(x, _a, _ρ0, _ρ1, cos.(_ρ0))
f(x, _a) = f(x, _a, ρ0(x)', ρ1(x)')

# grad_uθ(x) = Tracker.gradient((x) -> sum(uθ(x)), x; nest=true)[1]
# grad_φη(x) = Tracker.gradient((x) -> sum(φη(x)), x; nest=true)[1]

grad_uθ(x) = Tracker.forward(x -> uθ(x), x)[2](1)[1]
grad_φη(x) = Tracker.forward(x -> φη(x), x)[2](1)[1]

# function grad_uθ(x)
    # _, back = Tracker.forward(x -> uθ(x), x)[2](1)[1]
    # back(1)[1]
# end

# function grad_φη(x)
#     _, back = Tracker.forward(x -> φη(x), x)
#     back(1)[1]
# end

# Tracker.gradient((x) -> sum(uθ(x)), x; nest=true)[1]
# grad_varphi(x) = Tracker.gradient((x) -> sum(φη(x)), x; nest=true)[1]

function I(x, _∇uθ, _a)
    t1 = sum(_∇uθ .* grad_φη(x), dims = 1) .* _a
    t2 = sum(_∇uθ .^ 2, dims = 1) .* φη(x) ./ 2
    t2 .- f(x, _a) .- t1
end

I(x) = I(x, grad_uθ(x), a0(x))

# Primal network - weak solution to PDE
uθ = Flux.Chain(Dense(d, hlsθ, tanh),
                Dense(hlsθ, hlsθ, tanh),
                Dense(hlsθ, hlsθ, softplus),
                Dense(hlsθ, hlsθ, tanh),
                Dense(hlsθ, hlsθ, softplus),
                Dense(hlsθ, hlsθ, tanh),
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

loss_int(xrs) =  log((sum(I(xrs)) ^ 2) / sum(φη(xrs) .^ 2))
loss_bndry(xbs) = sum(((uθ(xbs) .- g(xbs)') .^ 2) ./ Nb)
loss(xrs, xbs) = loss_int(xrs) + α * Nb * loss_bndry(xbs) # To update primal network
loss(xrs) = loss_int(xrs) # Needed to update adversarial network

optθ = Flux.Optimise.ADAGrad(τθ)
optη = Flux.Optimise.ADAGrad(-τη)

# optθ = Flux.Optimise.Descent(τθ)
# optη = Flux.Optimise.Descent(-τη) # minus coz of gradient ascent

psθ = params(uθ)
psη = params(φη)

function train_step()
    # Sample points
    xr = 2 .* rand(Float32, d, Nr) # Sampling in region

    # Sample along the boundary
    xb = 2 .* rand(Float32, d, Nb) .- 1
    for i in 1:Nb
        (j=rand(1:10)) <= d ? xb[j, i] = 1 : xb[j - 5, i] = -1
    end

    for i in 1:Ku
        # update weak solution network parameter
        gradsθ = Flux.Tracker.gradient(() -> loss(xr, xb), psθ)
        Flux.Tracker.update!(optθ, psθ, gradsθ)
    end

    # for i in 1:Kφ
        # update adversarial network parameter
        gradsη = Flux.Tracker.gradient(() -> loss(xr), psη)
        Flux.Tracker.update!(optη, psη, gradsη)
    # end
end

NUM_ITERS = 2000

function custom_training_loop()
    for i in 1:NUM_ITERS
        train_step()
        if i % 5 == 1
            println("$(i)th iteration done!")
        end
        if i % 25 == 1
            @save "primal$(i).bson" uθ
            @save "adversary$(i).bson" φη
        end
    end
    println("Training done!")
    @save "primal.bson" uθ
    @save "adversary.bson" φη
end

custom_training_loop()

# Plots
pyplot(leg=false, ticks=nothing)
plot_x = plot_y = range(-1, stop = 1, length = 20)
l1 = @layout [a{0.7w} b; c{0.2h}]
l2 = @layout [a{0.7w} b; c{0.2h}]

u_true_plot(x, y) = u_true(vcat(x, y))[1]
u_theta_plot(x, y) = uθ(vcat(x, y, zeros(Float32, d -2, 1))).data[1]
p_true = plot(plot_x, plot_y, u_true_plot, st = [:surface, :contourf], layout=l1)
p_theta = plot(plot_x, plot_y, u_theta_plot, st = [:surface, :contourf], layout=l2)
