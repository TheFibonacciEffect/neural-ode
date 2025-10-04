using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
      OptimizationOptimisers, Random, Plots

rng = Xoshiro(0)

N = 3

u0 = collect(rand(Float32, N,N))

datasize = 30

tspan = (0.0f0, 1.5f0)

tsteps = range(tspan[1], tspan[2]; length = datasize)

function trueODEfunc(du, u, p, t)
    Nx, Ny = size(u)
	α, dx, dy = ones(3)
    d2x = [ (u[i+1,j] - 2u[i,j] + u[i-1,j]) / dx^2 for i in 2:Nx-1, j in 1:Ny ]
    d2y = [ (u[i,j+1] - 2u[i,j] + u[i,j-1]) / dy^2 for i in 1:Nx, j in 2:Ny-1 ]

    du .= 0
    du[2:Nx-1, :]      .+= α .* d2x
    du[:, 2:Ny-1]      .+= α .* d2y

    du[1,:]    .= du[2,:]
    du[end,:]  .= du[end-1,:]
    du[:,1]    .= du[:,2]
    du[:,end]  .= du[:,end-1]
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)

odesol = solve(prob_trueode, Tsit5(); saveat = tsteps)

scatter(odesol)

ode_data = Array(odesol)

heatmap(ode_data[:,:,end])

dudt2 = Chain(x-> reshape(x,N*N),Dense(N*N, 2*N*N, tanh), Dense(2*N*N, N*N), x-> reshape(x,N,N))

p, st = Lux.setup(rng, dudt2)

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat = tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss
end

function callback(state, l; doplot = false)
    println(l)
    if doplot
		px,py=2,2
        pred = predict_neuralode(state.u)
        plt = scatter(tsteps, ode_data[px,py, :]; label = "data")
        scatter!(plt, tsteps, pred[px,py, :]; label = "prediction")
        display(plot(plt))
    end
    return false
end

pinit = ComponentArray(p)

begin
	callback((; u = pinit), loss_neuralode(pinit); doplot = true)
	plot!()
end

adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.05); callback = callback, maxiters = 300)

optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(
    optprob2, Optim.BFGS(; initial_stepnorm = 0.01); callback, allow_f_increases = false)

result_neuralode2.u.layer_2.weight

begin
	callback((; u = result_neuralode2.u), loss_neuralode(result_neuralode2.u); doplot = true)
	plot!()
end
