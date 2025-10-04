using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
      OptimizationOptimisers, Random, Plots, Statistics
using Serialization

function generate_heat_equation_data(u0, tspan, tsteps, N)
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
    
    prob = ODEProblem(trueODEfunc, u0, tspan)
    sol = solve(prob, Tsit5(); saveat = tsteps)
    return Array(sol)
end

function generate_multiple_runs(N, datasize, tspan, num_runs, rng)
    tsteps = range(tspan[1], tspan[2]; length = datasize)
    all_data = []
    all_u0 = []
    
    for i in 1:num_runs
        u0 = collect(rand(rng, Float32, N, N))
        data = generate_heat_equation_data(u0, tspan, tsteps, N)
        push!(all_data, data)
        push!(all_u0, u0)
    end
    
    return all_data, all_u0, tsteps
end

function calculate_rmse(predictions, targets)
    return sqrt(mean((predictions .- targets).^2))
end

function save_model(model, params, state, filepath)
    model_data = Dict(
        "model" => model,
        "params" => params,
        "state" => state
    )
    serialize(filepath, model_data)
end

function load_model(filepath)
    model_data = deserialize(filepath)
    return model_data["model"], model_data["params"], model_data["state"]
end

function train_model(model, u0_list, data_list, tspan, tsteps, rng; maxiters=100, learning_rate=0.05)
    p, st = Lux.setup(rng, model)
    pinit = ComponentArray(p)
    
    function predict_model(params, u0)
        prob = NeuralODE(model, tspan, Tsit5(); saveat = tsteps)
        return Array(prob(u0, params, st)[1])
    end
    
    function loss_function(params)
        total_loss = 0.0
        for (u0, data) in zip(u0_list, data_list)
            pred = predict_model(params, u0)
            total_loss += sum(abs2, data .- pred)
        end
        return total_loss / length(u0_list)
    end
    
    function callback(state, l)
        println("Loss: ", l)
        return false
    end
    
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_function(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)
    
    result = Optimization.solve(optprob, OptimizationOptimisers.Adam(learning_rate); 
                                callback = callback, maxiters = maxiters)
    
    optprob2 = remake(optprob; u0 = result.u)
    result2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01); 
                                 callback, allow_f_increases = false)
    
    return result2.u, st
end

function evaluate_model(model, params, state, u0_test, data_test, tspan, tsteps)
    prob = NeuralODE(model, tspan, Tsit5(); saveat = tsteps)
    pred = Array(prob(u0_test, params, state)[1])
    rmse = calculate_rmse(pred, data_test)
    return rmse, pred
end
