include("training_framework.jl")

function run_model_from_file(model_file, u0, tspan, tsteps)
    println("Loading model from: ", model_file)
    model, params, state = load_model(model_file)

    println("Running model...")
    prob = NeuralODE(model, tspan, Tsit5(); saveat=tsteps)
    pred = Array(prob(u0, params, state)[1])

    println("Model prediction complete")
    return pred
end

rng = Xoshiro(0)
N = 3
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2]; length=datasize)

test_u0 = collect(rand(rng, Float32, N, N))
test_data = generate_heat_equation_data(test_u0, tspan, tsteps, N)

println("=== Testing Saved Models ===\n")

model_files = ["model_simple_node.jls", "model_encoder_decoder.jls",]
model_names = ["Simple Neural ODE", "Encoder-NODE-Decoder", "CNN Model"]

predictions = []
for (file, name) in zip(model_files, model_names)
    if isfile(file)
        println("Testing: ", name)
        pred = run_model_from_file(file, test_u0, tspan, tsteps)
        rmse = calculate_rmse(pred, test_data)
        println(name, " RMSE: ", rmse)
        println()
        push!(predictions, pred)
    else
        println("Warning: ", file, " not found. Skipping.")
        push!(predictions, nothing)
    end
end

println("All saved models tested successfully!")
