include("training_framework.jl")
using Plots

rng = Xoshiro(0)
N = 3
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2]; length = datasize)

test_u0 = collect(rand(rng, Float32, N, N))
test_data = generate_heat_equation_data(test_u0, tspan, tsteps, N)

model_files = ["model_simple_node.jls", "model_encoder_decoder.jls", "model_cnn.jls"]
model_names = ["Simple Neural ODE", "Encoder-NODE-Decoder", "CNN Model"]

predictions = []
for (file, name) in zip(model_files, model_names)
    if isfile(file)
        model, params, state = load_model(file)
        prob = NeuralODE(model, tspan, Tsit5(); saveat = tsteps)
        pred = Array(prob(test_u0, params, state)[1])
        push!(predictions, pred)
    else
        println("Warning: ", file, " not found. Skipping.")
        push!(predictions, nothing)
    end
end

px, py = 2, 2
p = plot(layout=(2,2), size=(800, 800))

plot!(p[1], tsteps, test_data[px, py, :], label="Ground Truth", 
      linewidth=2, title="Element ($px, $py) Over Time")
for (i, (pred, name)) in enumerate(zip(predictions, model_names))
    if pred !== nothing
        plot!(p[1], tsteps, pred[px, py, :], label=name, linewidth=2)
    end
end

rmse = zeros(length(model_names))
for (i, (pred, name)) in enumerate(zip(predictions, model_names))
    if pred !== nothing
        rmse[i] = calculate_rmse(pred, test_data)
    end
end
bar!(p[2], model_names, rmse)

savefig("model_comparison.png")
println("Plot saved to model_comparison.png")

p2 = plot(layout=(3,1), size=(600, 800))
heatmap!(p2[1], test_data[:,:,end], title="Ground Truth (final state)", c=:viridis)
for (i, (pred, name)) in enumerate(zip(predictions, model_names))
    if pred !== nothing && i <= 2
        heatmap!(p2[i+1], pred[:,:,end], title="$name (final state)", c=:viridis)
    end
end

savefig("heatmaps.png")
println("Heatmap saved to heatmaps.png")

p3 = plot(layout=(1,1), size=(800, 600))
scatter!(p3, tsteps, test_data[px, py, :], label="Ground Truth", 
         markersize=5, title="Prediction Comparison at ($px, $py)")
for (pred, name) in zip(predictions, model_names)
    if pred !== nothing
        scatter!(p3, tsteps, pred[px, py, :], label=name, markersize=4)
    end
end

savefig("scatter_comparison.png")
println("Scatter plot saved to scatter_comparison.png")

println("\nAll plots generated successfully!")
