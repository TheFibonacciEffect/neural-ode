using Distributed
addprocs()
include("training_framework.jl")
include("models.jl")

rng = Xoshiro(0)
N = 3
datasize = 30
tspan = (0.0f0, 1.5f0)
num_runs = 5

println("Generating training data...")
train_data, train_u0, tsteps = generate_multiple_runs(N, datasize, tspan, num_runs, rng)

println("Generating test data...")
test_u0 = collect(rand(rng, Float32, N, N))
test_data = generate_heat_equation_data(test_u0, tspan, tsteps, N)

using Distributed

@everywhere include("training_framework.jl")
@everywhere include("models.jl")

@sync begin
    Distributed.@spawn begin
        println("\n=== Training Simple Neural ODE ===")
        model1 = create_simple_neural_ode(N)
        params1, state1 = train_model(model1, train_u0, train_data, tspan, tsteps, rng; 
                                       maxiters=50, learning_rate=0.05)
        rmse1, pred1 = evaluate_model(model1, params1, state1, test_u0, test_data, tspan, tsteps)
        println("Simple Neural ODE Test RMSE: ", rmse1)
        save_model(model1, params1, state1, "models/model_simple_node.jls")
        println("Model saved to model_simple_node.jls")
    end

    Distributed.@spawn begin
        println("\n=== Training Encoder-NODE-Decoder ===")
        model2 = create_encoder_node_decoder(N)
        params2, state2 = train_model(model2, train_u0, train_data, tspan, tsteps, rng; 
                                       maxiters=50, learning_rate=0.05)
        rmse2, pred2 = evaluate_model(model2, params2, state2, test_u0, test_data, tspan, tsteps)
        println("Encoder-NODE-Decoder Test RMSE: ", rmse2)
        save_model(model2, params2, state2, "models/model_encoder_decoder.jls")
        println("Model saved to model_encoder_decoder.jls")
    end
end
