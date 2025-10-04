using Lux

function create_simple_neural_ode(N)
    model = Chain(
        x -> reshape(x, N*N),
        Dense(N*N, 2*N*N, tanh),
        Dense(2*N*N, N*N),
        x -> reshape(x, N, N)
    )
    return model
end

function create_encoder_node_decoder(N)
    encoder = Chain(
        x -> reshape(x, N*N),
        Dense(N*N, N*N÷2, tanh),
        Dense(N*N÷2, N*N÷4, tanh)
    )
    
    node_core = Chain(
        Dense(N*N÷4, N*N÷4, tanh)
    )
    
    decoder = Chain(
        Dense(N*N÷4, N*N÷2, tanh),
        Dense(N*N÷2, N*N),
        x -> reshape(x, N, N)
    )
    
    full_model = Chain(
        encoder,
        node_core,
        decoder
    )
    
    return full_model
end

function create_cnn_model(N)
    model = Chain(
        x -> reshape(x, N, N, 1, 1),
        Conv((3, 3), 1 => 8, pad=1, tanh),
        Conv((3, 3), 8 => 16, pad=1, tanh),
        Conv((3, 3), 16 => 8, pad=1, tanh),
        Conv((3, 3), 8 => 1, pad=1),
        x -> reshape(x, N, N)
    )
    return model
end
