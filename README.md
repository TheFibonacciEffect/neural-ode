# Heat Equation Neural ODE Project

This project implements and compares three different neural network approaches for solving the 2D heat equation using Neural ODEs.

## Project Structure

- `heat_neural_ode.jl` - Original Pluto notebook code converted to standard Julia
- `training_framework.jl` - Core functions for data generation, training, evaluation, and model persistence
- `models.jl` - Three different model architectures
- `main_training.jl` - Main script to train all three models
- `run_saved_models.jl` - Script to load and test saved models
- `plot_results.jl` - Script to generate comparison plots
- `Project.toml` - Julia project dependencies
- `Manifest.toml` - Locked dependency versions

## Models Implemented

### 1. Simple Neural ODE
A basic neural network that learns the heat equation dynamics:
- Flattens input → Dense layers → Reshape to 2D
- Architecture: N×N → (N²) → (2N²) → (N²) → N×N

### 2. Encoder-NODE-Decoder
A more complex architecture with dimensionality reduction:
- **Encoder**: Compresses the state to a latent representation
- **Neural ODE Core**: Learns dynamics in latent space
- **Decoder**: Reconstructs the full state
- Architecture: N×N → (N²/2) → (N²/4) → (N²/4) → (N²/2) → (N²) → N×N

### 3. Convolutional Neural Network (CNN)
Uses convolutional layers to capture spatial patterns:
- 4 convolutional layers with increasing then decreasing channels
- Preserves spatial structure throughout
- Architecture: 1 channel → 8 → 16 → 8 → 1 channel

## Usage

### 1. Setup Environment
```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

### 2. Train All Models
```bash
julia --project=. main_training.jl
```

This will:
- Generate training data (5 runs of the heat equation)
- Train all three models
- Evaluate on test data
- Save models to `.jls` files:
  - `model_simple_node.jls`
  - `model_encoder_decoder.jls`
  - `model_cnn.jls`

### 3. Test Saved Models
```bash
julia --project=. run_saved_models.jl
```

This loads each saved model and evaluates its performance on test data.

### 4. Generate Plots
```bash
julia --project=. plot_results.jl
```

This creates three visualization files:
- `model_comparison.png` - Comparison of predictions vs ground truth
- `heatmaps.png` - Heatmaps of final states
- `scatter_comparison.png` - Scatter plot comparison

## Key Functions

### Data Generation
- `generate_heat_equation_data(u0, tspan, tsteps, N)` - Solve heat equation for given initial conditions
- `generate_multiple_runs(N, datasize, tspan, num_runs, rng)` - Generate multiple training examples

### Model Training
- `train_model(model, u0_list, data_list, tspan, tsteps, rng)` - Train a model on multiple trajectories
- Uses Adam optimizer followed by BFGS for refinement

### Evaluation
- `calculate_rmse(predictions, targets)` - Calculate root mean squared error
- `evaluate_model(model, params, state, u0_test, data_test, tspan, tsteps)` - Full evaluation

### Model Persistence
- `save_model(model, params, state, filepath)` - Save model to disk
- `load_model(filepath)` - Load model from disk

## Heat Equation Details

The 2D heat equation with Neumann boundary conditions:
- ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
- Boundary conditions: zero flux (∂u/∂n = 0 at boundaries)
- Grid size: 3×3 (configurable)
- Time span: 0 to 1.5
- Thermal diffusivity: α = 1

## Requirements

All dependencies are specified in `Project.toml`:
- Lux - Modern neural network framework
- DiffEqFlux - Neural ODEs
- OrdinaryDiffEq - ODE solvers
- Optimization - Training optimization
- Plots - Visualization
- ComponentArrays - Parameter handling
- Serialization - Model saving/loading
