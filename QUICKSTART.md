# Quick Start Guide

## One-Line Commands

### Train all models
```bash
julia --project=. main_training.jl
```

### Test saved models
```bash
julia --project=. run_saved_models.jl
```

### Generate plots
```bash
julia --project=. plot_results.jl
```

### Run everything
```bash
julia --project=. main_training.jl && julia --project=. run_saved_models.jl && julia --project=. plot_results.jl
```

## Expected Output Files

After training:
- `model_simple_node.jls`
- `model_encoder_decoder.jls`
- `model_cnn.jls`

After plotting:
- `model_comparison.png`
- `heatmaps.png`
- `scatter_comparison.png`

## Customization

Edit these variables in scripts to change behavior:
- `N` - Grid size (default: 3)
- `datasize` - Number of time steps (default: 30)
- `tspan` - Time range (default: 0.0 to 1.5)
- `num_runs` - Training examples (default: 5)
- `maxiters` - Training iterations (default: 50)
- `learning_rate` - Learning rate (default: 0.05)
