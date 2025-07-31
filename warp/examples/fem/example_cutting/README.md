# Interactive cutting/sculpting example

This example demonstrates an interactive 3D cutting and sculpting simulation using NVIDIA Warp's FEM submodule. The simulation allows real-time manipulation, cutting and sculpting of deformable objects.

Optionally, this example may use:
- **Neural quadrature integration** for enhanced numerical accuracy, as described in [Neurally Integrated Finite Elements for Differentiable Elasticity on Evolving Domains](https://research.nvidia.com/labs/toronto-ai/flexisim/) (ACM Transactions on Graphics, 2025)
- **FlexiCubes** for surface extraction from the sculpted SDF, as described in [Flexible Isosurface Extraction for Gradient-Based Mesh Optimization](https://research.nvidia.com/labs/toronto-ai/flexicubes/) (SIGGRAPH 2023)

## Requirements

The easiest way to run these scripts is to install [uv](https://docs.astral.sh/uv/getting-started/installation/), which will automatically handle all Python dependencies.

**Install uv:**
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

**Alternative manual installation:**
If you prefer to manage dependencies manually, you'll need:
- Python ≥ 3.10
- warp-lang≥1.8.1, recommended≥1.9.0dev20250801 (e.g. the version from this branch)
- polyscope==2.1.*
- [kaolin](https://github.com/NVIDIAGameWorks/kaolin)≥0.17.0
- torch and torchvision (depending on kaolin version)
- Additional packages listed in each script's header

**Note:** This example depends on the NVIDIA Kaolin FlexiCubes implementation, which is part of the namespace `kaolin.non_commercial` released under the [NSCL license](https://github.com/NVIDIAGameWorks/kaolin/blob/master/LICENSE.NSCL).

## Usage

### Main Interactive Cutting Script

```bash
uv run example_cutting.py /path/to/mesh.obj [OPTIONS]
```

A classical test subject is the Armadillo mesh from the [The Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/)
```bash
uv run --script example_cutting.py -nh Armadillo.ply --y_min -0.5 --y_max 1.5 
```


**Basic Usage:**
```bash
# Run with default settings
uv run example_cutting.py /path/to/mesh.obj

# Use a neural quadrature model (see below for training a model)
uv run example_cutting.py /path/to/mesh.obj -qm model.pt

# Use higher resolution grid
uv run example_cutting.py /path/to/mesh.obj --resolution 128

# Enable adaptive refinement
uv run example_cutting.py /path/to/mesh.obj --levels 2
```

**Command-line Options:**

**Required:**
- `mesh` - Path to the input mesh file (.obj format)

**Optional:**
- `--quadrature_model`, `-qm` - Path to saved neural quadrature MLP weights. If not provided, uses regular quadrature
- `--variant`, `-v` - Simulation variant: `classic` (default), `mfem`, or `trusty22`
- `--resolution` - Grid resolution at finest level (default: 64)
- `--levels` - Number of adaptive refinement levels; 0 means no adaptivity (default: 0)
- `--force_scale` - Scaling factor for dynamic picking forces (default: 1.0)
- `--y_min` - Clamp points below this Y value (default: -0.9)
- `--y_max` - Clamp points above this Y value (default: 0.9)

Additional simulation-specific options are available depending on the chosen variant. Run with `--help` to see the complete list.

**Interactive Commands:**

- **Ctrl+left mouse:** add material
- **Ctrl+right mouse:** remove material
- **Shift+left drag:** apply picking force

## Neural Quadrature Models (MLP)

The `mlp/` folder contains scripts for training, testing, and visualizing neural quadrature models that can be used with the main cutting simulation. All scripts use `uv` for dependency management with dependencies listed at the top of each script.

### Training a Quadrature Model

Train a neural network to learn adaptive quadrature rules:

```bash
cd mlp/
uv run train.py [OPTIONS]
```

**Training Options:**
- `-d, --dim` - Dimension of the space (default: 3)
- `-o, --order` - Order of the quadrature (default: 2)
- `-i, --iters` - Number of training iterations (default: 64000)
- `--n_train` - Number of training samples (default: 24)
- `--n_test` - Number of test samples (default: 10)
- `--n_batches` - Number of batches (default: 16)
- `--conditioning_pen` - Conditioning penalty (default: 0.00001)
- `--outside_coords_pen` - Outside coordinates penalty (default: 10.0)
- `--gt_res` - Resolution for ground truth integration (default: 32)
- `--log_interval` - Iterations between logging (default: 100)
- `--write_interval` - Iterations between model saves (default: 1000)
- `--device` - Device to use: `cuda` or `cpu` (default: cuda)

**Example:**
```bash
# Train a 3D quadrature model with higher order
uv run train.py -d 3 -o 4 -i 100000

# Train with custom penalties
uv run train.py --conditioning_pen 0.0001 --outside_coords_pen 5.0
```

### Visualizing Quadrature Results

Visualize the learned quadrature points and weights:

```bash
cd mlp/
uv run quadrature_viz.py model.pt
```

This opens an interactive 3D visualization showing:
- Quadrature points and their positions
- Weight distributions
- FlexiCubes mesh reconstruction

### Testing Quadrature Models

Numerically evaluate and compare different quadrature models:

```bash
cd mlp/
uv run test.py model1.pt [model2.pt ...] [OPTIONS]
```

**Testing Options:**
- `quadrature_models` - Path(s) to quadrature models, or predefined formulas: `clip`, `full`, `muller`
- `-d, --dim` - Dimension of the space (default: 3)
- `-o, --orders` - Order(s) of quadrature to test (default: 2)
- `-c, --cond` - Conditioning penalty values for plot labels
- `-l, --labels` - Custom labels for models in plots
- `--n_test` - Number of test samples (default: 10)
- `--gt_res` - Resolution for ground truth integration (default: 32)
- `--device` - Device to use (default: cuda)

**Examples:**
```bash
# Compare multiple models
uv run test.py model1.pt model2.pt -l "Model 1" "Model 2"

# Test against predefined methods
uv run test.py trained_model.pt clip full

# Test different orders
uv run test.py model.pt -o 2 4 6
```

The test script generates plots comparing integration accuracy and numerical conditioning across different methods.

 
 