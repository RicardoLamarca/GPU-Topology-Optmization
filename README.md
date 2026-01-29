# GPU-Accelerated 3D Topology Optimization in Julia

![Julia](https://img.shields.io/badge/Julia-1.7%2B-9558B2?style=for-the-badge&logo=julia)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A fully **GPU-resident** 3D Topology Optimization solver written in Julia. This script utilizes `CUDA.jl` to perform Finite Element Analysis (FEA) and design updates entirely on the GPU (VRAM), minimizing CPU-GPU memory transfers.

It solves the standard **minimum compliance** problem (maximizing stiffness) using the **SIMP** (Solid Isotropic Material with Penalization) method.

---

## ⚙️ Implemented Features

Based on the codebase, this solver includes:

### 1. GPU-Resident Architecture
The optimization loop runs entirely on the GPU to avoid PCie bottlenecks:
* **Stiffness Assembly:** Parallel computation of element stiffness matrices using `gpu_compute_Ke_kernel!`.
* **Linear Solver:** Custom Conjugate Gradient (CG) solver (`gpu_cg_solve!`) optimized for sparse matrix-vector multiplication on the GPU.
* **Density Filtering:** 3D spatial convolution (`gpu_density_filter_kernel!`) to ensure mesh-independency.

### 2. Physics & Boundary Conditions
The script is hardcoded for the **Half-MBB Beam** problem:
* **Domain:** 3D rectangular grid ($N_x \times N_y \times N_z$).
* **Boundary Conditions:**
    * **Left Face:** Fixed (Cantilever support).
    * **Right Face:** Symmetry boundary condition (Roller).
    * **Load:** Point load applied vertically at the top-center of the domain.
* **Optimization Method:** Optimality Criteria (OC).

### 3. Output Capabilities
* **Interactive Visualization:** Real-time 3D voxel plotting using `GLMakie`.
* **Manufacturing Ready:** Automatic export to binary **.STL** format for 3D printing (`export_stl`).

---

## 💻 Hardware Requirements

To run this script, you must have:
1.  **NVIDIA GPU:** The code explicitly requires CUDA support.
2.  **Drivers:** Up-to-date NVIDIA drivers and a compatible CUDA Toolkit.

---

## 🛠️ Installation

1.  **Install Julia:** Download from [julialang.org](https://julialang.org/).
2.  **Install Dependencies:**
    Open the Julia REPL and run the following command to install required packages:
    ```julia
    using Pkg
    Pkg.add(["CUDA", "LinearAlgebra", "SparseArrays", "Printf", "Makie", "GLMakie", "GeometryBasics", "FileIO"])
    ```

---

## 🚀 Usage

The solver is self-contained in a single file. You can adjust the parameters in the `gpu_topopt_3d` function call at the bottom of the script.

### Running the Optimization
```julia
include("your_script_name.jl")

# Run with default parameters (Half-MBB Beam)
result = gpu_topopt_3d(
    nx = 60, ny = 20, nz = 10,   # Grid resolution
    volfrac = 0.5,               # Target volume fraction (50%)
    rmin = 2.0,                  # Filter radius
    penal = 3.0,                 # Penalization factor (SIMP)
    maxiter = 200,               # Max iterations
    verbose = true
)

# 1. Visualize in 3D (GLMakie)
# 'threshold' determines the density cutoff for showing voxels (0.0 - 1.0)
fig = visualize_result(result; threshold=0.3)
display(fig)

# 2. Export to STL
# Creates a binary STL file for 3D printing
export_stl(result, "output_mesh.stl"; threshold=0.3)

