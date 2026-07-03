# piezoMNN.py — Piezoelectric Mechanical Neural Network (MNN) Simulator

A JAX-accelerated finite element (FEM) framework for simulating and optimizing a **Mechanical Neural Network**: a lattice of beams with bonded piezoelectric patches. Applying voltages to the patches induces bending/axial actuation forces; the framework uses gradient-based optimization (via JAX autodiff + SciPy's SLSQP) to find the voltage patterns that drive selected output nodes to target displacements, including multi-objective (Pareto) trade-offs between competing load cases.

## What it does

1. **Builds a parametric truss/lattice geometry** (a "brick-wall" pattern of triangular cells) and discretizes each beam into finite elements.
2. **Assembles a 2D Euler-Bernoulli beam FEM model** (axial + bending + shear-free), embeds piezoelectric top/bottom patches on every beam, and reduces the stiffness matrix using fixed boundary conditions.
3. **Solves the static equilibrium** `K·u = F` for any combination of mechanical loads and patch voltages, using JAX for differentiable, JIT-compiled linear algebra.
4. **Optimizes patch voltages** so that specific output-node displacements match user-defined targets, either for a single load case, two independent load cases (with a Pareto sweep), or a continuous sinusoidal displacement profile along one edge of the lattice.
5. **Visualizes** the deformed structure, color-coding each piezo patch by actuation polarity (red = positive voltage, blue = negative), and plots Pareto trade-off curves.

## Requirements

```
python >= 3.9
numpy
jax
scipy
matplotlib
```

Install with:

```bash
pip install numpy jax scipy matplotlib
```

> JAX is configured for 64-bit precision (`jax_enable_x64`) at import time for numerically accurate FEM results. A CPU-only JAX install is sufficient; no GPU is required for the default lattice sizes.

## Running the script

```bash
python piezoMNN.py
```

All behavior is controlled by the `SimulationConfig` class at the top of the file — there are no command-line arguments. Edit the class attributes, then run the script; plots are shown via `matplotlib.pyplot.show()`.

## Configuration reference (`SimulationConfig`)

| Section | Parameter | Meaning |
|---|---|---|
| **Geometry & Mesh** | `Nx` | Number of base segments across the width (node pairs per layer) |
| | `Ny` | Number of triangular layers above/below the center row |
| | `l_beam` | Nominal beam length [mm] |
| | `n_elem` | Number of FE subdivisions per beam (mesh refinement) |
| **Core beam material** | `E_beam` | Young's modulus of the structural core [MPa] |
| | `t_beam` | Core thickness [mm] |
| | `width` | Out-of-plane beam width [mm] |
| | `sigma_yield` | Yield stress threshold [MPa] (reference value; not enforced as a constraint in the current optimization loops) |
| **Piezo patches** | `E_pzt` | Young's modulus of the PZT material [MPa] |
| | `t_pzt` | Piezo layer thickness [mm] |
| | `d31` | Piezoelectric strain coupling constant [mm/V] |
| | `sigma_max_pzt` | Allowable fracture stress limit [MPa] (reference value) |
| **Input loads** | `F_case1` | Magnitude of the horizontal (X) mechanical load [N] |
| | `F_case2` | Magnitude of the vertical (Y) mechanical load [N] |
| **Targets** | `single_target_1/2` | `[ux, uy]` targets for the two output nodes, single-case optimization [mm] |
| | `multi_t1_case1/2`, `multi_t2_case1/2` | `[ux, uy]` targets for two output nodes under two independent load cases [mm] |
| | `sinusoid_amplitude` | Amplitude of the target sinusoidal displacement profile [mm] |
| **Execution flags** | `run_random_voltage_test` | Apply random voltages and plot the resulting deformation (sanity check) |
| | `run_single_case_opt` | Run the single-load-case voltage optimization |
| | `run_multi_case_opt` | Run the two-load-case Pareto optimization |
| | `run_pareto_sinusoid_opt` | Run the sinusoidal-target Pareto optimization along the right edge |

Only one (or more) of the four `run_*` flags need be `True` at a time; each block is independent and skipped when its flag is `False`.

## Script structure

### 1. Geometry & mesh (`generate_parametric_truss`, `fea_setup`)
- `generate_parametric_truss` builds the coarse lattice topology: node coordinates, beam connectivity, fixed anchor nodes (top/bottom rows), and the designated input/output node pairs.
- `fea_setup` subdivides each coarse beam into `n_elem` finite elements, deduplicates shared nodes, and renumbers nodes by row (center → outward) then by X position for a consistent DOF ordering.

### 2. Boundary conditions (`boundary_conditions`)
Locks the `(u, v, θ)` degrees of freedom at the anchor nodes and returns the list of free (unconstrained) DOFs used for static condensation.

### 3. FEM core
- `beam_2d_element` — standard 6×6 Euler-Bernoulli 2D beam stiffness matrix, rotated into the global frame.
- `global_reduced_K` — assembles the global stiffness matrix element-by-element and condenses it to the free DOFs.
- `piezo_element_load` — converts top/bottom patch voltages into an equivalent nodal force/moment vector for one beam element.
- `reduced_global_f` — vectorized (via `jax.vmap`) assembly of the global load vector combining mechanical forces and all piezo actuation loads, reduced to free DOFs.
- `solve_fem` — solves `K_red · u_red = f_red` for the free-DOF displacements and scatters them back into the full DOF vector.

### 4. Optimization objectives
- `obj_single_case` — mean-squared error between two output nodes' displacements and their single-case targets.
- `obj_multi_case` — returns a pair of MSE errors `(e1, e2)`, one per independent load case, so they can be scalarized with a sweep of trade-off weights.
- `sinusoid_objective_split` — same idea, but targets are a full displacement profile (sinusoid vs. its negation) evaluated across every node on the lattice's right edge, under two opposing load cases.

All objectives are pure JAX functions, so `jax.value_and_grad` provides exact gradients to SciPy's `minimize(..., method='SLSQP', jac=True)`.

### 5. Visualization (`plot_deformed_structure`)
Plots the undeformed mesh in gray and the deformed mesh in black/colored lines; each beam's top and bottom piezo patch is drawn as an offset line colored red (positive V), blue (negative V), or dark gray (zero V). Optional `x` markers show target displacement locations for comparison against the achieved deformation.

### 6. Main execution block
Runs top-to-bottom when the script is executed directly:
1. Computes homogenized equivalent section properties (`A_eq`, `E_eq`, `I_eq`) for the beam+piezo composite cross-section.
2. Builds geometry, mesh, and the reduced stiffness matrix once (shared across all experiments).
3. Defines the two base mechanical load vectors (`f_mech_c1` horizontal, `f_mech_c2` vertical).
4. Executes whichever of the four experiments are enabled in `SimulationConfig`:
   - **Random Voltage Test** — quick sanity check with random actuation.
   - **Single Load Case Optimization** — finds one voltage set matching two node targets under one load case.
   - **Multi Load Case Optimization** — sweeps a scalarization weight `w ∈ [0, 1]` between two load cases' objectives, producing a Pareto front and plotting the two extreme (`w=0`, `w=1`) deformed shapes.
   - **Pareto Sinusoid Morphing** — same Pareto sweep, but the targets are a sinusoidal displacement wave imposed along the lattice's right-edge nodes, with the two load cases driving opposite-signed waves.

## Output

- Console output: optimizer convergence status and per-weight error values (`Case 1 Error`, `Case 2 Error`).
- Matplotlib figures: deformed-structure plots (per experiment) and Pareto trade-off curves (for the multi-case and sinusoid experiments).

## Notes & limitations

- `sigma_yield` and `sigma_max_pzt` are defined in the configuration but are not currently enforced as constraints inside the optimization objectives — they serve as reference/design values only.
- Node ordering and node-index lookups rely on exact floating-point coordinate matching (`node_map` keyed by `(x, y)` tuples); this works for the regular parametric lattice generated by `generate_parametric_truss` but would need adjustment for arbitrarily perturbed geometries.
- Optimization bounds are hard-coded to `±200 V` per patch in each experiment block.
