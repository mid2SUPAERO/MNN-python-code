"""
@file 2piezosMNN.py
@brief Mechanical Neural Network (MNN) Simulation and Optimization framework.
@author Francesco Ardrizzini (Refactored)
@date 2026
"""

import numpy as np
import jax.numpy as jnp
import jax
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Enable 64-bit precision for high-fidelity finite element calculations
jax.config.update("jax_enable_x64", True)


# =============================================================================
#                              USER CONFIGURATION
# =============================================================================
class SimulationConfig:
    """
    Centralized configuration class. 
    Modify your geometric, material, input loads, and target displacements here.
    """
    # --- 1. Geometry & Mesh ---
    Nx = 2              # Number of base segments in width (node pairs per planar layer)
    Ny = 2              # Number of layers upward/downward relative to center
    l_beam = 100.0      # Beam length [mm]
    n_elem = 10         # Number of finite element divisions per beam
    
    # --- 2. Material & Cross-Section Properties ---
    # Structural Core
    E_beam = 2790.0     # Young's Modulus of the core beam [MPa]
    t_beam = 1.0        # Core depth thickness [mm]
    width = 20.0        # Out-of-plane beam width [mm]
    sigma_yield = 38.0  # Material yield stress threshold [MPa]
    
    # Piezoelectric Patches
    E_pzt = 62000.0     # Elastic Young's Modulus [MPa]
    t_pzt = 0.5         # Piezo layer thickness [mm]
    d31 = -274e-9       # Piezoelectric coupling strain constant [mm/V]
    sigma_max_pzt = 50.0 # Allowable fracture safety limit stress [MPa]

    # --- 3. Input Loads (Mechanical Forces) ---
    F_case1 = 100.0     # External load case 1 value [N] (e.g., pushing horizontally)
    F_case2 = 100.0     # External load case 2 value [N] (e.g., pushing vertically)
    
    # --- 4. Target Conditions (Outputs) ---
    # Targets for Single Load Case Optimization [ux, uy] in mm
    single_target_1 = [5e-4, 9e-4]
    single_target_2 = [7e-4, 7e-4]

    # Targets for Multi-Case Optimization [ux, uy] in mm
    multi_t1_case1 = [0.0005, 0.0009]
    multi_t2_case1 = [0.0007, 0.0007]
    multi_t1_case2 = [0.0003, 0.0005]
    multi_t2_case2 = [0.0008, 0.0001]

    # Target for Pareto Sinusoid Morphing Front
    sinusoid_amplitude = .05 # Amplitude of the target sinusoidal wave [mm]
    
    # --- 5. Execution Flags (Toggle True/False to run specific blocks) ---
    run_random_voltage_test = False
    run_single_case_opt = False
    run_multi_case_opt = True
    run_pareto_sinusoid_opt = False


# =============================================================================
#                               CORE FEA FUNCTIONS
# =============================================================================

def fea_setup(n_elem, init_coordinates, init_connectivity_table):
    """Builds the Finite Element mesh by subdividing beams."""
    n_nodes = n_elem + 1
    nodes_coord = []
    connectivity_table = []
    node_index = {}
    
    for i, coord in enumerate(init_coordinates):
        new_node = (float(coord[0]), float(coord[1]))
        node_index[new_node] = len(nodes_coord)
        nodes_coord.append([float(coord[0]), float(coord[1])])

    # Subdivide beams into n_elem finite elements
    for beam in init_connectivity_table:
        start = init_coordinates[beam[0]]
        end = init_coordinates[beam[1]]
        dx, dy = end[0] - start[0], end[1] - start[1]
        local_nodes = []
    
        for i in range(n_nodes):
            x, y = start[0] + dx * i / n_elem, start[1] + dy * i / n_elem
            new_node = (float(x), float(y))
            if new_node not in node_index:
                node_index[new_node] = len(nodes_coord)
                nodes_coord.append([float(x), float(y)])
            local_nodes.append(node_index[new_node])
    
        for i in range(n_elem):
            connectivity_table.append([local_nodes[i], local_nodes[i + 1]])
    
    coordinates = jnp.array(nodes_coord)
    connectivity_table = jnp.array(connectivity_table)
    
    # Sort and renumber nodes by Y-level (centre -> up -> down) and then X
    levels = {}
    for i, (_, y) in enumerate(coordinates):
        levels.setdefault(float(y), []).append(i)
    
    y0 = [y for y in levels.keys() if y == 0]
    y_pos = sorted([y for y in levels.keys() if y > 0])
    y_neg = sorted([y for y in levels.keys() if y < 0], reverse=True)
    
    new_order = []
    for y in (y0 + y_pos + y_neg):
        new_order.extend(sorted(levels[y], key=lambda i: coordinates[i, 0]))
    
    old_to_new = {old: new for new, old in enumerate(new_order)}
    coordinates = coordinates[jnp.array(new_order)]
    connectivity_table = jnp.array([[old_to_new[float(n1)], old_to_new[float(n2)]] for n1, n2 in connectivity_table]) 
    
    return n_nodes, coordinates, connectivity_table, old_to_new


def boundary_conditions(coordinates, connectivity_table, fixed_nodes_original, node_map):
    """Locks appropriate Degrees of Freedom (DOFs) for fixed anchor nodes."""
    dofs_bcs = []
    for old_node in fixed_nodes_original:
        new_node_idx = int(node_map[float(old_node)])
        # Lock u, v, theta
        dofs_bcs.extend([3 * new_node_idx, 3 * new_node_idx + 1, 3 * new_node_idx + 2])
        
    n_elements = connectivity_table.shape[0]
    n_nodes = coordinates.shape[0]
    n_dofs = 3 * n_nodes
    
    dofs = np.arange(n_dofs)
    dofs_no_bcs = list(set(dofs) - set(dofs_bcs))
    return n_elements, n_nodes, n_dofs, dofs, dofs_no_bcs


def beam_2d_element(coord1, coord2, E, I, A):
    """Computes the 6x6 global stiffness matrix for a 2D Euler-Bernoulli beam element."""
    L = jnp.linalg.norm(coord2 - coord1)
    x1, y1 = coord1
    x2, y2 = coord2
    c, s = (x2 - x1) / L, (y2 - y1) / L 
    
    K = jnp.array([
        [(E * A) / L, 0, 0, -(E * A) / L, 0, 0],
        [0, (12 * E * I) / L**3, (6 * E * I) / L**2, 0, (-12 * E * I) / L**3, (6 * E * I) / L**2],
        [0, (6 * E * I) / L**2, (4 * E * I) / L, 0, (-6 * E * I) / L**2, (2 * E * I) / L],
        [-(E * A) / L, 0, 0, (E * A) / L, 0, 0],
        [0, -(12 * E * I) / L**3, (-6 * E * I) / L**2, 0, (12 * E * I) / L**3, (-6 * E * I) / L**2],
        [0, (6 * E * I) / L**2, (2 * E * I) / L, 0, (-6 * E * I) / L**2, (4 * E * I) / L]
    ])
    
    T = jnp.array([
        [c, s, 0, 0, 0, 0], [-s, c, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
        [0, 0, 0, c, s, 0], [0, 0, 0, -s, c, 0], [0, 0, 0, 0, 0, 1]
    ])
    
    return T.T @ K @ T


def global_reduced_K(n_dofs, dofs_no_bcs, n_elements, connectivity_table, coordinates, E_eq, I_eq, A_eq):
    """Assembles the global stiffness matrix and condenses it using Boundary Conditions."""
    K = jnp.zeros((n_dofs, n_dofs))
    for ii in range(n_elements):
        left_node, right_node = connectivity_table[ii] 
        K_element = beam_2d_element(coordinates[left_node], coordinates[right_node], E_eq, I_eq, A_eq) 
        
        dof_left, dof_right = 3 * left_node, 3 * right_node
        
        K = K.at[dof_left:dof_left + 3, dof_left:dof_left + 3].add(K_element[0:3, 0:3])
        K = K.at[dof_right:dof_right + 3, dof_right:dof_right + 3].add(K_element[3:6, 3:6])
        K = K.at[dof_left:dof_left + 3, dof_right:dof_right + 3].add(K_element[0:3, 3:6])
        K = K.at[dof_right:dof_right + 3, dof_left:dof_left + 3].add(K_element[3:6, 0:3])
    
    # Static condensation
    return K[jnp.ix_(jnp.array(dofs_no_bcs), jnp.array(dofs_no_bcs))]


def beam_voltage_to_element_voltage(V_beams, n_elem):
    """Maps macroscopic beam voltages to the underlying finite elements."""
    V_beams = jnp.asarray(V_beams)
    return jnp.repeat(V_beams[:, 0], n_elem), jnp.repeat(V_beams[:, 1], n_elem)

    
def piezo_element_load(coord1, coord2, V_top, V_bot, cst_axial, cst_bending):
    """Computes the equivalent nodal load vector for a piezo-actuated beam element."""
    L = jnp.linalg.norm(coord2 - coord1)
    c, s = (coord2[0] - coord1[0]) / L, (coord2[1] - coord1[1]) / L

    N_p = cst_axial * (V_top + V_bot)
    M_p = cst_bending * (V_top - V_bot)

    f_loc = jnp.array([-N_p / 2, 0.0, -M_p / 2, N_p / 2, 0.0, M_p / 2])
    T = jnp.array([
        [c, s, 0, 0, 0, 0], [-s, c, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
        [0, 0, 0, c, s, 0], [0, 0, 0, -s, c, 0], [0, 0, 0, 0, 0, 1]
    ])
    return T.T @ f_loc


def reduced_global_f(f, dofs_no_bcs, width, E_pzt, d31, t_pzt, y_pzt, coordinates, connectivity_table, V_top_elem, V_bot_elem):
    """Assembles the global load vector (mechanical + piezo) and applies boundary constraints."""
    cst_axial = width * E_pzt * d31 / t_pzt
    cst_bending = width * E_pzt * d31 * y_pzt

    n1, n2 = connectivity_table[:, 0], connectivity_table[:, 1]
    vec_piezo_element_load = jax.vmap(piezo_element_load, in_axes=(0, 0, 0, 0, None, None))
    f_e_all = vec_piezo_element_load(coordinates[n1], coordinates[n2], V_top_elem, V_bot_elem, cst_axial, cst_bending)

    idx_u1, idx_v1, idx_theta1 = 3 * n1, 3 * n1 + 1, 3 * n1 + 2
    idx_u2, idx_v2, idx_theta2 = 3 * n2, 3 * n2 + 1, 3 * n2 + 2
    indices = jnp.stack([idx_u1, idx_v1, idx_theta1, idx_u2, idx_v2, idx_theta2], axis=1)

    f = f.at[indices].add(f_e_all)
    return f[jnp.array(dofs_no_bcs)]


def solve_fem(n_elem, coordinates, connectivity_table, n_dofs, dofs_no_bcs, K_red, 
              width, E_pzt, d31, t_pzt, y_pzt, V_beams, f_mechanical):
    """Solves the static FE system: K * u = F."""
    if isinstance(V_beams, (tuple, list)) and len(V_beams) == 2:
        V_beams = jnp.stack([jnp.asarray(V_beams[0]), jnp.asarray(V_beams[1])], axis=1)
    else:
        V_beams = jnp.asarray(V_beams)
        if V_beams.ndim == 1:
            V_beams = V_beams.reshape((-1, 2), order='F')

    V_top_elem, V_bot_elem = beam_voltage_to_element_voltage(V_beams, n_elem)
    f_red = reduced_global_f(f_mechanical, dofs_no_bcs, width, E_pzt, d31, t_pzt, y_pzt,
                             coordinates, connectivity_table, V_top_elem, V_bot_elem)
    q_red = jnp.linalg.solve(K_red, f_red)
    return jnp.zeros(n_dofs).at[jnp.array(dofs_no_bcs)].set(q_red)


# =============================================================================
#                            OPTIMIZATION OBJECTIVES
# =============================================================================

def obj_single_case(V_opt, target1, target2, n_elem, coordinates, connectivity_table, n_dofs, dofs_no_bcs, K_red, width, E_pzt, d31, t_pzt, y_pzt, F_mech, node_t1, node_t2):
    """Minimizes MSE between attained and target displacement for a single load case."""
    # Pass V_opt directly instead of slicing it
    q = solve_fem(n_elem, coordinates, connectivity_table, n_dofs, dofs_no_bcs, K_red, width, E_pzt, d31, t_pzt, y_pzt, V_opt, F_mech)
    
    scale_opt = 1e3  
    u1 = scale_opt * jnp.array([q[node_t1 * 3], q[node_t1 * 3 + 1]])
    u2 = scale_opt * jnp.array([q[node_t2 * 3], q[node_t2 * 3 + 1]])
    return jnp.sum((u1 - scale_opt * target1)**2) + jnp.sum((u2 - scale_opt * target2)**2)


def obj_multi_case(V_opt, t1_1, t1_2, t2_1, t2_2, n_elem, coordinates, connectivity_table, n_dofs, dofs_no_bcs, K_red, width, E_pzt, d31, t_pzt, y_pzt, F_mech1, F_mech2, node_t1, node_t2):
    """Minimizes MSE between attained and target displacement across TWO independent load cases."""
    # Pass V_opt directly to both systems
    q1 = solve_fem(n_elem, coordinates, connectivity_table, n_dofs, dofs_no_bcs, K_red, width, E_pzt, d31, t_pzt, y_pzt, V_opt, F_mech1)
    q2 = solve_fem(n_elem, coordinates, connectivity_table, n_dofs, dofs_no_bcs, K_red, width, E_pzt, d31, t_pzt, y_pzt, V_opt, F_mech2)

    scale_opt = 1e3
    dof_u1, dof_u2 = node_t1 * 3, node_t2 * 3

    u1_c1, u2_c1 = scale_opt * jnp.array([q1[dof_u1], q1[dof_u1+1]]), scale_opt * jnp.array([q1[dof_u2], q1[dof_u2+1]])
    u1_c2, u2_c2 = scale_opt * jnp.array([q2[dof_u1], q2[dof_u1+1]]), scale_opt * jnp.array([q2[dof_u2], q2[dof_u2+1]])

    e1 = jnp.sum((u1_c1 - scale_opt*t1_1)**2) + jnp.sum((u2_c1 - scale_opt*t1_2)**2)
    e2 = jnp.sum((u1_c2 - scale_opt*t2_1)**2) + jnp.sum((u2_c2 - scale_opt*t2_2)**2)
    return e1, e2


def sinusoid_objective_split(V_opt, right_nodes, targets_c1, targets_c2, f_m1, f_m2, n_elem, coordinates, connectivity_table, n_dofs, dofs_no_bcs, K_red, width, E_pzt, d31, t_pzt, y_pzt):
    """Computes split objective errors for the Sinusoidal Pareto Optimization."""
    q1 = solve_fem(n_elem, coordinates, connectivity_table, n_dofs, dofs_no_bcs, K_red, width, E_pzt, d31, t_pzt, y_pzt, V_opt, f_m1)
    q2 = solve_fem(n_elem, coordinates, connectivity_table, n_dofs, dofs_no_bcs, K_red, width, E_pzt, d31, t_pzt, y_pzt, V_opt, f_m2)

    scale = 1e3 
    idx = jnp.array(right_nodes)
    
    u_out1 = jnp.stack([q1[idx * 3], q1[idx * 3 + 1]], axis=1)
    u_out2 = jnp.stack([q2[idx * 3], q2[idx * 3 + 1]], axis=1)

    err1 = jnp.mean(jnp.sum(((u_out1 - targets_c1) * scale)**2, axis=1))
    err2 = jnp.mean(jnp.sum(((u_out2 - targets_c2) * scale)**2, axis=1))
    return err1, err2


# =============================================================================
#                        GEOMETRY & VISUALIZATION
# =============================================================================

def generate_parametric_truss(Nx=2, Ny=2, l=400.0, h=None):
    """Generates base MNN geometric topology."""
    if h is None: h = 0.86602540378 * l  # preserve equilateral triangles
    node_map, init_coordinates, init_connectivity = {}, [], []
    current_id = 0
    layer_order = [0] + list(range(1, Ny + 1)) + list(range(-1, -Ny - 1, -1))
    
    for j in layer_order:
        num_k = Nx if j % 2 == 0 else Nx + 1
        offset = 0 if j % 2 == 0 else -0.5 * l
        for k in range(num_k):
            init_coordinates.append([k * l + offset, j * h])
            node_map[(j, k)] = current_id
            current_id += 1
            
    for j in layer_order:
        for k in range((Nx if j % 2 == 0 else Nx + 1) - 1):
            init_connectivity.append([node_map[(j, k)], node_map[(j, k + 1)]])
            
    for j in range(-Ny, Ny):
        for k in range(Nx if j % 2 == 0 else Nx + 1):
            n_base = node_map[(j, k)]
            if j % 2 == 0:
                init_connectivity.extend([[n_base, node_map[(j+1, k)]], [n_base, node_map[(j+1, k+1)]]])
            else:
                if k - 1 >= 0: init_connectivity.append([n_base, node_map[(j+1, k-1)]])
                if k < Nx:     init_connectivity.append([n_base, node_map[(j+1, k)]])
                    
    fixed_anchors = [node_map[(Ny, k)] for k in range(Nx if Ny % 2 == 0 else Nx + 1)] + \
                    [node_map[(-Ny, k)] for k in range(Nx if (-Ny) % 2 == 0 else Nx + 1)]
    input_nodes = [node_map[(1, 0)], node_map[(-1, 0)]]
    output_nodes = [node_map[(1, Nx)], node_map[(-1, Nx)]]
    
    return jnp.array(init_coordinates), jnp.array(init_connectivity), fixed_anchors, input_nodes, output_nodes


def plot_deformed_structure(coordinates, connectivity_table, q, V_elem_top, V_elem_bot, scale, target1=None, target2=None, node_target_1=None, node_target_2=None):
    """Renders the deformed layout with voltages color-mapped onto piezo patches."""
    u, v = q[0::3], q[1::3]
    coord_def = coordinates.copy()
    coord_def = coord_def.at[:, 0].add(scale * u).at[:, 1].add(scale * v)

    plt.figure(figsize=(7, 7))
    for n1, n2 in connectivity_table:
        plt.plot([coordinates[n1, 0], coordinates[n2, 0]], [coordinates[n1, 1], coordinates[n2, 1]], color='gray', linewidth=1.0, zorder=1)

    offset = 10  
    for e, (n1, n2) in enumerate(connectivity_table):
        x1, y1, x2, y2 = coord_def[n1, 0], coord_def[n1, 1], coord_def[n2, 0], coord_def[n2, 1]
        nx, ny = -(y2 - y1) / jnp.hypot(x2 - x1, y2 - y1), (x2 - x1) / jnp.hypot(x2 - x1, y2 - y1)

        plt.plot([x1, x2], [y1, y2], color='black', linewidth=1.0, zorder=2)
        plt.plot([x1 + offset * nx, x2 + offset * nx], [y1 + offset * ny, y2 + offset * ny],
                 color='red' if V_elem_top[e] > 0 else ('blue' if V_elem_top[e] < 0 else 'darkgrey'), linewidth=1.0, zorder=3)
        plt.plot([x1 - offset * nx, x2 - offset * nx], [y1 - offset * ny, y2 - offset * ny],
                 color='red' if V_elem_bot[e] > 0 else ('blue' if V_elem_bot[e] < 0 else 'darkgrey'), linewidth=1.0, zorder=3)

    if target1 is not None and node_target_1 is not None:
        plt.scatter(coordinates[node_target_1, 0] + target1[0]*scale, coordinates[node_target_1, 1] + target1[1]*scale, marker='x', s=120, color='black', zorder=5)
    if target2 is not None and node_target_2 is not None:
        plt.scatter(coordinates[node_target_2, 0] + target2[0]*scale, coordinates[node_target_2, 1] + target2[1]*scale, marker='x', s=120, color='black', zorder=5, label='Targets')
        plt.legend()

    plt.axis('equal'); plt.xlabel("x [mm]"); plt.ylabel("y [mm]"); plt.title(f"Deformed Mesh (scale = {scale})")
    plt.grid(True, linestyle='--', alpha=0.3); plt.show()


# =============================================================================
#                               MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    cfg = SimulationConfig()

    # Derived Properties
    A_beam = cfg.t_beam * cfg.width 
    I_beam = cfg.width * cfg.t_beam**3 / 12  
    A_pzt = cfg.t_pzt * 10       
    y_pzt = (cfg.t_beam + cfg.t_pzt) / 2  
    I_pzt = 2 * (cfg.width * cfg.t_pzt**3 / 12 + cfg.width * cfg.t_pzt * y_pzt**2)  
    
    A_eq = A_beam + 2 * A_pzt
    E_eq = (cfg.E_beam * A_beam + 2 * (cfg.E_pzt * A_pzt)) / A_eq
    I_eq = (cfg.E_beam * I_beam + cfg.E_pzt * I_pzt) / E_eq

    # 1. Geometry Setup
    init_coords, init_conn, anchors, inputs, outputs = generate_parametric_truss(cfg.Nx, cfg.Ny, cfg.l_beam)
    n_node, coords, conn, node_map = fea_setup(cfg.n_elem, init_coords, init_conn)
    n_beams, n_elements = init_conn.shape[0], conn.shape[0]

    node_f1, node_f2 = int(node_map[inputs[0]]), int(node_map[inputs[1]])
    node_t1, node_t2 = int(node_map[outputs[0]]), int(node_map[outputs[1]])

    # 2. Matrix Assembly & BCs
    n_elements, n_nodes, n_dofs, dofs, dofs_no_bcs = boundary_conditions(coords, conn, anchors, node_map)
    K_red = global_reduced_K(n_dofs, dofs_no_bcs, n_elements, conn, coords, E_eq, I_eq, A_eq)
    
    # 3. Define External Force Vectors
    f_mech_c1, f_mech_c2 = jnp.zeros(n_dofs), jnp.zeros(n_dofs)
    # Case 1 pushes horizontally (X-axis)
    f_mech_c1 = f_mech_c1.at[node_f1 * 3].add(cfg.F_case1).at[node_f2 * 3].add(cfg.F_case1)
    # Case 2 pushes vertically (Y-axis)
    f_mech_c2 = f_mech_c2.at[node_f1 * 3 + 1].add(cfg.F_case2).at[node_f2 * 3 + 1].add(cfg.F_case2)

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: Random Voltage Test
    # -------------------------------------------------------------------------
    if cfg.run_random_voltage_test:
        print("\n--- Running Random Voltage Test ---")
        V_rand = np.random.uniform(-150, 150, size=n_beams * 2)
        q_rand = solve_fem(cfg.n_elem, coords, conn, n_dofs, dofs_no_bcs, K_red, cfg.width, cfg.E_pzt, cfg.d31, cfg.t_pzt, y_pzt, V_rand, f_mech_c1)
        V_top_elem, V_bot_elem = beam_voltage_to_element_voltage(V_rand.reshape((-1, 2), order='F'), cfg.n_elem)
        plot_deformed_structure(coords, conn, q_rand, V_top_elem, V_bot_elem, scale=3e2)

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: Single Load Case Optimization
    # -------------------------------------------------------------------------
    if cfg.run_single_case_opt:
        print("\n--- Optimizing Single Load Case ---")
        t1, t2 = jnp.array(cfg.single_target_1), jnp.array(cfg.single_target_2)
        
        @jax.jit
        def loss_single(V):
            return jax.value_and_grad(obj_single_case)(V, t1, t2, cfg.n_elem, coords, conn, n_dofs, dofs_no_bcs, K_red, cfg.width, cfg.E_pzt, cfg.d31, cfg.t_pzt, y_pzt, f_mech_c1, node_t1, node_t2)

        res = minimize(lambda V: (float(loss_single(V)[0]), np.array(loss_single(V)[1])), np.zeros(n_beams*2), method='SLSQP', bounds=[(-200, 200)]*(n_beams*2), jac=True, options={'disp': True})
        print(f"Success: {res.success}. Objective Value: {res.fun}")

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: Multi-Load Case Optimization
    # -------------------------------------------------------------------------
    if cfg.run_multi_case_opt:
        print("\n--- Optimizing Multi Load Case ---")
        t1_1, t1_2 = jnp.array(cfg.multi_t1_case1), jnp.array(cfg.multi_t2_case1)
        t2_1, t2_2 = jnp.array(cfg.multi_t1_case2), jnp.array(cfg.multi_t2_case2)

        pareto_errors = []
        weights = np.linspace(0.0, 1.0, 11)

        for w in weights:
            @jax.jit
            # Define the scalar loss function
            def loss_fn(V):
                e1, e2 = obj_multi_case(V, t1_1, t1_2, t2_1, t2_2, cfg.n_elem, coords, conn, n_dofs, dofs_no_bcs, K_red, cfg.width, cfg.E_pzt, cfg.d31, cfg.t_pzt, y_pzt, f_mech_c1, f_mech_c2, node_t1, node_t2)
                return w * e1 + (1.0 - w) * e2

            # Wrap it with value_and_grad to get both the loss and the gradients, then JIT compile
            compiled_loss = jax.jit(jax.value_and_grad(loss_fn))

            # Run the optimizer (compiled_loss(V)[0] and [1] will now work properly)
            res = minimize(
                lambda V: (float(compiled_loss(V)[0]), np.array(compiled_loss(V)[1])), 
                np.zeros(n_beams * 2), 
                method='SLSQP', 
                bounds=[(-200, 200)] * (n_beams * 2), 
                jac=True,
                options={'ftol': 1e-9, 'maxiter': 200}
            )
            e1, e2 = obj_multi_case(res.x, t1_1, t1_2, t2_1, t2_2, cfg.n_elem, coords, conn, n_dofs, dofs_no_bcs, K_red, cfg.width, cfg.E_pzt, cfg.d31, cfg.t_pzt, y_pzt, f_mech_c1, f_mech_c2, node_t1, node_t2)
            
            pareto_errors.append((float(e1), float(e2)))
            print(f"Weight w={w:.1f} -> Case 1 Error: {float(e1):.4e}, Case 2 Error: {float(e2):.4e}")

            # Plot extremes (w=0 vs w=1)
            if w == 0.0 or w == 1.0:
                V_top, V_bot = res.x[:n_beams], res.x[n_beams:]
                Vt_elem, Vb_elem = beam_voltage_to_element_voltage(jnp.array([V_top, V_bot]).T, cfg.n_elem)
                q_opt = solve_fem(cfg.n_elem, coords, conn, n_dofs, dofs_no_bcs, K_red, cfg.width, cfg.E_pzt, cfg.d31, cfg.t_pzt, y_pzt, res.x, f_mech_c2 if w == 0 else f_mech_c1)
                
                t1_plot = t2_1 if w == 0.0 else t1_1
                t2_plot = t2_2 if w == 0.0 else t1_2

                plot_deformed_structure(coords, conn, q_opt, Vt_elem, Vb_elem, scale=500, target1=t1_plot, target2=t2_plot, node_target_1=node_t1, node_target_2=node_t2)

        # Plot Pareto Front Curve
        plt.figure(figsize=(6, 5))
        plt.plot(*zip(*pareto_errors), 'o--', color='darkblue', label='MNN Trade-off Curve')
        plt.xlabel('Case 1 Residual Error')
        plt.ylabel('Case 2 Residual Error')
        plt.title('Pareto Front Optimization Curve')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()

    # -------------------------------------------------------------------------
    # EXPERIMENT 4: Pareto Sweep for Sinusoid Outputs
    # -------------------------------------------------------------------------
    if cfg.run_pareto_sinusoid_opt:
        print("\n--- Optimizing Sinusoid Pareto Front ---")
        
        # 1. Identify Output Nodes (Right boundary)
        x_max = jnp.max(coords[:, 0])
        right_nodes = sorted([i for i in range(len(coords)) if np.isclose(coords[i, 0], x_max)], key=lambda idx: coords[idx, 1])
        right_nodes_jax = jnp.array(right_nodes)
        
        # 2. Define Sinusoid Profiles (normalized Y map)
        y_r = jnp.array([coords[idx, 1] for idx in right_nodes])

        y_global_min = jnp.min(coords[:,1])
        y_global_max = jnp.max(coords[:,1])

        norm_y = (y_r - y_global_min) / (y_global_max - y_global_min)
        
        # UX varies like a Sine wave, UY is mapped to 0. Case 2 opposes Case 1.
        targets_c1 = jnp.stack([cfg.sinusoid_amplitude * jnp.sin(2*jnp.pi * norm_y), jnp.zeros_like(norm_y)], axis=1) 
        targets_c2 = jnp.stack([-cfg.sinusoid_amplitude * jnp.sin(2*jnp.pi * norm_y), jnp.zeros_like(norm_y)], axis=1)

        pareto_errors = []
        weights = np.linspace(0.0, 1.0, 11)
        
        for w in weights:
            @jax.jit
            # Define the scalar loss function
            def loss_fn(V):
                e1, e2 = sinusoid_objective_split(
                    V, right_nodes_jax, targets_c1, targets_c2, f_mech_c1, f_mech_c2, 
                    cfg.n_elem, coords, conn, n_dofs, dofs_no_bcs, K_red, 
                    cfg.width, cfg.E_pzt, cfg.d31, cfg.t_pzt, y_pzt
                )
                return w * e1 + (1.0 - w) * e2

            # Wrap it with value_and_grad to get both the loss and the gradients, then JIT compile
            compiled_loss = jax.jit(jax.value_and_grad(loss_fn))

            # Run the optimizer (compiled_loss(V)[0] and [1] will now work properly)
            res = minimize(
                lambda V: (float(compiled_loss(V)[0]), np.array(compiled_loss(V)[1])), 
                np.zeros(n_beams * 2), 
                method='SLSQP', 
                bounds=[(-200, 200)] * (n_beams * 2), 
                jac=True,
                options={'ftol': 1e-9, 'maxiter': 200}
            )
            e1, e2 = sinusoid_objective_split(res.x, right_nodes_jax, targets_c1, targets_c2, f_mech_c1, f_mech_c2, cfg.n_elem, coords, conn, n_dofs, dofs_no_bcs, K_red, cfg.width, cfg.E_pzt, cfg.d31, cfg.t_pzt, y_pzt)
            
            pareto_errors.append((float(e1), float(e2)))
            print(f"Weight w={w:.1f} -> Case 1 Error: {float(e1):.4e}, Case 2 Error: {float(e2):.4e}")

            # Plot extremes (w=0 vs w=1)
            if w == 0.0 or w == 1.0:
                V_top, V_bot = res.x[:n_beams], res.x[n_beams:]
                Vt_elem, Vb_elem = beam_voltage_to_element_voltage(jnp.array([V_top, V_bot]).T, cfg.n_elem)
                q_opt = solve_fem(cfg.n_elem, coords, conn, n_dofs, dofs_no_bcs, K_red, cfg.width, cfg.E_pzt, cfg.d31, cfg.t_pzt, y_pzt, res.x, f_mech_c2 if w == 0 else f_mech_c1)
                
                t1 = targets_c2[right_nodes.index(node_t1)] if w == 0 else targets_c1[right_nodes.index(node_t1)]
                t2 = targets_c2[right_nodes.index(node_t2)] if w == 0 else targets_c1[right_nodes.index(node_t2)]
                
                plot_deformed_structure(coords, conn, q_opt, Vt_elem, Vb_elem, scale=500, target1=t1, target2=t2, node_target_1=node_t1, node_target_2=node_t2)

        # Plot Pareto Front Curve
        plt.figure(figsize=(6, 5))
        plt.plot(*zip(*pareto_errors), 'o--', color='darkblue', label='MNN Trade-off Curve')
        plt.xlabel('Case 1 Residual Error (Sinusoid)')
        plt.ylabel('Case 2 Residual Error (Opposed Sinusoid)')
        plt.title('Pareto Front Optimization Curve')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()