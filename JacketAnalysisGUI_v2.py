# -*- coding: utf-8 -*-
"""
Morison Equation + FEM Structural Analysis for Multi-Legged Jacket Structure
INTERACTIVE GUI VERSION v8 - CUSTOMIZABLE GEOMETRY

Features:
- Fully customizable node geometry (any number of legs and levels)
- User-defined member connections
- Custom self-weight input option
- All material properties are customizable
- Morison equation + FEM analysis
- Raschii wave library for nonlinear wave theory

Author：王中王，火腿肠，果冻我吃喜之郎，看完代码如果觉得还不错记得一键三连，有问题随时dd我，如果你线下也有我的联系方式的话，做兄弟，在心中，有事电话打不通，谢谢
"""

import subprocess
import sys

# =============================================================================
# AUTOMATIC DEPENDENCY INSTALLATION
# =============================================================================
def install_package(package_name, pip_name=None):
    if pip_name is None:
        pip_name = package_name
    try:
        __import__(package_name)
        return True
    except ImportError:
        print(f"Installing {pip_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])
            print(f"  ✓ {pip_name} installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to install {pip_name}: {e}")
            return False

def check_and_install_dependencies():
    print("=" * 60)
    print("Checking dependencies...")
    print("=" * 60)
    
    required_packages = [("numpy", "numpy"), ("pandas", "pandas"), 
                        ("matplotlib", "matplotlib"), ("tkinter", None)]
    optional_packages = [("raschii", "raschii")]
    
    for import_name, pip_name in required_packages:
        if pip_name is None:
            try:
                __import__(import_name)
                print(f"  ✓ {import_name} is available")
            except ImportError:
                print(f"  ✗ {import_name} not available")
        else:
            try:
                __import__(import_name)
                print(f"  ✓ {import_name} is available")
            except ImportError:
                install_package(import_name, pip_name)
    
    print("\nOptional packages:")
    raschii_available = False
    for import_name, pip_name in optional_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {import_name} is available")
            raschii_available = True
        except ImportError:
            print(f"  ? {import_name} not found, attempting to install...")
            if install_package(import_name, pip_name):
                raschii_available = True
    
    print("=" * 60 + "\n")
    return raschii_available

RASCHII_AVAILABLE = check_and_install_dependencies()

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import json
import copy

warnings.filterwarnings('ignore')

try:
    import raschii
    RASCHII_AVAILABLE = True
except ImportError:
    RASCHII_AVAILABLE = False

# =============================================================================
# CONSTANTS
# =============================================================================
g = 9.81
DEFAULT_RHO_WATER = 1025
DEFAULT_E = 210000
DEFAULT_NU = 0.3
DEFAULT_FY = 355
DEFAULT_RHO_STEEL = 7850

# =============================================================================
# TUBULAR SECTION CLASS
# =============================================================================
@dataclass
class TubularSection:
    D_outer: float
    t: float
    name: str = ""
    rho_steel: float = 7850

    def __post_init__(self):
        self.D_inner = self.D_outer - 2 * self.t
        self.R_outer = self.D_outer / 2.0
        self.R_inner = self.D_inner / 2.0
        self.Ax_mm2 = np.pi / 4.0 * (self.D_outer**2 - self.D_inner**2)
        self.Ax_m2 = self.Ax_mm2 / 1e6
        self.Iy_mm4 = np.pi / 64.0 * (self.D_outer**4 - self.D_inner**4)
        self.Iz_mm4 = self.Iy_mm4
        self.Ix_mm4 = np.pi / 32.0 * (self.D_outer**4 - self.D_inner**4)
        self.Ay_mm2 = 0.5 * self.Ax_mm2
        self.Az_mm2 = 0.5 * self.Ax_mm2
        self.Wy_mm3 = self.Iy_mm4 / self.R_outer
        self.Wz_mm3 = self.Iz_mm4 / self.R_outer
        self.Wx_mm3 = self.Ix_mm4 / self.R_outer
        self.mass_per_m = self.Ax_m2 * self.rho_steel
        self.D_t_ratio = self.D_outer / self.t

    def get_stress_points(self):
        R = self.R_outer
        points = {}
        for i, angle in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
            rad = np.radians(angle)
            points[f'A{i+1}'] = (R * np.cos(rad), R * np.sin(rad))
        return points

    def calc_stress_at_point(self, Fx, Fy, Fz, Mx, My, Mz, point_id):
        points = self.get_stress_points()
        y, z = points[point_id]
        sigma_Fx = Fx / self.Ax_mm2
        sigma_My = My * z / self.Iy_mm4 if self.Iy_mm4 > 0 else 0.0
        sigma_Mz = Mz * y / self.Iz_mm4 if self.Iz_mm4 > 0 else 0.0
        sigma_total = sigma_Fx + sigma_My + sigma_Mz
        R = np.sqrt(y**2 + z**2)
        tau_Mx = Mx * R / self.Ix_mm4 if self.Ix_mm4 > 0 else 0.0
        tau_Fy = Fy / self.Ay_mm2 if self.Ay_mm2 > 0 else 0.0
        tau_Fz = Fz / self.Az_mm2 if self.Az_mm2 > 0 else 0.0
        tau_total = np.sqrt(tau_Mx**2 + tau_Fy**2 + tau_Fz**2)
        sigma_vm = np.sqrt(sigma_total**2 + 3.0 * tau_total**2)
        return {'sigma_total': sigma_total, 'tau_total': tau_total, 'von_mises': sigma_vm}


# =============================================================================
# WAVE CLASS - ENHANCED WITH AUTO MODEL SELECTION
# =============================================================================
class RaschiiWave:
    """
    Wave model wrapper with automatic model selection capability.
    Tracks which model and order was actually used.
    """
    def __init__(self, H, T, d, U_c=0.0, wave_model='Fenton', N=10, dt=0.001):
        self.H, self.T, self.d, self.U_c = H, T, d, U_c
        self.requested_model = wave_model
        self.requested_N = N
        self.dt = dt
        self.a = H / 2.0
        
        # These will be set after model creation
        self.actual_model = None
        self.actual_N = None
        self.steepness = None

        if RASCHII_AVAILABLE:
            self.wave = self._create_wave(wave_model, N)
            self.omega, self.k, self.L, self.c = self.wave.omega, self.wave.k, self.wave.length, self.wave.c
            self.steepness = self.H / self.L
        else:
            self.omega = 2.0 * np.pi / T
            self.k = self._solve_dispersion(self.omega, d)
            self.L = 2.0 * np.pi / self.k
            self.c = self.L / T
            self.wave = None
            self.actual_model = 'Airy (fallback)'
            self.actual_N = 1
            self.steepness = self.H / self.L

    def _solve_dispersion(self, omega, d):
        k = omega**2 / g
        for _ in range(50):
            f = omega**2 - g * k * np.tanh(k * d)
            df = -g * (np.tanh(k * d) + k * d / np.cosh(k * d)**2)
            k_new = k - f / df
            if abs(k_new - k) < 1e-10:
                break
            k = k_new
        return k

    def _create_wave(self, model, N):
        """Create wave with auto-selection based on steepness if requested."""
        
        # First, calculate steepness using Airy to decide
        airy_test = raschii.AiryWave(height=self.H, depth=self.d, period=self.T)
        steepness = self.H / airy_test.length
        
        if model.lower() == 'auto':
            # Automatic model selection based on wave steepness
            if steepness < 0.01:
                # Very mild waves - Airy is sufficient
                self.actual_model = 'Airy'
                self.actual_N = 1
                return raschii.AiryWave(height=self.H, depth=self.d, period=self.T)
            elif steepness < 0.03:
                # Mild waves - Stokes 3rd order
                self.actual_model = 'Stokes'
                self.actual_N = 3
                return raschii.StokesWave(height=self.H, depth=self.d, period=self.T, N=3)
            elif steepness < 0.06:
                # Moderate waves - Stokes 5th order
                self.actual_model = 'Stokes'
                self.actual_N = 5
                return raschii.StokesWave(height=self.H, depth=self.d, period=self.T, N=5)
            else:
                # Steep waves - Fenton stream function
                N_fenton = min(max(int(steepness * 200), 10), 20)  # 10-20 based on steepness
                self.actual_model = 'Fenton'
                self.actual_N = N_fenton
                return raschii.FentonWave(height=self.H, depth=self.d, period=self.T, N=N_fenton)
        
        elif model.lower() == 'fenton':
            self.actual_model = 'Fenton'
            self.actual_N = N
            return raschii.FentonWave(height=self.H, depth=self.d, period=self.T, N=N)
        
        elif model.lower() == 'stokes':
            actual_N = min(N, 5)  # Stokes max is 5
            self.actual_model = 'Stokes'
            self.actual_N = actual_N
            return raschii.StokesWave(height=self.H, depth=self.d, period=self.T, N=actual_N)
        
        else:  # Airy
            self.actual_model = 'Airy'
            self.actual_N = 1
            return raschii.AiryWave(height=self.H, depth=self.d, period=self.T)
    
    def get_model_info(self):
        """Return string describing the wave model used."""
        return f"{self.actual_model} (Order/N={self.actual_N}), Steepness H/L={self.steepness:.4f}"

    def eta(self, x, t=0.0):
        if self.wave is not None:
            eta_raschii = self.wave.surface_elevation(x, t=t)
            if hasattr(eta_raschii, '__len__'):
                eta_raschii = eta_raschii[0]
            return eta_raschii - self.d
        return self.a * np.cos(self.k * x - self.omega * t)

    def velocity(self, x, z_mwl, t=0.0):
        eta_local = self.eta(x, t)
        if z_mwl > eta_local:
            return (0.0, 0.0)
        if self.wave is not None:
            z_raschii = max(0.01, min(z_mwl + self.d, self.d + eta_local - 0.01))
            vel = self.wave.velocity(x, z_raschii, t=t)
            u = float(vel[0, 0]) if hasattr(vel, 'shape') else float(vel[0])
            w = float(vel[0, 1]) if hasattr(vel, 'shape') else float(vel[1])
        else:
            kd, kz = self.k * self.d, self.k * (z_mwl + self.d)
            phase = self.k * x - self.omega * t
            u = self.a * self.omega * np.cosh(kz) / np.sinh(kd) * np.cos(phase)
            w = self.a * self.omega * np.sinh(kz) / np.sinh(kd) * np.sin(phase)
        return (u + self.U_c, w)

    def acceleration(self, x, z_mwl, t=0.0):
        if z_mwl > self.eta(x, t):
            return (0.0, 0.0)
        u0, w0 = self.velocity(x, z_mwl, t)
        u1, w1 = self.velocity(x, z_mwl, t + self.dt)
        return ((u1 - u0) / self.dt, (w1 - w0) / self.dt)

    def get_kinematics(self, x, z_mwl, t=0.0):
        eta_local = self.eta(x, t)
        if z_mwl > eta_local:
            return {'u': 0, 'w': 0, 'du_dt': 0, 'dw_dt': 0, 'submerged': False, 'eta': eta_local}
        u, w = self.velocity(x, z_mwl, t)
        du_dt, dw_dt = self.acceleration(x, z_mwl, t)
        return {'u': u, 'w': w, 'du_dt': du_dt, 'dw_dt': dw_dt, 'submerged': True, 'eta': eta_local}


# =============================================================================
# CUSTOMIZABLE STRUCTURE CLASS
# =============================================================================
class CustomJacketStructure:
    """
    Customizable jacket structure with user-defined nodes and members.
    """
    def __init__(self, nodes_dict, members_list, section_leg, section_brace, 
                 fixed_nodes, top_nodes, rho_steel=7850):
        """
        Parameters:
        - nodes_dict: Dict[str, np.ndarray] - node name -> [x, y, z] coordinates
        - members_list: List[Dict] - list of member definitions
        - section_leg: TubularSection for legs
        - section_brace: TubularSection for braces
        - fixed_nodes: List[str] - names of fixed (support) nodes
        - top_nodes: List[str] - names of top (interface) nodes
        """
        self.nodes = nodes_dict
        self.node_list = list(self.nodes.keys())
        self.n_nodes = len(self.node_list)
        self.n_dof = 6 * self.n_nodes
        self.node_index = {name: i for i, name in enumerate(self.node_list)}
        
        self.section_leg = section_leg
        self.section_brace = section_brace
        self.rho_steel = rho_steel
        
        self.members = []
        for m in members_list:
            section = section_leg if m.get('type', 'brace') == 'leg' else section_brace
            self.members.append({
                'name': m['name'],
                'node1': m['node1'],
                'node2': m['node2'],
                'section': section,
                'type': m.get('type', 'brace')
            })
        
        self.n_members = len(self.members)
        self._fixed_nodes = fixed_nodes
        self._top_nodes = top_nodes

    def get_member_geometry(self, member):
        coord1 = self.nodes[member['node1']]
        coord2 = self.nodes[member['node2']]
        dL = coord2 - coord1
        L = np.linalg.norm(dL)
        return {'coord1': coord1, 'coord2': coord2, 'dL': dL, 'L': L, 
                'L_mm': L * 1000.0, 'unit_vec': dL / L if L > 0 else np.array([1, 0, 0])}

    def get_top_nodes(self):
        return self._top_nodes

    def get_bottom_nodes(self):
        return self._fixed_nodes


# =============================================================================
# BEAM ELEMENT CLASS
# =============================================================================
class BeamElement3D:
    def __init__(self, node1_coords, node2_coords, section, E=210000, G=80769, include_shear=True):
        self.node1, self.node2 = node1_coords, node2_coords
        self.section, self.E, self.G = section, E, G
        self.dL = node2_coords - node1_coords
        self.L = np.linalg.norm(self.dL)
        self.L_mm = self.L * 1000.0
        self.T = self._compute_transformation_matrix()
        self.K_local = self._compute_local_stiffness(include_shear)
        self.K_global = self.T.T @ self.K_local @ self.T

    def _compute_transformation_matrix(self):
        lx = self.dL / self.L
        global_z = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(lx, global_z)) > 0.999:
            ly = np.cross(global_z, lx)
            ly_norm = np.linalg.norm(ly)
            ly = ly / ly_norm if ly_norm > 1e-10 else np.array([0.0, 1.0, 0.0])
            lz = np.cross(lx, ly)
        else:
            lz = np.cross(lx, global_z)
            lz = lz / np.linalg.norm(lz)
            ly = np.cross(lz, lx)
        R = np.array([lx, ly, lz])
        T = np.zeros((12, 12))
        for i in range(4):
            T[3*i:3*i+3, 3*i:3*i+3] = R
        return T

    def _compute_local_stiffness(self, include_shear):
        L = self.L_mm
        s = self.section
        E, G = self.E, self.G
        
        if include_shear and s.Ay_mm2 > 0 and s.Az_mm2 > 0:
            Phi_y = 12.0 * E * s.Iz_mm4 / (G * s.Az_mm2 * L**2)
            Phi_z = 12.0 * E * s.Iy_mm4 / (G * s.Ay_mm2 * L**2)
        else:
            Phi_y = Phi_z = 0.0

        alpha = E * s.Ax_mm2 / L
        bz = E * s.Iz_mm4 / ((1.0 + Phi_y) * L**3)
        by = E * s.Iy_mm4 / ((1.0 + Phi_z) * L**3)
        t = G * s.Ix_mm4 / L

        K = np.zeros((12, 12))
        K[0, 0] = K[6, 6] = alpha
        K[0, 6] = K[6, 0] = -alpha
        K[1, 1] = K[7, 7] = 12.0 * bz
        K[1, 7] = K[7, 1] = -12.0 * bz
        K[1, 5] = K[5, 1] = K[1, 11] = K[11, 1] = 6.0 * bz * L
        K[7, 5] = K[5, 7] = K[7, 11] = K[11, 7] = -6.0 * bz * L
        K[5, 5] = K[11, 11] = (4.0 + Phi_y) * bz * L**2
        K[5, 11] = K[11, 5] = (2.0 - Phi_y) * bz * L**2
        K[2, 2] = K[8, 8] = 12.0 * by
        K[2, 8] = K[8, 2] = -12.0 * by
        K[2, 4] = K[4, 2] = K[2, 10] = K[10, 2] = -6.0 * by * L
        K[8, 4] = K[4, 8] = K[8, 10] = K[10, 8] = 6.0 * by * L
        K[4, 4] = K[10, 10] = (4.0 + Phi_z) * by * L**2
        K[4, 10] = K[10, 4] = (2.0 - Phi_z) * by * L**2
        K[3, 3] = K[9, 9] = t
        K[3, 9] = K[9, 3] = -t
        return K

    def get_internal_forces(self, u_global):
        u_local = self.T @ u_global
        F_local = self.K_local @ u_local
        return {
            'node1': {'Fx': -F_local[0], 'Fy': -F_local[1], 'Fz': -F_local[2],
                      'Mx': -F_local[3], 'My': -F_local[4], 'Mz': -F_local[5]},
            'node2': {'Fx': F_local[6], 'Fy': F_local[7], 'Fz': F_local[8],
                      'Mx': F_local[9], 'My': F_local[10], 'Mz': F_local[11]}
        }


# =============================================================================
# FEM SOLVER CLASS
# =============================================================================
class FEMSolver:
    def __init__(self, structure, E=210000, nu=0.3):
        self.structure = structure
        self.n_dof = structure.n_dof
        self.E = E
        self.G = E / (2 * (1 + nu))
        self.K_global = np.zeros((self.n_dof, self.n_dof))
        self.F_global = np.zeros(self.n_dof)
        self.U_global = np.zeros(self.n_dof)
        self.elements = []
        self._build_elements()
        self._assemble_global_stiffness()

    def _build_elements(self):
        for member in self.structure.members:
            coord1 = self.structure.nodes[member['node1']]
            coord2 = self.structure.nodes[member['node2']]
            self.elements.append(BeamElement3D(coord1, coord2, member['section'], self.E, self.G))

    def _assemble_global_stiffness(self):
        for i, member in enumerate(self.structure.members):
            element = self.elements[i]
            idx1 = self.structure.node_index[member['node1']]
            idx2 = self.structure.node_index[member['node2']]
            dof1 = np.arange(6 * idx1, 6 * idx1 + 6)
            dof2 = np.arange(6 * idx2, 6 * idx2 + 6)
            dofs = np.concatenate([dof1, dof2])
            for ii, di in enumerate(dofs):
                for jj, dj in enumerate(dofs):
                    self.K_global[di, dj] += element.K_global[ii, jj]

    def apply_nodal_force(self, node_name, force_vector):
        idx = self.structure.node_index[node_name]
        self.F_global[6*idx:6*idx+6] += force_vector

    def apply_boundary_conditions(self, fixed_nodes):
        self.fixed_dofs = []
        for node_name in fixed_nodes:
            idx = self.structure.node_index[node_name]
            self.fixed_dofs.extend(range(6 * idx, 6 * idx + 6))
        self.fixed_dofs = np.array(self.fixed_dofs, dtype=int)
        self.free_dofs = np.setdiff1d(np.arange(self.n_dof), self.fixed_dofs)

    def solve(self):
        K_ff = self.K_global[np.ix_(self.free_dofs, self.free_dofs)]
        F_f = self.F_global[self.free_dofs]
        try:
            U_f = np.linalg.solve(K_ff, F_f)
        except:
            U_f = np.linalg.lstsq(K_ff, F_f, rcond=None)[0]
        self.U_global = np.zeros(self.n_dof)
        self.U_global[self.free_dofs] = U_f
        return self.U_global

    def get_reactions(self):
        R = self.K_global @ self.U_global - self.F_global
        reactions = {}
        for dof in self.fixed_dofs:
            node_idx = dof // 6
            local_dof = dof % 6
            node_name = self.structure.node_list[node_idx]
            if node_name not in reactions:
                reactions[node_name] = np.zeros(6)
            reactions[node_name][local_dof] = R[dof]
        return reactions

    def get_member_internal_forces(self, fy=355):
        results = []
        for i, member in enumerate(self.structure.members):
            element = self.elements[i]
            idx1 = self.structure.node_index[member['node1']]
            idx2 = self.structure.node_index[member['node2']]
            u_elem = np.concatenate([self.U_global[6*idx1:6*idx1+6], self.U_global[6*idx2:6*idx2+6]])
            forces = element.get_internal_forces(u_elem)
            section = member['section']
            
            max_vm = 0.0
            for pt in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']:
                st = section.calc_stress_at_point(
                    forces['node1']['Fx'], forces['node1']['Fy'], forces['node1']['Fz'],
                    forces['node1']['Mx'], forces['node1']['My'], forces['node1']['Mz'], pt)
                max_vm = max(max_vm, st['von_mises'])
            
            results.append({
                'member': member['name'], 'type': member['type'],
                'node1': member['node1'], 'node2': member['node2'],
                'length_m': element.L,
                'Fx_max_kN': max(abs(forces['node1']['Fx']), abs(forces['node2']['Fx'])) / 1000,
                'Fy_max_kN': max(abs(forces['node1']['Fy']), abs(forces['node2']['Fy'])) / 1000,
                'Fz_max_kN': max(abs(forces['node1']['Fz']), abs(forces['node2']['Fz'])) / 1000,
                'My_max_kNm': max(abs(forces['node1']['My']), abs(forces['node2']['My'])) / 1e6,
                'Mz_max_kNm': max(abs(forces['node1']['Mz']), abs(forces['node2']['Mz'])) / 1e6,
                'von_mises_max_MPa': max_vm,
                'utilization': max_vm / fy
            })
        return results


# =============================================================================
# MORISON CALCULATOR - ENHANCED VERSION
# =============================================================================
class MorisonCalculator:
    """
    Enhanced Morison force calculator with separate drag/inertia output
    and support for different wave and current directions.
    """
    def __init__(self, structure, wave, wave_direction=0.0, current_direction=0.0,
                 Cd=0.7, Cm=2.0, rho_water=1025):
        """
        Parameters:
        - wave_direction: Wave propagation direction in degrees from North (clockwise)
        - current_direction: Current direction in degrees from North (clockwise)
        """
        self.structure = structure
        self.wave = wave
        self.wave_dir_deg = wave_direction
        self.current_dir_deg = current_direction
        self.theta_wave = np.deg2rad(90.0 - wave_direction)  # Convert to math angle
        self.theta_current = np.deg2rad(90.0 - current_direction)
        self.Cd, self.Cm, self.rho = Cd, Cm, rho_water

    def get_kinematics_3d(self, x, y, z, t):
        """Get 3D wave kinematics with separate wave and current contributions."""
        # Wave coordinate (along wave direction)
        x_wave = x * np.cos(self.theta_wave) + y * np.sin(self.theta_wave)
        kin2d = self.wave.get_kinematics(x_wave, z, t)
        
        if not kin2d['submerged']:
            return {'u_wave': 0, 'v_wave': 0, 'w_wave': 0,
                    'u_current': 0, 'v_current': 0,
                    'du_dt': 0, 'dv_dt': 0, 'dw_dt': 0,
                    'submerged': False, 'eta': kin2d['eta']}
        
        # Wave velocity components (in global X-Y)
        cos_w, sin_w = np.cos(self.theta_wave), np.sin(self.theta_wave)
        u_wave_only = kin2d['u'] - self.wave.U_c  # Remove current from wave solution
        
        # Current velocity components (in global X-Y)
        cos_c, sin_c = np.cos(self.theta_current), np.sin(self.theta_current)
        
        return {
            'u_wave': u_wave_only * cos_w,
            'v_wave': u_wave_only * sin_w,
            'w_wave': kin2d['w'],
            'u_current': self.wave.U_c * cos_c,
            'v_current': self.wave.U_c * sin_c,
            'du_dt': kin2d['du_dt'] * cos_w,
            'dv_dt': kin2d['du_dt'] * sin_w,
            'dw_dt': kin2d['dw_dt'],
            'submerged': True,
            'eta': kin2d['eta']
        }

    def compute_all_morison_forces(self, t=0.0, n_gauss=15):
        """
        Compute Morison forces with detailed breakdown:
        - Drag force (from velocity)
        - Inertia force (from acceleration)
        - Total Morison force
        """
        nodal_forces = {name: np.zeros(6) for name in self.structure.nodes.keys()}
        
        # Detailed force tracking
        total_drag = np.zeros(3)
        total_inertia = np.zeros(3)
        total_morison = np.zeros(3)
        
        member_details = []
        
        for member in self.structure.members:
            coord1 = self.structure.nodes[member['node1']]
            coord2 = self.structure.nodes[member['node2']]
            D = member['section'].D_outer / 1000.0
            dL = coord2 - coord1
            L = np.linalg.norm(dL)
            unit_vec = dL / L
            
            xi, weights = np.polynomial.legendre.leggauss(n_gauss)
            s_values = (xi + 1.0) / 2.0
            w_scaled = weights / 2.0
            
            F1, F2 = np.zeros(3), np.zeros(3)
            member_drag = np.zeros(3)
            member_inertia = np.zeros(3)
            submerged_length = 0.0
            
            for s, w in zip(s_values, w_scaled):
                pos = coord1 + s * dL
                kin = self.get_kinematics_3d(*pos, t)
                if not kin['submerged']:
                    continue
                
                submerged_length += w * L
                
                # Total velocity = wave + current
                U_vec = np.array([
                    kin['u_wave'] + kin['u_current'],
                    kin['v_wave'] + kin['v_current'],
                    kin['w_wave']
                ])
                A_vec = np.array([kin['du_dt'], kin['dv_dt'], kin['dw_dt']])
                
                # Perpendicular components
                U_perp = U_vec - np.dot(U_vec, unit_vec) * unit_vec
                A_perp = A_vec - np.dot(A_vec, unit_vec) * unit_vec
                U_perp_mag = np.linalg.norm(U_perp)
                
                A_cross = np.pi * D**2 / 4.0
                
                # Separate drag and inertia
                if U_perp_mag > 1e-10:
                    F_drag = 0.5 * self.rho * self.Cd * D * U_perp_mag * U_perp * L * w
                else:
                    F_drag = np.zeros(3)
                F_inertia = self.rho * self.Cm * A_cross * A_perp * L * w
                
                f_total = F_drag + F_inertia
                
                member_drag += F_drag
                member_inertia += F_inertia
                F1 += (1.0 - s) * f_total
                F2 += s * f_total
            
            nodal_forces[member['node1']][:3] += F1
            nodal_forces[member['node2']][:3] += F2
            
            total_drag += member_drag
            total_inertia += member_inertia
            total_morison += member_drag + member_inertia
            
            member_details.append({
                'member': member['name'],
                'drag_kN': np.linalg.norm(member_drag) / 1000,
                'inertia_kN': np.linalg.norm(member_inertia) / 1000,
                'total_kN': np.linalg.norm(member_drag + member_inertia) / 1000,
                'submerged_length': submerged_length
            })
        
        return {
            'nodal_forces': nodal_forces,
            'total_drag': total_drag,
            'total_inertia': total_inertia,
            'total_morison': total_morison,
            'member_details': member_details
        }

    def find_critical_phase(self, n_steps=36):
        """
        Scan through one wave period to find the critical phase (max total force).
        
        Returns:
        - Critical time, phase angle, and forces at that instant
        """
        T = self.wave.T
        omega = self.wave.omega
        
        results = []
        for i in range(n_steps):
            t = i * T / n_steps
            phase = omega * t  # Phase angle in radians
            phase_deg = np.degrees(phase) % 360
            
            forces = self.compute_all_morison_forces(t)
            total_mag = np.linalg.norm(forces['total_morison'])
            drag_mag = np.linalg.norm(forces['total_drag'])
            inertia_mag = np.linalg.norm(forces['total_inertia'])
            
            results.append({
                't': t,
                'phase_deg': phase_deg,
                'total_kN': total_mag / 1000,
                'drag_kN': drag_mag / 1000,
                'inertia_kN': inertia_mag / 1000,
                'Fx_kN': forces['total_morison'][0] / 1000,
                'Fy_kN': forces['total_morison'][1] / 1000,
                'Fz_kN': forces['total_morison'][2] / 1000,
            })
        
        # Find maximum
        max_result = max(results, key=lambda x: x['total_kN'])
        
        return {
            'all_phases': results,
            'critical': max_result,
            'T': T,
            'omega': omega
        }


# =============================================================================
# DEFAULT 3-LEG JACKET GEOMETRY (Your original structure)
# =============================================================================
def create_default_3leg_jacket(z_water_ref=47.0):
    """Create default 3-leg jacket geometry matching your original structure."""
    nodes = {}
    
    # LEG A (nodes 1-4 from bottom to top)
    nodes['A1'] = np.array([-9.2376, -16.0,      0.0 - z_water_ref])
    nodes['A2'] = np.array([-7.9254, -13.7272,  28.41 - z_water_ref])
    nodes['A3'] = np.array([-6.7947, -11.7688,  52.89 - z_water_ref])
    nodes['A4'] = np.array([-5.8197, -10.08,    74.0  - z_water_ref])

    # LEG B
    nodes['B1'] = np.array([18.4752,  0.0,      0.0 - z_water_ref])
    nodes['B2'] = np.array([15.8508,  0.0,     28.41 - z_water_ref])
    nodes['B3'] = np.array([13.5894,  0.0,     52.89 - z_water_ref])
    nodes['B4'] = np.array([11.6394,  0.0,     74.0  - z_water_ref])

    # LEG C
    nodes['C1'] = np.array([-9.2376,  16.0,     0.0 - z_water_ref])
    nodes['C2'] = np.array([-7.9254,  13.7272, 28.41 - z_water_ref])
    nodes['C3'] = np.array([-6.7947,  11.7688, 52.89 - z_water_ref])
    nodes['C4'] = np.array([-5.8197,  10.08,   74.0  - z_water_ref])

    # Hinge nodes (Level 1)
    nodes['HAB1'] = np.array([4.2657, -7.3884, 15.291 - z_water_ref])
    nodes['HBC1'] = np.array([4.2657,  7.3884, 15.291 - z_water_ref])
    nodes['HCA1'] = np.array([-8.5313, 0.0,    15.291 - z_water_ref])

    # Hinge nodes (Level 2)
    nodes['HAB2'] = np.array([3.6583, -6.3364, 41.5902 - z_water_ref])
    nodes['HBC2'] = np.array([3.6583,  6.3364, 41.5902 - z_water_ref])
    nodes['HCA2'] = np.array([-7.3166, 0.0,    41.5902 - z_water_ref])

    # Hinge nodes (Level 3)
    nodes['HAB3'] = np.array([3.1348, -5.4296, 64.2608 - z_water_ref])
    nodes['HBC3'] = np.array([3.1348,  5.4296, 64.2608 - z_water_ref])
    nodes['HCA3'] = np.array([-6.2695, 0.0,    64.2608 - z_water_ref])
    
    # Define members
    members = []
    
    # Legs
    for leg in ['A', 'B', 'C']:
        for i in [1, 2, 3]:
            members.append({'name': f'Leg_{leg}{i}-{leg}{i+1}', 
                           'node1': f'{leg}{i}', 'node2': f'{leg}{i+1}', 'type': 'leg'})
    
    # Bottom horizontal braces
    for n1, n2 in [('A1', 'B1'), ('B1', 'C1'), ('C1', 'A1')]:
        members.append({'name': f'HBrace_{n1}-{n2}', 'node1': n1, 'node2': n2, 'type': 'h_brace'})
    
    # Level 2 horizontal braces
    for n1, n2 in [('A2', 'B2'), ('B2', 'C2'), ('C2', 'A2')]:
        members.append({'name': f'HBrace_{n1}-{n2}', 'node1': n1, 'node2': n2, 'type': 'h_brace'})
    
    # X-bracing
    xbrace_config = [
        [('A1', 'HAB1'), ('HAB1', 'B2'), ('B1', 'HAB1'), ('HAB1', 'A2'),
         ('B1', 'HBC1'), ('HBC1', 'C2'), ('C1', 'HBC1'), ('HBC1', 'B2'),
         ('C1', 'HCA1'), ('HCA1', 'A2'), ('A1', 'HCA1'), ('HCA1', 'C2')],
        [('A2', 'HAB2'), ('HAB2', 'B3'), ('B2', 'HAB2'), ('HAB2', 'A3'),
         ('B2', 'HBC2'), ('HBC2', 'C3'), ('C2', 'HBC2'), ('HBC2', 'B3'),
         ('C2', 'HCA2'), ('HCA2', 'A3'), ('A2', 'HCA2'), ('HCA2', 'C3')],
        [('A3', 'HAB3'), ('HAB3', 'B4'), ('B3', 'HAB3'), ('HAB3', 'A4'),
         ('B3', 'HBC3'), ('HBC3', 'C4'), ('C3', 'HBC3'), ('HBC3', 'B4'),
         ('C3', 'HCA3'), ('HCA3', 'A4'), ('A3', 'HCA3'), ('HCA3', 'C4')],
    ]
    for level_braces in xbrace_config:
        for n1, n2 in level_braces:
            members.append({'name': f'XBr_{n1}-{n2}', 'node1': n1, 'node2': n2, 'type': 'x_brace'})
    
    fixed_nodes = ['A1', 'B1', 'C1']
    top_nodes = ['A4', 'B4', 'C4']
    
    return nodes, members, fixed_nodes, top_nodes


# =============================================================================
# GUI APPLICATION CLASS
# =============================================================================
class JacketAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Jacket Structural Analysis - Customizable Geometry v8")
        self.root.geometry("1500x950")
        
        # Geometry data storage
        self.nodes_data = {}  # {node_name: [x, y, z]}
        self.members_data = []  # [{name, node1, node2, type}]
        self.fixed_nodes = []
        self.top_nodes = []
        
        # Analysis parameters
        self.params = {}
        self.analysis_results = None
        
        self.create_widgets()
        self.load_default_geometry()
        self.load_default_params()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.tab_geometry = ttk.Frame(self.notebook)
        self.tab_members = ttk.Frame(self.notebook)
        self.tab_material = ttk.Frame(self.notebook)
        self.tab_wave = ttk.Frame(self.notebook)
        self.tab_loads = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)
        self.tab_results = ttk.Frame(self.notebook)
        self.tab_info = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_geometry, text="1. Node Geometry")
        self.notebook.add(self.tab_members, text="2. Members")
        self.notebook.add(self.tab_material, text="3. Material & Sections")
        self.notebook.add(self.tab_wave, text="4. Wave Parameters")
        self.notebook.add(self.tab_loads, text="5. Loads")
        self.notebook.add(self.tab_analysis, text="6. Run Analysis")
        self.notebook.add(self.tab_results, text="7. Results")
        self.notebook.add(self.tab_info, text="8. Info & Assumptions")

        self._create_geometry_tab()
        self._create_members_tab()
        self._create_material_tab()
        self._create_wave_tab()
        self._create_loads_tab()
        self._create_analysis_tab()
        self._create_results_tab()
        self._create_info_tab()

    # =========================================================================
    # GEOMETRY TAB
    # =========================================================================
    def _create_geometry_tab(self):
        main_frame = ttk.Frame(self.tab_geometry)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left frame - Node input
        left_frame = ttk.LabelFrame(main_frame, text="Node Definition", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # COORDINATE SYSTEM WARNING - VERY IMPORTANT
        coord_frame = ttk.LabelFrame(left_frame, text="⚠️ COORDINATE SYSTEM (IMPORTANT!)", padding=5)
        coord_frame.pack(fill=tk.X, pady=5)
        
        coord_text = """GLOBAL COORDINATE SYSTEM:
  • X-axis: Points EAST (+X = East, -X = West)
  • Y-axis: Points NORTH (+Y = North, -Y = South)  
  • Z-axis: Points UP (+Z = Up, -Z = Down)
  • Z = 0 is at Mean Water Level (MWL)

When inputting node coordinates, ensure your structure 
orientation matches: +Y = NORTH direction"""
        
        coord_label = ttk.Label(coord_frame, text=coord_text, justify='left', 
                               font=('Consolas', 9), foreground='darkblue')
        coord_label.pack(anchor='w', padx=5)

        ttk.Separator(left_frame, orient='horizontal').pack(fill='x', pady=5)

        # Instructions
        ttk.Label(left_frame, text="Node Naming Convention:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
        ttk.Label(left_frame, text="• Main Legs: A1, A2, A3... B1, B2, B3... (letter=leg, number=level from bottom)").pack(anchor='w')
        ttk.Label(left_frame, text="• Hinge nodes: HAB1, HAB2... (H=hinge, AB=face, number=level)").pack(anchor='w')
        ttk.Label(left_frame, text="• Z=0 is at Mean Water Level (MWL), negative below").pack(anchor='w')
        ttk.Separator(left_frame, orient='horizontal').pack(fill='x', pady=10)

        # Node entry frame
        entry_frame = ttk.Frame(left_frame)
        entry_frame.pack(fill=tk.X, pady=5)

        ttk.Label(entry_frame, text="Name:").grid(row=0, column=0, padx=2)
        self.node_name_entry = ttk.Entry(entry_frame, width=10)
        self.node_name_entry.grid(row=0, column=1, padx=2)

        ttk.Label(entry_frame, text="X:").grid(row=0, column=2, padx=2)
        self.node_x_entry = ttk.Entry(entry_frame, width=10)
        self.node_x_entry.grid(row=0, column=3, padx=2)

        ttk.Label(entry_frame, text="Y:").grid(row=0, column=4, padx=2)
        self.node_y_entry = ttk.Entry(entry_frame, width=10)
        self.node_y_entry.grid(row=0, column=5, padx=2)

        ttk.Label(entry_frame, text="Z:").grid(row=0, column=6, padx=2)
        self.node_z_entry = ttk.Entry(entry_frame, width=10)
        self.node_z_entry.grid(row=0, column=7, padx=2)

        ttk.Button(entry_frame, text="Add Node", command=self.add_node).grid(row=0, column=8, padx=5)
        ttk.Button(entry_frame, text="Delete Selected", command=self.delete_node).grid(row=0, column=9, padx=5)

        # Node list with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ('name', 'x', 'y', 'z', 'fixed', 'top')
        self.node_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        for col in columns:
            self.node_tree.heading(col, text=col.upper())
            self.node_tree.column(col, width=80)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.node_tree.yview)
        self.node_tree.configure(yscrollcommand=scrollbar.set)
        self.node_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Fixed/Top node controls
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Toggle Fixed (Support)", command=self.toggle_fixed).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Toggle Top (Interface)", command=self.toggle_top).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Default Geometry", command=self.load_default_geometry).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear All", command=self.clear_geometry).pack(side=tk.LEFT, padx=5)

        # Right frame - 3D preview
        right_frame = ttk.LabelFrame(main_frame, text="3D Preview", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        ttk.Button(right_frame, text="Update Preview", command=self.update_3d_preview).pack(pady=5)
        
        # Placeholder for matplotlib canvas
        self.preview_frame = ttk.Frame(right_frame)
        self.preview_frame.pack(fill=tk.BOTH, expand=True)

    def add_node(self):
        name = self.node_name_entry.get().strip().upper()
        try:
            x = float(self.node_x_entry.get())
            y = float(self.node_y_entry.get())
            z = float(self.node_z_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid coordinate values")
            return
        
        if not name:
            messagebox.showerror("Error", "Node name cannot be empty")
            return
        
        self.nodes_data[name] = [x, y, z]
        self.refresh_node_list()
        
        # Clear entries
        self.node_name_entry.delete(0, tk.END)
        self.node_x_entry.delete(0, tk.END)
        self.node_y_entry.delete(0, tk.END)
        self.node_z_entry.delete(0, tk.END)

    def delete_node(self):
        selection = self.node_tree.selection()
        if not selection:
            return
        for item in selection:
            name = self.node_tree.item(item)['values'][0]
            if name in self.nodes_data:
                del self.nodes_data[name]
            if name in self.fixed_nodes:
                self.fixed_nodes.remove(name)
            if name in self.top_nodes:
                self.top_nodes.remove(name)
        self.refresh_node_list()

    def toggle_fixed(self):
        selection = self.node_tree.selection()
        for item in selection:
            name = self.node_tree.item(item)['values'][0]
            if name in self.fixed_nodes:
                self.fixed_nodes.remove(name)
            else:
                self.fixed_nodes.append(name)
        self.refresh_node_list()

    def toggle_top(self):
        selection = self.node_tree.selection()
        for item in selection:
            name = self.node_tree.item(item)['values'][0]
            if name in self.top_nodes:
                self.top_nodes.remove(name)
            else:
                self.top_nodes.append(name)
        self.refresh_node_list()

    def refresh_node_list(self):
        self.node_tree.delete(*self.node_tree.get_children())
        for name, coords in sorted(self.nodes_data.items()):
            fixed = "✓" if name in self.fixed_nodes else ""
            top = "✓" if name in self.top_nodes else ""
            self.node_tree.insert('', tk.END, values=(name, f"{coords[0]:.3f}", 
                                 f"{coords[1]:.3f}", f"{coords[2]:.3f}", fixed, top))

    def clear_geometry(self):
        if messagebox.askyesno("Confirm", "Clear all geometry data?"):
            self.nodes_data = {}
            self.members_data = []
            self.fixed_nodes = []
            self.top_nodes = []
            self.refresh_node_list()
            self.refresh_member_list()

    def load_default_geometry(self):
        """Load the default 3-leg jacket geometry."""
        nodes, members, fixed, top = create_default_3leg_jacket(47.0)
        self.nodes_data = {k: list(v) for k, v in nodes.items()}
        self.members_data = members
        self.fixed_nodes = fixed
        self.top_nodes = top
        self.refresh_node_list()
        self.refresh_member_list()

    def update_3d_preview(self):
        """Update the 3D preview plot - opens in a new window."""
        if not self.nodes_data:
            messagebox.showwarning("Warning", "No nodes defined!")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get coordinate ranges
        coords_list = list(self.nodes_data.values())
        x_min, x_max = min(c[0] for c in coords_list), max(c[0] for c in coords_list)
        y_min, y_max = min(c[1] for c in coords_list), max(c[1] for c in coords_list)
        z_min, z_max = min(c[2] for c in coords_list), max(c[2] for c in coords_list)
        
        # Plot nodes
        for name, coords in self.nodes_data.items():
            if name in self.fixed_nodes:
                color, marker, size = 'red', '^', 150
            elif name in self.top_nodes:
                color, marker, size = 'blue', 's', 120
            else:
                color, marker, size = 'gray', 'o', 60
            ax.scatter(coords[0], coords[1], coords[2], c=color, marker=marker, s=size, 
                      edgecolors='black', linewidths=1)
            ax.text(coords[0], coords[1], coords[2], f'  {name}', fontsize=8)
        
        # Plot members
        for member in self.members_data:
            if member['node1'] in self.nodes_data and member['node2'] in self.nodes_data:
                c1 = self.nodes_data[member['node1']]
                c2 = self.nodes_data[member['node2']]
                if member['type'] == 'leg':
                    color, lw = 'darkblue', 4
                elif 'h_brace' in member['type']:
                    color, lw = 'green', 2.5
                elif 'x_brace' in member['type']:
                    color, lw = 'orange', 2
                else:
                    color, lw = 'gray', 1.5
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], color=color, lw=lw, alpha=0.8)
        
        # Plot water surface at z=0
        x_range = [x_min - 5, x_max + 5]
        y_range = [y_min - 5, y_max + 5]
        X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], 10),
                          np.linspace(y_range[0], y_range[1], 10))
        ax.plot_surface(X, Y, np.zeros_like(X), alpha=0.15, color='cyan')
        ax.text((x_range[0]+x_range[1])/2, (y_range[0]+y_range[1])/2, 0.5, 'MWL (z=0)', 
               fontsize=10, color='cyan', ha='center')
        
        # =====================================================================
        # DRAW COMPASS / NORTH ARROW at corner
        # =====================================================================
        arrow_base_x = x_min - 3
        arrow_base_y = y_min - 3
        arrow_base_z = z_max + 5
        arrow_length = 8
        
        # North arrow (Y direction)
        ax.quiver(arrow_base_x, arrow_base_y, arrow_base_z, 
                 0, arrow_length, 0, 
                 color='darkgreen', arrow_length_ratio=0.15, linewidth=3)
        ax.text(arrow_base_x, arrow_base_y + arrow_length + 1, arrow_base_z, 
               'N\n(+Y)', fontsize=12, fontweight='bold', color='darkgreen', ha='center')
        
        # East arrow (X direction)
        ax.quiver(arrow_base_x, arrow_base_y, arrow_base_z, 
                 arrow_length * 0.7, 0, 0, 
                 color='darkred', arrow_length_ratio=0.15, linewidth=2)
        ax.text(arrow_base_x + arrow_length * 0.7 + 1, arrow_base_y, arrow_base_z, 
               'E (+X)', fontsize=10, color='darkred', ha='left')
        
        # Draw coordinate system box at origin
        ax.scatter([0], [0], [0], c='black', s=100, marker='+', linewidths=2)
        ax.text(0, 0, 2, 'Origin\n(0,0,0)', fontsize=8, ha='center')
        
        ax.set_xlabel('X [m] → EAST', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y [m] → NORTH', fontsize=11, fontweight='bold')
        ax.set_zlabel('Z [m] → UP', fontsize=11, fontweight='bold')
        ax.set_title(f'Structure Preview\n{len(self.nodes_data)} nodes, {len(self.members_data)} members\n'
                    f'COORDINATE SYSTEM: X=East, Y=North, Z=Up', fontsize=12)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Fixed (support)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Top (interface)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Other nodes'),
            Line2D([0], [0], color='darkblue', linewidth=3, label='Legs'),
            Line2D([0], [0], color='green', linewidth=2, label='H-Braces'),
            Line2D([0], [0], color='orange', linewidth=2, label='X-Braces'),
            Line2D([0], [0], color='darkgreen', linewidth=3, label='North (+Y)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # MEMBERS TAB
    # =========================================================================
    def _create_members_tab(self):
        main_frame = ttk.Frame(self.tab_members)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Member entry frame
        entry_frame = ttk.LabelFrame(main_frame, text="Add Member", padding=10)
        entry_frame.pack(fill=tk.X, pady=5)

        ttk.Label(entry_frame, text="Name:").grid(row=0, column=0, padx=2)
        self.member_name_entry = ttk.Entry(entry_frame, width=15)
        self.member_name_entry.grid(row=0, column=1, padx=2)

        ttk.Label(entry_frame, text="Node 1:").grid(row=0, column=2, padx=2)
        self.member_n1_entry = ttk.Entry(entry_frame, width=10)
        self.member_n1_entry.grid(row=0, column=3, padx=2)

        ttk.Label(entry_frame, text="Node 2:").grid(row=0, column=4, padx=2)
        self.member_n2_entry = ttk.Entry(entry_frame, width=10)
        self.member_n2_entry.grid(row=0, column=5, padx=2)

        ttk.Label(entry_frame, text="Type:").grid(row=0, column=6, padx=2)
        self.member_type_var = tk.StringVar(value='brace')
        self.member_type_combo = ttk.Combobox(entry_frame, textvariable=self.member_type_var, 
                                              values=['leg', 'h_brace', 'x_brace', 'brace'], width=10)
        self.member_type_combo.grid(row=0, column=7, padx=2)

        ttk.Button(entry_frame, text="Add Member", command=self.add_member).grid(row=0, column=8, padx=5)
        ttk.Button(entry_frame, text="Delete Selected", command=self.delete_member).grid(row=0, column=9, padx=5)

        # Member list
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ('name', 'node1', 'node2', 'type')
        self.member_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=20)
        for col in columns:
            self.member_tree.heading(col, text=col.upper())
            self.member_tree.column(col, width=150)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.member_tree.yview)
        self.member_tree.configure(yscrollcommand=scrollbar.set)
        self.member_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Auto-generate buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Label(btn_frame, text="Auto-generate:").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Legs (A1→A2→A3...)", command=self.auto_generate_legs).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Horizontal Braces", command=self.auto_generate_h_braces).pack(side=tk.LEFT, padx=5)

    def add_member(self):
        name = self.member_name_entry.get().strip()
        n1 = self.member_n1_entry.get().strip().upper()
        n2 = self.member_n2_entry.get().strip().upper()
        mtype = self.member_type_var.get()
        
        if not all([name, n1, n2]):
            messagebox.showerror("Error", "All fields are required")
            return
        
        if n1 not in self.nodes_data or n2 not in self.nodes_data:
            messagebox.showerror("Error", f"Nodes {n1} or {n2} not defined")
            return
        
        self.members_data.append({'name': name, 'node1': n1, 'node2': n2, 'type': mtype})
        self.refresh_member_list()
        
        self.member_name_entry.delete(0, tk.END)
        self.member_n1_entry.delete(0, tk.END)
        self.member_n2_entry.delete(0, tk.END)

    def delete_member(self):
        selection = self.member_tree.selection()
        for item in selection:
            name = self.member_tree.item(item)['values'][0]
            self.members_data = [m for m in self.members_data if m['name'] != name]
        self.refresh_member_list()

    def refresh_member_list(self):
        self.member_tree.delete(*self.member_tree.get_children())
        for m in self.members_data:
            self.member_tree.insert('', tk.END, values=(m['name'], m['node1'], m['node2'], m['type']))

    def auto_generate_legs(self):
        """Auto-generate leg members based on node naming pattern."""
        # Find all leg nodes (letter + number pattern)
        import re
        leg_nodes = {}
        for name in self.nodes_data.keys():
            match = re.match(r'^([A-Z])(\d+)$', name)
            if match:
                leg = match.group(1)
                level = int(match.group(2))
                if leg not in leg_nodes:
                    leg_nodes[leg] = []
                leg_nodes[leg].append((level, name))
        
        # Sort by level and create members
        for leg, nodes in leg_nodes.items():
            nodes.sort()
            for i in range(len(nodes) - 1):
                n1 = nodes[i][1]
                n2 = nodes[i+1][1]
                name = f"Leg_{n1}-{n2}"
                if not any(m['name'] == name for m in self.members_data):
                    self.members_data.append({'name': name, 'node1': n1, 'node2': n2, 'type': 'leg'})
        
        self.refresh_member_list()
        messagebox.showinfo("Done", "Leg members generated!")

    def auto_generate_h_braces(self):
        """Auto-generate horizontal braces at each level."""
        import re
        levels = {}
        for name in self.nodes_data.keys():
            match = re.match(r'^([A-Z])(\d+)$', name)
            if match:
                level = int(match.group(2))
                if level not in levels:
                    levels[level] = []
                levels[level].append(name)
        
        for level, nodes in levels.items():
            nodes.sort()
            for i in range(len(nodes)):
                n1 = nodes[i]
                n2 = nodes[(i + 1) % len(nodes)]
                name = f"HBrace_{n1}-{n2}"
                if not any(m['name'] == name for m in self.members_data):
                    self.members_data.append({'name': name, 'node1': n1, 'node2': n2, 'type': 'h_brace'})
        
        self.refresh_member_list()
        messagebox.showinfo("Done", "Horizontal braces generated!")

    # =========================================================================
    # MATERIAL TAB
    # =========================================================================
    def _create_material_tab(self):
        frame = ttk.Frame(self.tab_material, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Material properties
        mat_frame = ttk.LabelFrame(frame, text="Material Properties", padding=10)
        mat_frame.pack(fill=tk.X, pady=5)

        row = 0
        ttk.Label(mat_frame, text="Young's Modulus (E):").grid(row=row, column=0, sticky='e', padx=5)
        self.entry_E = ttk.Entry(mat_frame, width=12)
        self.entry_E.grid(row=row, column=1, padx=5)
        ttk.Label(mat_frame, text="N/mm² (MPa)").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(mat_frame, text="Poisson's Ratio (ν):").grid(row=row, column=0, sticky='e', padx=5)
        self.entry_nu = ttk.Entry(mat_frame, width=12)
        self.entry_nu.grid(row=row, column=1, padx=5)

        row += 1
        ttk.Label(mat_frame, text="Yield Strength (fy):").grid(row=row, column=0, sticky='e', padx=5)
        self.entry_fy = ttk.Entry(mat_frame, width=12)
        self.entry_fy.grid(row=row, column=1, padx=5)
        ttk.Label(mat_frame, text="N/mm² (MPa) - CUSTOMIZABLE").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(mat_frame, text="Steel Density:").grid(row=row, column=0, sticky='e', padx=5)
        self.entry_rho_steel = ttk.Entry(mat_frame, width=12)
        self.entry_rho_steel.grid(row=row, column=1, padx=5)
        ttk.Label(mat_frame, text="kg/m³").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(mat_frame, text="Seawater Density:").grid(row=row, column=0, sticky='e', padx=5)
        self.entry_rho_water = ttk.Entry(mat_frame, width=12)
        self.entry_rho_water.grid(row=row, column=1, padx=5)
        ttk.Label(mat_frame, text="kg/m³").grid(row=row, column=2, sticky='w')

        # Section dimensions
        sec_frame = ttk.LabelFrame(frame, text="Tubular Section Dimensions", padding=10)
        sec_frame.pack(fill=tk.X, pady=10)

        row = 0
        ttk.Label(sec_frame, text="LEG SECTION:", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky='w')
        
        row += 1
        ttk.Label(sec_frame, text="Outer Diameter (D_leg):").grid(row=row, column=0, sticky='e', padx=5)
        self.entry_D_leg = ttk.Entry(sec_frame, width=12)
        self.entry_D_leg.grid(row=row, column=1, padx=5)
        ttk.Label(sec_frame, text="mm").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(sec_frame, text="Wall Thickness (t_leg):").grid(row=row, column=0, sticky='e', padx=5)
        self.entry_t_leg = ttk.Entry(sec_frame, width=12)
        self.entry_t_leg.grid(row=row, column=1, padx=5)
        ttk.Label(sec_frame, text="mm").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(sec_frame, text="BRACE SECTION:", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky='w', pady=(10,0))

        row += 1
        ttk.Label(sec_frame, text="Outer Diameter (D_brace):").grid(row=row, column=0, sticky='e', padx=5)
        self.entry_D_brace = ttk.Entry(sec_frame, width=12)
        self.entry_D_brace.grid(row=row, column=1, padx=5)
        ttk.Label(sec_frame, text="mm").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(sec_frame, text="Wall Thickness (t_brace):").grid(row=row, column=0, sticky='e', padx=5)
        self.entry_t_brace = ttk.Entry(sec_frame, width=12)
        self.entry_t_brace.grid(row=row, column=1, padx=5)
        ttk.Label(sec_frame, text="mm").grid(row=row, column=2, sticky='w')

    # =========================================================================
    # WAVE TAB
    # =========================================================================
    def _create_wave_tab(self):
        frame = ttk.Frame(self.tab_wave, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Raschii status
        if RASCHII_AVAILABLE:
            ttk.Label(frame, text="✓ Raschii Library: AVAILABLE", foreground='green').pack(anchor='w')
        else:
            ttk.Label(frame, text="⚠ Raschii Library: NOT AVAILABLE (using Airy)", foreground='orange').pack(anchor='w')

        wave_frame = ttk.LabelFrame(frame, text="Wave Parameters", padding=10)
        wave_frame.pack(fill=tk.X, pady=10)

        row = 0
        ttk.Label(wave_frame, text="Wave Height (H):").grid(row=row, column=0, sticky='e', padx=5, pady=2)
        self.entry_H = ttk.Entry(wave_frame, width=12)
        self.entry_H.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(wave_frame, text="m").grid(row=row, column=2, sticky='w')
        
        row += 1
        ttk.Label(wave_frame, text="Wave Period (T):").grid(row=row, column=0, sticky='e', padx=5, pady=2)
        self.entry_T = ttk.Entry(wave_frame, width=12)
        self.entry_T.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(wave_frame, text="s").grid(row=row, column=2, sticky='w')
        
        row += 1
        ttk.Label(wave_frame, text="Water Depth (d):").grid(row=row, column=0, sticky='e', padx=5, pady=2)
        self.entry_d = ttk.Entry(wave_frame, width=12)
        self.entry_d.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(wave_frame, text="m").grid(row=row, column=2, sticky='w')
        
        row += 1
        ttk.Label(wave_frame, text="Current Velocity (U_c):").grid(row=row, column=0, sticky='e', padx=5, pady=2)
        self.entry_Uc = ttk.Entry(wave_frame, width=12)
        self.entry_Uc.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(wave_frame, text="m/s").grid(row=row, column=2, sticky='w')
        
        row += 1
        ttk.Separator(wave_frame, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew', pady=5)
        
        # DIRECTIONS - Important!
        row += 1
        ttk.Label(wave_frame, text="DIRECTIONS (from North, clockwise):", 
                 font=('TkDefaultFont', 9, 'bold')).grid(row=row, column=0, columnspan=3, sticky='w', pady=5)
        
        row += 1
        ttk.Label(wave_frame, text="Wave Direction:").grid(row=row, column=0, sticky='e', padx=5, pady=2)
        self.entry_wave_dir = ttk.Entry(wave_frame, width=12)
        self.entry_wave_dir.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(wave_frame, text="° from North").grid(row=row, column=2, sticky='w')
        
        row += 1
        ttk.Label(wave_frame, text="Current Direction:").grid(row=row, column=0, sticky='e', padx=5, pady=2)
        self.entry_current_dir = ttk.Entry(wave_frame, width=12)
        self.entry_current_dir.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(wave_frame, text="° from North").grid(row=row, column=2, sticky='w')
        
        row += 1
        ttk.Label(wave_frame, text="(Same direction = collinear, max force)", 
                 foreground='gray').grid(row=row, column=0, columnspan=3, sticky='w', padx=5)

        # Wave model with AUTO option
        row += 1
        ttk.Separator(wave_frame, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew', pady=5)
        
        row += 1
        ttk.Label(wave_frame, text="Wave Model:").grid(row=row, column=0, sticky='e', padx=5)
        self.wave_model_var = tk.StringVar(value='auto')
        models = ['auto', 'Fenton', 'Stokes', 'Airy'] if RASCHII_AVAILABLE else ['Airy']
        ttk.Combobox(wave_frame, textvariable=self.wave_model_var, values=models, width=12).grid(row=row, column=1, padx=5)
        ttk.Label(wave_frame, text="('auto' selects best model)").grid(row=row, column=2, sticky='w')
        
        row += 1
        ttk.Label(wave_frame, text="N (harmonics):").grid(row=row, column=0, sticky='e', padx=5)
        self.entry_N_harm = ttk.Entry(wave_frame, width=12)
        self.entry_N_harm.insert(0, "10")
        self.entry_N_harm.grid(row=row, column=1, padx=5)
        ttk.Label(wave_frame, text="(Fenton:5-20, Stokes:max 5)").grid(row=row, column=2, sticky='w')

        # Morison coefficients
        mor_frame = ttk.LabelFrame(frame, text="Morison Coefficients", padding=10)
        mor_frame.pack(fill=tk.X, pady=10)

        ttk.Label(mor_frame, text="Drag Coefficient (Cd):").grid(row=0, column=0, sticky='e', padx=5)
        self.entry_Cd = ttk.Entry(mor_frame, width=12)
        self.entry_Cd.grid(row=0, column=1, padx=5)
        ttk.Label(mor_frame, text="(typical: 0.6-1.2)").grid(row=0, column=2, sticky='w')

        ttk.Label(mor_frame, text="Inertia Coefficient (Cm):").grid(row=1, column=0, sticky='e', padx=5)
        self.entry_Cm = ttk.Entry(mor_frame, width=12)
        self.entry_Cm.grid(row=1, column=1, padx=5)
        ttk.Label(mor_frame, text="(typical: 1.5-2.0)").grid(row=1, column=2, sticky='w')
        
        # Phase scan option
        phase_frame = ttk.LabelFrame(frame, text="Phase Analysis", padding=10)
        phase_frame.pack(fill=tk.X, pady=10)
        
        self.do_phase_scan = tk.BooleanVar(value=True)
        ttk.Checkbutton(phase_frame, text="Find critical phase (scan through wave period)", 
                       variable=self.do_phase_scan).pack(anchor='w')
        ttk.Label(phase_frame, text="Note: Drag ∝ |u|·u (in phase with velocity), "
                 "Inertia ∝ du/dt (90° phase lead)", foreground='gray').pack(anchor='w')

    # =========================================================================
    # LOADS TAB
    # =========================================================================
    def _create_loads_tab(self):
        frame = ttk.Frame(self.tab_loads, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Interface loads
        int_frame = ttk.LabelFrame(frame, text="Topside Interface Loads", padding=10)
        int_frame.pack(fill=tk.X, pady=5)

        loads = [
            ("Axial Force (compression):", "entry_F_axial", "kN"),
            ("Horizontal Shear:", "entry_F_shear", "kN"),
            ("Overturning Moment:", "entry_M_moment", "kNm"),
            ("Torsional Moment:", "entry_M_torsion", "kNm"),
        ]
        
        for i, (label, attr, unit) in enumerate(loads):
            ttk.Label(int_frame, text=label).grid(row=i, column=0, sticky='e', padx=5, pady=2)
            entry = ttk.Entry(int_frame, width=12)
            entry.grid(row=i, column=1, padx=5, pady=2)
            ttk.Label(int_frame, text=unit).grid(row=i, column=2, sticky='w')
            setattr(self, attr, entry)

        # Self-weight options
        sw_frame = ttk.LabelFrame(frame, text="Self-Weight", padding=10)
        sw_frame.pack(fill=tk.X, pady=10)

        self.self_weight_mode = tk.StringVar(value='calculated')
        ttk.Radiobutton(sw_frame, text="Calculate from geometry (ρ × A × L)", 
                       variable=self.self_weight_mode, value='calculated').pack(anchor='w')
        
        custom_frame = ttk.Frame(sw_frame)
        custom_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(custom_frame, text="Custom self-weight:", 
                       variable=self.self_weight_mode, value='custom').pack(side=tk.LEFT)
        self.entry_custom_sw = ttk.Entry(custom_frame, width=12)
        self.entry_custom_sw.pack(side=tk.LEFT, padx=5)
        ttk.Label(custom_frame, text="tonnes (distributed to all nodes)").pack(side=tk.LEFT)
        
        ttk.Radiobutton(sw_frame, text="No self-weight", 
                       variable=self.self_weight_mode, value='none').pack(anchor='w')

    # =========================================================================
    # ANALYSIS TAB
    # =========================================================================
    def _create_analysis_tab(self):
        frame = ttk.Frame(self.tab_analysis, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Analysis time
        time_frame = ttk.LabelFrame(frame, text="Analysis Time", padding=10)
        time_frame.pack(fill=tk.X, pady=5)

        ttk.Label(time_frame, text="Time (t):").pack(side=tk.LEFT, padx=5)
        self.entry_t = ttk.Entry(time_frame, width=12)
        self.entry_t.insert(0, "0.0")
        self.entry_t.pack(side=tk.LEFT, padx=5)
        ttk.Label(time_frame, text="seconds").pack(side=tk.LEFT)

        # Run button
        ttk.Button(frame, text="RUN ANALYSIS", command=self.run_analysis, 
                  style='Accent.TButton').pack(pady=20)

        # Log
        ttk.Label(frame, text="Analysis Log:").pack(anchor='w')
        self.log_text = tk.Text(frame, height=25, width=100)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # =========================================================================
    # RESULTS TAB
    # =========================================================================
    def _create_results_tab(self):
        frame = ttk.Frame(self.tab_results, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Show Summary", command=self.show_summary).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Plot Structure", command=self.plot_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5)

        self.results_text = tk.Text(frame, height=30, width=100)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    # =========================================================================
    # INFO & ASSUMPTIONS TAB
    # =========================================================================
    def _create_info_tab(self):
        frame = ttk.Frame(self.tab_info, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        info_text = tk.Text(frame, height=50, width=120, wrap=tk.WORD, font=('Consolas', 9))
        info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        info_text.config(yscrollcommand=scrollbar.set)

        assumptions_text = """
================================================================================
              CODE ASSUMPTIONS, LIMITATIONS AND IMPORTANT NOTES
================================================================================

1. COORDINATE SYSTEM
   =================
   • X-axis: Points EAST  (+X = East, -X = West)
   • Y-axis: Points NORTH (+Y = North, -Y = South)
   • Z-axis: Points UP    (+Z = Up, -Z = Down)
   • Z = 0 is at Mean Water Level (MWL)
   
   When inputting node coordinates, ensure your structure orientation matches
   this convention. The North arrow in 3D plots indicates the +Y direction.

2. STRUCTURAL ASSUMPTIONS
   ======================
   • Cross-section type: CIRCULAR TUBULAR (pipe) sections ONLY
   • Section theory: THIN-WALL ASSUMPTION
     - Valid when D/t > 10 (diameter/thickness ratio)
     - Current defaults: Leg D/t = 2000/75 = 26.7 ✓
     -                   Brace D/t = 800/30 = 26.7 ✓
   • Beam theory: TIMOSHENKO beam (includes shear deformation)
   • Material behavior: LINEAR ELASTIC (no plasticity, no buckling)
   • Analysis type: STATIC or QUASI-STATIC (no dynamic/inertial effects)
   • Support conditions: FIXED at mudline (all 6 DOF restrained)
   • Connections: RIGID (no joint flexibility)

3. THIN-WALL CROSS-SECTION FORMULAS
   =================================
   The code uses thin-walled tube formulas:
   
   • Cross-sectional Area:
       A = π/4 × (D² - d²) ≈ π × D × t   (for D >> t)
   
   • Second Moment of Area:
       I = π/64 × (D⁴ - d⁴)
   
   • Polar Moment (Torsion):
       J = π/32 × (D⁴ - d⁴) = 2 × I
   
   • Shear Area (Timoshenko):
       A_shear ≈ 0.5 × A
   
   VALIDITY: D/t ratio MUST be > 10 for thin-wall assumption to be accurate.
   If your sections have D/t < 10, results may be inaccurate!

4. MORISON EQUATION
   =================
   F = F_drag + F_inertia
   
   F_drag   = ½ × ρ × Cd × D × |u_perp| × u_perp    (per unit length)
   F_inertia = ρ × Cm × A × (du/dt)_perp            (per unit length)
   
   Where:
   • ρ = seawater density (default: 1025 kg/m³)
   • Cd = drag coefficient (typical: 0.6 - 1.2)
   • Cm = inertia coefficient (typical: 1.5 - 2.0)
   • D = member diameter
   • u_perp = velocity component perpendicular to member axis
   • (du/dt)_perp = acceleration component perpendicular to member axis
   
   PHASE RELATIONSHIP:
   • Drag force is IN PHASE with velocity (∝ |u|×u)
   • Inertia force LEADS velocity by 90° (∝ du/dt)
   • Maximum total force occurs at a specific phase angle
   • Use "Phase Scan" option to find the critical (maximum) phase

   MORISON ASSUMPTIONS:
   • Slender member assumption: D/L < 0.2 (member diameter << wavelength)
   • Only perpendicular velocity/acceleration components used
   • No end effects
   • No marine growth (add to diameter if needed)
   • No interference between members
   • Cd and Cm are constant along member length
   • Gauss-Legendre integration along member (default: 15 points)

5. WAVE THEORY LIMITATIONS (RASCHII LIBRARY)
   ==========================================
   
   AVAILABLE WAVE MODELS:
   ┌─────────────┬──────────────────┬─────────────────────────────────────┐
   │ Model       │ Order / N        │ Best For                            │
   ├─────────────┼──────────────────┼─────────────────────────────────────┤
   │ Airy        │ 1st order        │ Small amplitude, H/L < 0.01         │
   │ Stokes      │ Max N = 5        │ Moderate waves, H/L = 0.01 - 0.06   │
   │ Fenton      │ Typically 5-20   │ Steep waves, H/L > 0.06             │
   │ Auto        │ Selected auto.   │ Let the code choose best model      │
   └─────────────┴──────────────────┴─────────────────────────────────────┘
   
   AUTO SELECTION CRITERIA:
   • H/L < 0.01  → Airy (1st order)
   • H/L < 0.03  → Stokes 3rd order
   • H/L < 0.06  → Stokes 5th order
   • H/L ≥ 0.06  → Fenton (N = 10-20 based on steepness)
   
   BREAKING WAVE LIMITS:
   • Deep water limit: H/L < 0.142 (Miche criterion)
   • Shallow water limit: H/d < 0.78 (solitary wave)
   • If wave is too steep, raschii will fail to converge!
   
   RECOMMENDED PARAMETER RANGES:
   • Wave height H: 0.5m - 30m (extreme storms)
   • Wave period T: 3s - 20s (typical ocean waves)
   • Water depth d: 10m - 300m (typical offshore)
   • Current U_c: 0 - 3 m/s
   • Wave steepness: H/L < 0.10 for numerical stability

   ERROR CONDITIONS:
   • "Wave too steep" → Reduce H or increase T
   • "Convergence failed" → Try different wave model or parameters
   • "Invalid input" → Check H > 0, T > 0, d > 0

6. WAVE AND CURRENT DIRECTIONS
   ============================
   • Both directions are specified as degrees FROM NORTH, CLOCKWISE
   • 0° = North, 90° = East, 180° = South, 270° = West
   
   Example:
   • Wave direction = 38° means waves traveling toward 38° from North (NNE)
   • Current direction can be different from wave direction
   • Collinear (same direction) gives maximum combined force
   
   The code properly handles non-collinear wave and current by:
   • Computing wave kinematics along wave direction
   • Adding current velocity along current direction
   • Combining both for Morison force calculation

7. INTERFACE LOADS
   ================
   Interface loads are applied at TOP NODES (interface level):
   
   • F_axial:   Vertical compression (distributed equally to all top nodes)
   • F_shear:   Horizontal shear (applied in wave direction)
   • M_moment:  Overturning moment about horizontal axis (My)
   • M_torsion: Torsional moment about vertical axis (Mx)
   
   All interface loads are distributed equally among the top nodes.
   
   NOTE: Interface moments (M_moment, M_torsion) ARE implemented!
   Set them to non-zero values if your topside applies moments.

8. SELF-WEIGHT OPTIONS
   ====================
   Three modes available:
   
   a) CALCULATED: Self-weight = ρ_steel × A × L for each member
      - Uses steel density from material properties
      - Applied as distributed load converted to nodal forces
      
   b) CUSTOM: User specifies total self-weight in tonnes
      - Distributed equally to all nodes
      - Use this if you know the actual jacket weight
      
   c) NONE: Self-weight excluded from analysis

9. STRESS CALCULATION
   ===================
   • Von Mises stress evaluated at 8 points around circumference
   • Maximum von Mises stress is used for utilization check
   
   Utilization = σ_vm / f_y
   
   Where:
   • σ_vm = von Mises equivalent stress (max of 8 points)
   • f_y = yield strength (customizable, default 355 MPa for S355 steel)
   
   WARNING: This code performs ELASTIC STRESS CHECK ONLY!
   • NO buckling check (Euler, local, interaction)
   • NO fatigue analysis
   • NO safety factors applied
   
   For actual design, additional checks per design codes are required!

10. FEM FORMULATION
    ================
    • 3D beam elements with 12 DOF (6 per node)
    • DOFs per node: u, v, w (translations), θx, θy, θz (rotations)
    • Timoshenko beam formulation with shear correction
    • Global stiffness matrix assembly by direct stiffness method
    • Linear static solver (K × U = F)

11. UNITS CONVENTION
    =================
    INPUT:
    • Lengths: m (structure geometry), mm (section dimensions)
    • Forces: kN
    • Moments: kNm
    • Stresses: MPa (N/mm²)
    • Density: kg/m³
    • Self-weight: tonnes (1000 kg)
    
    INTERNAL CALCULATIONS:
    • Lengths: mm
    • Forces: N
    • Moments: N·mm
    • Stresses: N/mm² = MPa

12. LIMITATIONS AND SIMPLIFICATIONS
    ================================
    This code does NOT consider:
    
    ✗ Buckling (Euler, local, lateral-torsional)
    ✗ P-delta (second-order) effects
    ✗ Dynamic response (natural frequencies, DAF)
    ✗ Fatigue damage
    ✗ Corrosion allowance
    ✗ Marine growth effects (can be approximated by increasing Cd/D)
    ✗ Joint flexibility (SCF, punching shear)
    ✗ Foundation flexibility
    ✗ Soil-structure interaction
    ✗ Partial safety factors (load/material factors)
    ✗ Wind loads on topside
    ✗ Slamming/impact loads
    ✗ Current profile variation with depth
    
    For final design, use a certified structural analysis software and
    follow applicable design codes (API RP 2A, ISO 19902, NORSOK, etc.)

13. OUTPUT DATA EXPLANATION
    ========================
    
    MORISON FORCES (Pure hydrodynamic):
    • Drag Force: From velocity term, in phase with particle velocity
    • Inertia Force: From acceleration term, 90° phase lead
    • Total Morison: Vector sum of drag and inertia
    
    PHASE SCAN:
    • Scans one wave period to find maximum total force
    • Reports critical phase angle (θ = ωt in degrees)
    • Useful for identifying worst-case loading condition
    
    FEM RESULTS:
    • Reactions: Support forces at fixed nodes
    • Displacements: Nodal translations (mm) and rotations (mrad)
    • Internal Forces: Axial, shear, bending, torsion in each member
    • Von Mises Stress: Maximum equivalent stress
    • Utilization: Stress ratio (σ_vm / f_y)

================================================================================
                              END OF DOCUMENTATION
================================================================================
"""
        info_text.insert('1.0', assumptions_text)
        info_text.config(state='disabled')

    # =========================================================================
    # LOAD DEFAULT PARAMETERS
    # =========================================================================
    def load_default_params(self):
        defaults = {
            'entry_E': '210000', 'entry_nu': '0.3', 'entry_fy': '355',
            'entry_rho_steel': '7850', 'entry_rho_water': '1025',
            'entry_D_leg': '2000', 'entry_t_leg': '75',
            'entry_D_brace': '800', 'entry_t_brace': '30',
            'entry_H': '17.038', 'entry_T': '9.4', 'entry_d': '50.0',
            'entry_Uc': '1.7', 'entry_wave_dir': '38.0', 'entry_current_dir': '38.0',
            'entry_N_harm': '10',
            'entry_Cd': '0.7', 'entry_Cm': '2.0',
            'entry_F_axial': '25100', 'entry_F_shear': '2900',
            'entry_M_moment': '0', 'entry_M_torsion': '0',
            'entry_custom_sw': '1100',
        }
        for attr, val in defaults.items():
            if hasattr(self, attr):
                entry = getattr(self, attr)
                entry.delete(0, tk.END)
                entry.insert(0, val)

    # =========================================================================
    # RUN ANALYSIS
    # =========================================================================
    def run_analysis(self):
        self.log_text.delete('1.0', tk.END)
        self.log("=" * 70)
        self.log("JACKET STRUCTURAL ANALYSIS - DETAILED OUTPUT")
        self.log("=" * 70)

        try:
            # Get parameters
            E = float(self.entry_E.get())
            nu = float(self.entry_nu.get())
            fy = float(self.entry_fy.get())
            rho_steel = float(self.entry_rho_steel.get())
            rho_water = float(self.entry_rho_water.get())
            
            D_leg = float(self.entry_D_leg.get())
            t_leg = float(self.entry_t_leg.get())
            D_brace = float(self.entry_D_brace.get())
            t_brace = float(self.entry_t_brace.get())
            
            H = float(self.entry_H.get())
            T = float(self.entry_T.get())
            d = float(self.entry_d.get())
            U_c = float(self.entry_Uc.get())
            wave_dir = float(self.entry_wave_dir.get())
            current_dir = float(self.entry_current_dir.get())
            wave_model = self.wave_model_var.get()
            N_harm = int(self.entry_N_harm.get())
            
            Cd = float(self.entry_Cd.get())
            Cm = float(self.entry_Cm.get())
            
            F_axial = float(self.entry_F_axial.get())
            F_shear = float(self.entry_F_shear.get())
            M_moment = float(self.entry_M_moment.get())
            M_torsion = float(self.entry_M_torsion.get())
            
            t_analysis = float(self.entry_t.get())

            # Create sections
            section_leg = TubularSection(D_leg, t_leg, "Leg", rho_steel)
            section_brace = TubularSection(D_brace, t_brace, "Brace", rho_steel)

            self.log(f"\n[SECTIONS]")
            self.log(f"  Leg: D={D_leg}mm, t={t_leg}mm, D/t={section_leg.D_t_ratio:.1f}")
            self.log(f"  Brace: D={D_brace}mm, t={t_brace}mm, D/t={section_brace.D_t_ratio:.1f}")

            # Create structure
            nodes_np = {k: np.array(v) for k, v in self.nodes_data.items()}
            structure = CustomJacketStructure(
                nodes_np, self.members_data, section_leg, section_brace,
                self.fixed_nodes, self.top_nodes, rho_steel
            )
            
            self.log(f"\n[STRUCTURE]")
            self.log(f"  Nodes: {structure.n_nodes}, Members: {structure.n_members}")
            self.log(f"  Fixed (support): {self.fixed_nodes}")
            self.log(f"  Top (interface): {self.top_nodes}")

            # Create wave
            self.log(f"\n[WAVE MODEL]")
            self.log(f"  Requested: {wave_model}, N={N_harm}")
            wave = RaschiiWave(H, T, d, U_c, wave_model, N_harm)
            self.log(f"  Actual used: {wave.get_model_info()}")
            self.log(f"  H={H}m, T={T}s, d={d}m, L={wave.L:.1f}m")
            self.log(f"  Wave direction: {wave_dir}° from North")
            self.log(f"  Current: U_c={U_c}m/s, direction={current_dir}° from North")
            
            # Store wave info for results
            self.wave_info = wave.get_model_info()
            self.wave_dir = wave_dir
            self.current_dir = current_dir

            # Create Morison calculator with separate wave/current directions
            morison = MorisonCalculator(structure, wave, wave_dir, current_dir, Cd, Cm, rho_water)

            # =====================================================================
            # MORISON FORCE ANALYSIS (PURE - before FEM)
            # =====================================================================
            self.log(f"\n" + "=" * 70)
            self.log("MORISON FORCE ANALYSIS (Pure hydrodynamic loads)")
            self.log("=" * 70)
            
            morison_results = morison.compute_all_morison_forces(t_analysis)
            
            self.log(f"\n[AT TIME t = {t_analysis:.2f}s]")
            self.log(f"  DRAG FORCE:    Fx={morison_results['total_drag'][0]/1000:8.1f} kN, "
                    f"Fy={morison_results['total_drag'][1]/1000:8.1f} kN, "
                    f"Fz={morison_results['total_drag'][2]/1000:8.1f} kN")
            self.log(f"                 |F_drag| = {np.linalg.norm(morison_results['total_drag'])/1000:.1f} kN")
            
            self.log(f"  INERTIA FORCE: Fx={morison_results['total_inertia'][0]/1000:8.1f} kN, "
                    f"Fy={morison_results['total_inertia'][1]/1000:8.1f} kN, "
                    f"Fz={morison_results['total_inertia'][2]/1000:8.1f} kN")
            self.log(f"                 |F_inertia| = {np.linalg.norm(morison_results['total_inertia'])/1000:.1f} kN")
            
            self.log(f"  TOTAL MORISON: Fx={morison_results['total_morison'][0]/1000:8.1f} kN, "
                    f"Fy={morison_results['total_morison'][1]/1000:8.1f} kN, "
                    f"Fz={morison_results['total_morison'][2]/1000:8.1f} kN")
            self.log(f"                 |F_total| = {np.linalg.norm(morison_results['total_morison'])/1000:.1f} kN")

            # Phase scan to find critical phase
            critical_info = None
            if self.do_phase_scan.get():
                self.log(f"\n[PHASE SCAN - Finding Critical Phase]")
                self.log(f"  Scanning through one wave period (T={T}s)...")
                
                phase_results = morison.find_critical_phase(n_steps=36)
                critical = phase_results['critical']
                
                self.log(f"\n  CRITICAL PHASE FOUND:")
                self.log(f"    Time: t = {critical['t']:.3f}s")
                self.log(f"    Phase angle: θ = {critical['phase_deg']:.1f}° (ωt)")
                self.log(f"    Drag force: {critical['drag_kN']:.1f} kN")
                self.log(f"    Inertia force: {critical['inertia_kN']:.1f} kN")
                self.log(f"    TOTAL MORISON: {critical['total_kN']:.1f} kN (MAX)")
                self.log(f"    Components: Fx={critical['Fx_kN']:.1f}kN, Fy={critical['Fy_kN']:.1f}kN, Fz={critical['Fz_kN']:.1f}kN")
                
                critical_info = critical
                self.phase_scan_results = phase_results

            # =====================================================================
            # FEM ANALYSIS (Combined loads)
            # =====================================================================
            self.log(f"\n" + "=" * 70)
            self.log("FEM STRUCTURAL ANALYSIS (All loads combined)")
            self.log("=" * 70)

            # Create FEM solver
            fem = FEMSolver(structure, E, nu)

            # Apply interface loads
            self.log(f"\n[APPLIED LOADS]")
            top_nodes = structure.get_top_nodes()
            n_legs = len(top_nodes)
            
            F_axial_N = F_axial * 1000.0
            F_shear_N = F_shear * 1000.0
            M_moment_Nmm = M_moment * 1e6
            M_torsion_Nmm = M_torsion * 1e6

            theta = np.deg2rad(90.0 - wave_dir)
            for node in top_nodes:
                force = np.array([
                    F_shear_N * np.cos(theta) / n_legs,
                    F_shear_N * np.sin(theta) / n_legs,
                    -F_axial_N / n_legs,
                    M_torsion_Nmm / n_legs,
                    M_moment_Nmm / n_legs,
                    0.0
                ])
                fem.apply_nodal_force(node, force)

            self.log(f"  Interface loads:")
            self.log(f"    Axial (compression): {F_axial} kN")
            self.log(f"    Shear (horizontal):  {F_shear} kN")
            self.log(f"    Overturning moment:  {M_moment} kNm")
            self.log(f"    Torsional moment:    {M_torsion} kNm")

            # Apply Morison loads
            for node_name, force in morison_results['nodal_forces'].items():
                force_vector = np.zeros(6)
                force_vector[:3] = force[:3]
                fem.apply_nodal_force(node_name, force_vector)

            self.log(f"  Morison loads: Total |F| = {np.linalg.norm(morison_results['total_morison'])/1000:.1f} kN")

            # Apply self-weight
            sw_mode = self.self_weight_mode.get()
            if sw_mode == 'calculated':
                total_weight = 0.0
                for member in structure.members:
                    geom = structure.get_member_geometry(member)
                    w = member['section'].mass_per_m * g
                    member_weight = w * geom['L']
                    total_weight += member_weight
                    F_weight = member_weight / 2.0
                    idx1 = structure.node_index[member['node1']]
                    idx2 = structure.node_index[member['node2']]
                    fem.F_global[6*idx1 + 2] -= F_weight
                    fem.F_global[6*idx2 + 2] -= F_weight
                self.log(f"  Self-weight (calculated): {total_weight/1000:.1f} kN = {total_weight/1000/g:.1f} tonnes")
            elif sw_mode == 'custom':
                custom_sw = float(self.entry_custom_sw.get()) * 1000 * g
                sw_per_node = custom_sw / structure.n_nodes
                for i in range(structure.n_nodes):
                    fem.F_global[6*i + 2] -= sw_per_node
                self.log(f"  Self-weight (custom): {custom_sw/1000:.1f} kN = {custom_sw/1000/g:.1f} tonnes")
            else:
                self.log("  Self-weight: EXCLUDED")

            # Solve
            fem.apply_boundary_conditions(structure.get_bottom_nodes())
            self.log(f"\n[SOLVING FEM SYSTEM]")
            U = fem.solve()

            # Results
            reactions = fem.get_reactions()
            internal_forces = fem.get_member_internal_forces(fy)

            self.log(f"\n[SUPPORT REACTIONS]")
            total_Rx, total_Ry, total_Rz = 0, 0, 0
            for node, R in reactions.items():
                self.log(f"  {node}: Rx={R[0]/1000:8.1f}kN, Ry={R[1]/1000:8.1f}kN, Rz={R[2]/1000:8.1f}kN")
                total_Rx += R[0]
                total_Ry += R[1]
                total_Rz += R[2]
            self.log(f"  TOTAL: Rx={total_Rx/1000:.1f}kN, Ry={total_Ry/1000:.1f}kN, Rz={total_Rz/1000:.1f}kN")

            max_disp = 0.0
            max_disp_node = ""
            for i, node in enumerate(structure.node_list):
                disp = np.linalg.norm(U[6*i:6*i+3])
                if disp > max_disp:
                    max_disp, max_disp_node = disp, node
            self.log(f"\n[DISPLACEMENTS]")
            self.log(f"  Maximum: {max_disp:.2f} mm at node {max_disp_node}")

            self.log(f"\n[STRESS CHECK]")
            self.log(f"  Yield Strength: fy = {fy} MPa")
            
            self.log(f"\n[CRITICAL MEMBERS - Top 10 by utilization]")
            sorted_members = sorted(internal_forces, key=lambda x: x['utilization'], reverse=True)
            self.log(f"  {'Member':<25} {'VM [MPa]':>10} {'Util':>10}")
            self.log(f"  {'-'*45}")
            for m in sorted_members[:10]:
                self.log(f"  {m['member']:<25} {m['von_mises_max_MPa']:>10.1f} {m['utilization']:>10.2%}")

            max_util = max(m['utilization'] for m in internal_forces)
            if max_util > 1.0:
                self.log(f"\n  *** WARNING: Max utilization {max_util:.2%} EXCEEDS YIELD! ***")
            else:
                self.log(f"\n  Maximum utilization: {max_util:.2%} (< 100%, OK)")

            # Store results
            self.analysis_results = {
                'U': U, 'reactions': reactions, 'internal_forces': internal_forces,
                'structure': structure, 'max_util': max_util,
                'morison_results': morison_results,
                'critical_phase': critical_info,
                'wave_info': wave.get_model_info()
            }

            self.log("\n" + "=" * 70)
            self.log("ANALYSIS COMPLETE")
            self.log("=" * 70)

            msg = f"Analysis complete!\n\nWave model: {wave.get_model_info()}\nMax utilization: {max_util:.2%}"
            if critical_info:
                msg += f"\nCritical phase: θ={critical_info['phase_deg']:.1f}° (t={critical_info['t']:.3f}s)"
            messagebox.showinfo("Complete", msg)

        except Exception as e:
            self.log(f"\nERROR: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Error", str(e))

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def show_summary(self):
        if self.analysis_results is None:
            messagebox.showwarning("Warning", "Run analysis first!")
            return
        self.results_text.delete('1.0', tk.END)
        for m in self.analysis_results['internal_forces']:
            self.results_text.insert(tk.END, 
                f"{m['member']}: Fx={m['Fx_max_kN']:.1f}kN, VM={m['von_mises_max_MPa']:.1f}MPa, "
                f"Util={m['utilization']:.2%}\n")

    def plot_results(self):
        if self.analysis_results is None:
            messagebox.showwarning("Warning", "Run analysis first!")
            return
        
        # Create a new popup window with 3D plot
        fig = plt.figure(figsize=(14, 11))
        ax = fig.add_subplot(111, projection='3d')
        
        structure = self.analysis_results['structure']
        internal_forces = self.analysis_results['internal_forces']
        
        # Create utilization color map
        util_map = {m['member']: m['utilization'] for m in internal_forces}
        max_util = max(m['utilization'] for m in internal_forces)
        
        # Get coordinate ranges
        coords_list = list(structure.nodes.values())
        x_min, x_max = min(c[0] for c in coords_list), max(c[0] for c in coords_list)
        y_min, y_max = min(c[1] for c in coords_list), max(c[1] for c in coords_list)
        z_min, z_max = min(c[2] for c in coords_list), max(c[2] for c in coords_list)
        
        # Plot members with color based on utilization
        for member in structure.members:
            if member['node1'] in structure.nodes and member['node2'] in structure.nodes:
                c1 = structure.nodes[member['node1']]
                c2 = structure.nodes[member['node2']]
                
                util = util_map.get(member['name'], 0)
                # Color: green (low util) -> yellow -> red (high util)
                if util < 0.5:
                    color = (2*util, 1, 0)  # green to yellow
                else:
                    color = (1, 2*(1-util), 0)  # yellow to red
                
                lw = 5 if member['type'] == 'leg' else 2.5
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], 
                       color=color, linewidth=lw, alpha=0.8)
        
        # Plot nodes
        for name, coords in structure.nodes.items():
            if name in self.fixed_nodes:
                color, marker, size = 'red', '^', 150
            elif name in self.top_nodes:
                color, marker, size = 'blue', 's', 120
            else:
                color, marker, size = 'gray', 'o', 50
            ax.scatter(coords[0], coords[1], coords[2], c=color, marker=marker, s=size, 
                      edgecolors='black', linewidths=1)
        
        # Plot water surface at z=0
        x_range = [x_min - 5, x_max + 5]
        y_range = [y_min - 5, y_max + 5]
        X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], 10),
                          np.linspace(y_range[0], y_range[1], 10))
        ax.plot_surface(X, Y, np.zeros_like(X), alpha=0.2, color='cyan')
        
        # =====================================================================
        # DRAW COMPASS / NORTH ARROW
        # =====================================================================
        arrow_base_x = x_min - 3
        arrow_base_y = y_min - 3
        arrow_base_z = z_max + 5
        arrow_length = 8
        
        # North arrow (Y direction)
        ax.quiver(arrow_base_x, arrow_base_y, arrow_base_z, 
                 0, arrow_length, 0, 
                 color='darkgreen', arrow_length_ratio=0.15, linewidth=3)
        ax.text(arrow_base_x, arrow_base_y + arrow_length + 1, arrow_base_z, 
               'N\n(+Y)', fontsize=12, fontweight='bold', color='darkgreen', ha='center')
        
        # East arrow (X direction)
        ax.quiver(arrow_base_x, arrow_base_y, arrow_base_z, 
                 arrow_length * 0.7, 0, 0, 
                 color='darkred', arrow_length_ratio=0.15, linewidth=2)
        ax.text(arrow_base_x + arrow_length * 0.7 + 1, arrow_base_y, arrow_base_z, 
               'E (+X)', fontsize=10, color='darkred', ha='left')
        
        # =====================================================================
        # DRAW WAVE AND CURRENT DIRECTION ARROWS
        # =====================================================================
        if hasattr(self, 'wave_dir') and hasattr(self, 'current_dir'):
            # Arrow origin at structure center, near water surface
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            arrow_z = 3  # Slightly above water
            dir_arrow_len = 12
            
            # Wave direction arrow (convert from compass to math angle)
            wave_theta = np.deg2rad(90.0 - self.wave_dir)
            wave_dx = dir_arrow_len * np.cos(wave_theta)
            wave_dy = dir_arrow_len * np.sin(wave_theta)
            ax.quiver(center_x, center_y, arrow_z, wave_dx, wave_dy, 0,
                     color='blue', arrow_length_ratio=0.12, linewidth=3, alpha=0.8)
            ax.text(center_x + wave_dx*1.1, center_y + wave_dy*1.1, arrow_z + 1,
                   f'Wave\n{self.wave_dir}°', fontsize=9, color='blue', ha='center', fontweight='bold')
            
            # Current direction arrow
            current_theta = np.deg2rad(90.0 - self.current_dir)
            curr_dx = dir_arrow_len * 0.8 * np.cos(current_theta)
            curr_dy = dir_arrow_len * 0.8 * np.sin(current_theta)
            ax.quiver(center_x, center_y, arrow_z - 5, curr_dx, curr_dy, 0,
                     color='cyan', arrow_length_ratio=0.12, linewidth=2.5, alpha=0.8)
            ax.text(center_x + curr_dx*1.1, center_y + curr_dy*1.1, arrow_z - 4,
                   f'Current\n{self.current_dir}°', fontsize=9, color='cyan', ha='center')
        
        ax.set_xlabel('X [m] → EAST', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y [m] → NORTH', fontsize=11, fontweight='bold')
        ax.set_zlabel('Z [m] → UP', fontsize=11, fontweight='bold')
        
        # Title with wave model info
        wave_info = self.analysis_results.get('wave_info', '') if self.analysis_results else ''
        ax.set_title(f'Jacket Structure Analysis Results\n'
                    f'Max Utilization: {max_util:.1%} | {wave_info}\n'
                    f'Color: Green (low) → Yellow → Red (high utilization)', fontsize=11)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Fixed nodes'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Top nodes'),
            Line2D([0], [0], color='green', linewidth=3, label='Low utilization'),
            Line2D([0], [0], color='red', linewidth=3, label='High utilization'),
            Line2D([0], [0], color='darkgreen', linewidth=3, label='North (+Y)'),
            Line2D([0], [0], color='blue', linewidth=3, label='Wave direction'),
            Line2D([0], [0], color='cyan', linewidth=2, label='Current direction'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.show()

    def export_csv(self):
        if self.analysis_results is None:
            messagebox.showwarning("Warning", "Run analysis first!")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if filename:
            df = pd.DataFrame(self.analysis_results['internal_forces'])
            df.to_csv(filename, index=False)
            messagebox.showinfo("Exported", f"Saved to {filename}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print("  JACKET STRUCTURAL ANALYSIS - CUSTOMIZABLE GEOMETRY")
    print("  Version 8")
    print("=" * 60 + "\n")
    
    root = tk.Tk()
    root.update_idletasks()
    width, height = 1500, 950
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    app = JacketAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

