# -*- coding: utf-8 -*-
"""
Morison Equation + FEM Structural Analysis for 3-Legged OSP Jacket Structure
INTERACTIVE GUI VERSION v7

Features:
- Interactive GUI for all input parameters
- Morison equation for hydrodynamic loads (using raschii wave library)
- 3D beam FEM analysis based on STEEL Solver displacement method
- Internal forces and stress calculation
- Comprehensive input validation
- Automatic dependency installation

Author: Design Project Team
"""

import subprocess
import sys

# =============================================================================
# AUTOMATIC DEPENDENCY INSTALLATION
# =============================================================================
def install_package(package_name, pip_name=None):
    """
    Install a package using pip if not already installed.
    
    Parameters:
        package_name: Name used for import
        pip_name: Name used for pip install (if different from package_name)
    """
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
    """Check and install all required dependencies."""
    print("=" * 60)
    print("Checking dependencies...")
    print("=" * 60)
    
    # List of required packages: (import_name, pip_name)
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("tkinter", None),  # tkinter comes with Python, skip pip install
    ]
    
    # Optional packages
    optional_packages = [
        ("raschii", "raschii"),
    ]
    
    all_installed = True
    
    # Check required packages
    for import_name, pip_name in required_packages:
        if pip_name is None:
            # Skip packages that can't be pip installed (like tkinter)
            try:
                __import__(import_name)
                print(f"  ✓ {import_name} is available")
            except ImportError:
                print(f"  ✗ {import_name} not available (built-in package, please reinstall Python with tkinter)")
                all_installed = False
        else:
            try:
                __import__(import_name)
                print(f"  ✓ {import_name} is available")
            except ImportError:
                success = install_package(import_name, pip_name)
                if not success:
                    all_installed = False
    
    # Check optional packages
    print("\nOptional packages:")
    raschii_available = False
    for import_name, pip_name in optional_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {import_name} is available")
            if import_name == "raschii":
                raschii_available = True
        except ImportError:
            print(f"  ? {import_name} not found, attempting to install...")
            success = install_package(import_name, pip_name)
            if success:
                raschii_available = True
            else:
                print(f"    (Will use fallback Airy wave theory)")
    
    print("=" * 60)
    if all_installed:
        print("All required dependencies are ready!")
    else:
        print("Some dependencies could not be installed.")
        print("Please install them manually using:")
        print("  pip install numpy pandas matplotlib raschii")
    print("=" * 60 + "\n")
    
    return raschii_available

# Run dependency check before importing
RASCHII_AVAILABLE = check_and_install_dependencies()

# =============================================================================
# IMPORTS (after dependency check)
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import io

warnings.filterwarnings('ignore')

# Try to import raschii again after potential installation
try:
    import raschii
    RASCHII_AVAILABLE = True
except ImportError:
    RASCHII_AVAILABLE = False

# =============================================================================
# CONSTANTS (Default Values - Can be modified via GUI)
# =============================================================================
DEFAULT_PARAMS = {
    # Physical Constants
    'g': 9.81,           # Gravitational acceleration [m/s²]
    'rho_water': 1025,   # Seawater density [kg/m³]
    
    # Material Properties - Steel S355
    'E': 210000,         # Young's modulus [N/mm²] = [MPa]
    'nu': 0.3,           # Poisson's ratio [-]
    'fy': 355,           # Yield strength [N/mm²] = [MPa]
    'rho_steel': 7850,   # Steel density [kg/m³]
    
    # Section Dimensions [mm]
    'D_leg': 2000.0,     # Leg outer diameter
    't_leg': 75.0,       # Leg wall thickness
    'D_brace': 800.0,    # Brace outer diameter
    't_brace': 30.0,     # Brace wall thickness
    
    # Wave Parameters
    'H': 17.038,         # Wave height [m]
    'T': 9.4,            # Wave period [s]
    'd': 50.0,           # Water depth [m]
    'U_c': 1.7,          # Current velocity [m/s]
    'wave_direction': 38.0,  # From North [degrees]
    'N_harmonics': 10,   # Number of harmonics for Fenton wave
    
    # Morison Coefficients
    'Cd': 0.7,           # Drag coefficient
    'Cm': 2.0,           # Inertia coefficient
    
    # Interface Loads [kN / kNm]
    'F_axial': 25100.0,  # Axial compression
    'F_shear': 2900.0,   # Horizontal shear
    'M_moment': 0.0,     # Global overturning moment
    'M_torsion': 0.0,    # Global torsional moment
    
    # Structure Reference
    'z_water_ref': 47.0,  # MWL depth from original coordinate system
    
    # Analysis Options
    'include_self_weight': True,
    'n_gauss': 15,       # Gauss integration points
}

# =============================================================================
# RASCHII LIBRARY CONSTRAINTS AND WARNINGS
# =============================================================================
"""
RASCHII LIBRARY LIMITATIONS AND NOTES:
=====================================

1. WAVE MODEL ORDERS:
   - AiryWave: Linear wave theory (1st order), valid for small amplitude waves
   - StokesWave: Maximum N=5 (5th order Stokes wave theory)
   - FentonWave: Typically N=5-20 harmonics, higher accuracy for steep waves

2. WAVE STEEPNESS LIMITS:
   - Breaking wave limit: H/L ≈ 0.142 (Miche criterion for deep water)
   - Deep water breaking: H/gT² < 0.142
   - Shallow water breaking: H/d < 0.78 (solitary wave limit)
   - Recommended safe range: H/L < 0.10 for numerical stability

3. COMMON ERROR CONDITIONS:
   - Wave too steep: H/L > 0.14 will cause convergence failure
   - Wave period too short: Very short periods with large heights fail
   - Shallow water with steep waves: Combination causes instability
   - Invalid depth: d ≤ 0 or d < H/2 can cause errors

4. INPUT VALIDATION RULES:
   a) Wave Height (H):
      - Must be positive: H > 0
      - Physical limit: H < 0.78 * d (shallow water)
      - Steepness check: H < 0.142 * g * T² / (2π) for deep water
   
   b) Wave Period (T):
      - Must be positive: T > 0
      - Typical ocean range: 3s < T < 25s
      - Very short periods (T < 2s) may cause numerical issues
   
   c) Water Depth (d):
      - Must be positive: d > 0
      - Should be greater than H/2 for physical validity
   
   d) Combined check:
      - Wave length L = g*T²/(2π)*tanh(2πd/(g*T²/(2π))) approximately
      - Check H/L < 0.14 for valid wave

5. WAVELENGTH ESTIMATION:
   - Deep water: L₀ = g*T²/(2π) ≈ 1.56*T² [m]
   - General: L = L₀ * tanh(2πd/L₀) (iterative)
   - Quick check: if H > 0.14 * 1.56 * T², wave is likely too steep

6. RECOMMENDED PARAMETER RANGES:
   - Wave height: 0.5m < H < 30m (extreme storms)
   - Wave period: 3s < T < 20s (typical ocean waves)
   - Water depth: 10m < d < 300m (typical offshore)
   - Current: 0 < U_c < 3 m/s (typical)
"""

def validate_wave_parameters(H, T, d, show_warnings=True):
    """
    Validate wave parameters before running analysis.
    
    Returns:
        (bool, str): (is_valid, message)
    """
    errors = []
    warnings_list = []
    
    # Basic checks
    if H <= 0:
        errors.append("Wave height H must be positive")
    if T <= 0:
        errors.append("Wave period T must be positive")
    if d <= 0:
        errors.append("Water depth d must be positive")
    
    if errors:
        return False, "\n".join(errors)
    
    # Physical validity checks
    g = 9.81
    
    # Deep water wavelength approximation
    L0 = g * T**2 / (2 * np.pi)
    
    # Iterative wavelength calculation
    L = L0
    for _ in range(20):
        L_new = L0 * np.tanh(2 * np.pi * d / L)
        if abs(L_new - L) < 0.001:
            break
        L = L_new
    
    # Wave steepness
    steepness = H / L
    
    # Deep water steepness parameter
    deep_steepness = H / (g * T**2 / (2 * np.pi))
    
    # Shallow water limit
    relative_depth = d / L
    
    # Check breaking limit
    if steepness > 0.142:
        errors.append(f"Wave too steep! H/L = {steepness:.4f} > 0.142 (breaking limit)\n"
                     f"Either reduce wave height H or increase wave period T\n"
                     f"Current: H={H}m, T={T}s, L≈{L:.1f}m")
    
    if H > 0.78 * d:
        errors.append(f"Wave height exceeds shallow water limit! H/d = {H/d:.3f} > 0.78\n"
                     f"Either reduce H or increase water depth d")
    
    if H > d:
        errors.append(f"Wave height cannot exceed water depth! H={H}m > d={d}m")
    
    # Warnings for potentially problematic cases
    if steepness > 0.10:
        warnings_list.append(f"Wave steepness H/L = {steepness:.4f} is high (> 0.10)\n"
                           f"May cause numerical convergence issues")
    
    if T < 3:
        warnings_list.append(f"Wave period T = {T}s is very short\n"
                           f"Typical ocean waves: 3s < T < 20s")
    
    if T > 25:
        warnings_list.append(f"Wave period T = {T}s is very long\n"
                           f"May represent tsunami or infragravity waves")
    
    if relative_depth < 0.05:
        warnings_list.append(f"Very shallow water! d/L = {relative_depth:.4f}\n"
                           f"Linear wave theory may not be accurate")
    
    if errors:
        return False, "\n".join(errors)
    
    info = (f"Wave validation passed:\n"
            f"  Wavelength L ≈ {L:.1f} m\n"
            f"  Steepness H/L = {steepness:.4f} (limit: 0.142)\n"
            f"  Relative depth d/L = {relative_depth:.3f}\n"
            f"  Wave regime: {'Deep' if relative_depth > 0.5 else 'Intermediate' if relative_depth > 0.05 else 'Shallow'}")
    
    if warnings_list and show_warnings:
        info += "\n\nWarnings:\n" + "\n".join(warnings_list)
    
    return True, info


def calculate_max_wave_height(T, d):
    """Calculate maximum allowable wave height for given period and depth."""
    g = 9.81
    L0 = g * T**2 / (2 * np.pi)
    
    # Iterative wavelength
    L = L0
    for _ in range(20):
        L_new = L0 * np.tanh(2 * np.pi * d / L)
        if abs(L_new - L) < 0.001:
            break
        L = L_new
    
    # Breaking limit
    H_max_steepness = 0.14 * L  # Slightly below 0.142 for safety
    H_max_shallow = 0.78 * d
    
    return min(H_max_steepness, H_max_shallow, d)


# =============================================================================
# TUBULAR SECTION PROPERTIES CLASS
# =============================================================================
@dataclass
class TubularSection:
    """
    Tubular (pipe) cross-section properties.
    
    ASSUMPTION: Thin-walled circular cross-section (D/t > 10 typically)
    """
    D_outer: float  # Outer diameter [mm]
    t: float        # Wall thickness [mm]
    name: str = ""
    rho_steel: float = 7850  # Steel density [kg/m³]

    def __post_init__(self):
        self.D_inner = self.D_outer - 2 * self.t
        self.R_outer = self.D_outer / 2.0
        self.R_inner = self.D_inner / 2.0

        # Cross-sectional area [mm²]
        self.Ax_mm2 = np.pi / 4.0 * (self.D_outer**2 - self.D_inner**2)
        self.Ax_cm2 = self.Ax_mm2 / 100.0
        self.Ax_m2 = self.Ax_mm2 / 1e6

        # Second moments of area [mm⁴]
        self.Iy_mm4 = np.pi / 64.0 * (self.D_outer**4 - self.D_inner**4)
        self.Iz_mm4 = self.Iy_mm4
        self.Iy_cm4 = self.Iy_mm4 / 1e4
        self.Iz_cm4 = self.Iz_mm4 / 1e4
        self.Iy_m4 = self.Iy_mm4 / 1e12
        self.Iz_m4 = self.Iz_mm4 / 1e12

        # Polar moment of inertia [mm⁴]
        self.Ix_mm4 = np.pi / 32.0 * (self.D_outer**4 - self.D_inner**4)
        self.Ix_cm4 = self.Ix_mm4 / 1e4
        self.Ix_m4 = self.Ix_mm4 / 1e12

        # Shear areas [mm²]
        self.Ay_mm2 = 0.5 * self.Ax_mm2
        self.Az_mm2 = 0.5 * self.Ax_mm2
        self.Ay_m2 = self.Ay_mm2 / 1e6
        self.Az_m2 = self.Az_mm2 / 1e6

        # Section moduli [mm³]
        self.Wy_mm3 = self.Iy_mm4 / self.R_outer
        self.Wz_mm3 = self.Iz_mm4 / self.R_outer
        self.Wy_cm3 = self.Wy_mm3 / 1e3
        self.Wz_cm3 = self.Wz_mm3 / 1e3

        # Torsional section modulus [mm³]
        self.Wx_mm3 = self.Ix_mm4 / self.R_outer
        self.Wx_cm3 = self.Wx_mm3 / 1e3

        # Effective areas for shear stress
        self.Sx_mm2 = self.Ax_mm2
        self.Sy_mm2 = self.Ax_mm2
        self.Sz_mm2 = self.Ax_mm2

        # Mass per unit length [kg/m]
        self.mass_per_m = self.Ax_m2 * self.rho_steel
        
        # Check thin-wall assumption
        self.D_t_ratio = self.D_outer / self.t

    def get_stress_points(self) -> Dict[str, Tuple[float, float]]:
        """Return stress calculation points around the circumference."""
        R = self.R_outer
        points = {}
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        for i, angle in enumerate(angles):
            rad = np.radians(angle)
            y = R * np.cos(rad)
            z = R * np.sin(rad)
            points[f'A{i+1}'] = (y, z)
        return points

    def calc_stress_at_point(self, Fx: float, Fy: float, Fz: float,
                             Mx: float, My: float, Mz: float,
                             point_id: str) -> Dict[str, float]:
        """Calculate stresses at a given point on the section."""
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

        return {
            'sigma_Fx': sigma_Fx, 'sigma_My': sigma_My, 'sigma_Mz': sigma_Mz,
            'sigma_total': sigma_total, 'tau_Mx': tau_Mx, 'tau_Fy': tau_Fy,
            'tau_Fz': tau_Fz, 'tau_total': tau_total, 'von_mises': sigma_vm
        }


# =============================================================================
# RASCHII WAVE WRAPPER CLASS
# =============================================================================
class RaschiiWave:
    """Wrapper class for raschii wave models."""

    def __init__(self, H: float, T: float, d: float, U_c: float = 0.0,
                 wave_model: str = 'Fenton', N: int = 10, dt: float = 0.001):
        self.H = H
        self.T = T
        self.d = d
        self.U_c = U_c
        self.wave_model_name = wave_model
        self.N = N
        self.dt = dt

        if RASCHII_AVAILABLE:
            self.wave = self._create_wave(wave_model, N)
            self.omega = self.wave.omega
            self.k = self.wave.k
            self.L = self.wave.length
            self.c = self.wave.c
        else:
            self.omega = 2.0 * np.pi / T
            self.k = self._solve_dispersion(self.omega, d)
            self.L = 2.0 * np.pi / self.k
            self.c = self.L / T
            self.wave = None

        self.a = H / 2.0

    def _solve_dispersion(self, omega: float, d: float) -> float:
        """Solve dispersion relation using Newton-Raphson."""
        g = 9.81
        k = omega**2 / g
        for _ in range(50):
            f = omega**2 - g * k * np.tanh(k * d)
            df = -g * (np.tanh(k * d) + k * d / np.cosh(k * d)**2)
            k_new = k - f / df
            if abs(k_new - k) < 1e-10:
                break
            k = k_new
        return k

    def _create_wave(self, model: str, N: int):
        """Create raschii wave object."""
        if model.lower() == 'auto':
            airy = raschii.AiryWave(height=self.H, depth=self.d, period=self.T)
            steepness = self.H / airy.length
            if steepness < 0.01:
                model = 'Airy'
            elif steepness < 0.05:
                model = 'Stokes'
                N = min(N, 5)
            else:
                model = 'Fenton'
                N = max(N, 10)

        if model.lower() == 'fenton':
            return raschii.FentonWave(height=self.H, depth=self.d, period=self.T, N=N)
        elif model.lower() == 'stokes':
            return raschii.StokesWave(height=self.H, depth=self.d, period=self.T, N=min(N, 5))
        elif model.lower() == 'airy':
            return raschii.AiryWave(height=self.H, depth=self.d, period=self.T)
        else:
            raise ValueError(f"Unknown wave model: {model}")

    def eta(self, x: float, t: float = 0.0) -> float:
        """Free surface elevation relative to MWL."""
        if self.wave is not None:
            eta_raschii = self.wave.surface_elevation(x, t=t)
            if hasattr(eta_raschii, '__len__'):
                eta_raschii = eta_raschii[0]
            return eta_raschii - self.d
        else:
            return self.a * np.cos(self.k * x - self.omega * t)

    def velocity(self, x: float, z_mwl: float, t: float = 0.0) -> Tuple[float, float]:
        """Horizontal and vertical water particle velocities."""
        eta_local = self.eta(x, t)
        if z_mwl > eta_local:
            return (0.0, 0.0)

        if self.wave is not None:
            z_raschii = z_mwl + self.d
            z_raschii = max(0.01, min(z_raschii, self.d + eta_local - 0.01))
            vel = self.wave.velocity(x, z_raschii, t=t)
            if hasattr(vel, 'shape'):
                u = float(vel[0, 0])
                w = float(vel[0, 1])
            else:
                u = float(vel[0])
                w = float(vel[1])
        else:
            kd = self.k * self.d
            kz = self.k * (z_mwl + self.d)
            phase = self.k * x - self.omega * t
            u = self.a * self.omega * np.cosh(kz) / np.sinh(kd) * np.cos(phase)
            w = self.a * self.omega * np.sinh(kz) / np.sinh(kd) * np.sin(phase)

        u += self.U_c
        return (u, w)

    def acceleration(self, x: float, z_mwl: float, t: float = 0.0) -> Tuple[float, float]:
        """Water particle accelerations."""
        eta_local = self.eta(x, t)
        if z_mwl > eta_local:
            return (0.0, 0.0)

        u0, w0 = self.velocity(x, z_mwl, t)
        u1, w1 = self.velocity(x, z_mwl, t + self.dt)
        return ((u1 - u0) / self.dt, (w1 - w0) / self.dt)

    def get_kinematics(self, x: float, z_mwl: float, t: float = 0.0) -> Dict:
        """Get wave kinematics at a point."""
        eta_local = self.eta(x, t)
        if z_mwl > eta_local:
            return {'u': 0.0, 'w': 0.0, 'du_dt': 0.0, 'dw_dt': 0.0,
                    'submerged': False, 'eta': eta_local}

        u, w = self.velocity(x, z_mwl, t)
        du_dt, dw_dt = self.acceleration(x, z_mwl, t)
        return {'u': u, 'w': w, 'du_dt': du_dt, 'dw_dt': dw_dt,
                'submerged': True, 'eta': eta_local}


# =============================================================================
# STRUCTURE DEFINITION
# =============================================================================
@dataclass
class JacketStructure:
    """3-Legged OSP Jacket Structure with FEM definition."""

    z_water_ref: float = 47.0
    D_leg: float = 2000.0
    t_leg: float = 75.0
    D_brace: float = 800.0
    t_brace: float = 30.0
    rho_steel: float = 7850

    def __post_init__(self):
        self.section_leg = TubularSection(self.D_leg, self.t_leg, "Column/Leg", self.rho_steel)
        self.section_brace = TubularSection(self.D_brace, self.t_brace, "Brace", self.rho_steel)

        self.nodes = self._define_nodes()
        self.node_list = list(self.nodes.keys())
        self.n_nodes = len(self.node_list)
        self.n_dof = 6 * self.n_nodes

        self.node_index = {name: i for i, name in enumerate(self.node_list)}
        self.members = self._define_members()
        self.n_members = len(self.members)

    def _define_nodes(self) -> Dict[str, np.ndarray]:
        """Define all nodes in MWL-based coordinates."""
        z_ref = self.z_water_ref
        nodes: Dict[str, np.ndarray] = {}

        # LEG A
        nodes['A1'] = np.array([-9.2376, -16.0,      0.0 - z_ref])
        nodes['A2'] = np.array([-7.9254, -13.7272,  28.41 - z_ref])
        nodes['A3'] = np.array([-6.7947, -11.7688,  52.89 - z_ref])
        nodes['A4'] = np.array([-5.8197, -10.08,    74.0  - z_ref])

        # LEG B
        nodes['B1'] = np.array([18.4752,  0.0,      0.0 - z_ref])
        nodes['B2'] = np.array([15.8508,  0.0,     28.41 - z_ref])
        nodes['B3'] = np.array([13.5894,  0.0,     52.89 - z_ref])
        nodes['B4'] = np.array([11.6394,  0.0,     74.0  - z_ref])

        # LEG C
        nodes['C1'] = np.array([-9.2376,  16.0,     0.0 - z_ref])
        nodes['C2'] = np.array([-7.9254,  13.7272, 28.41 - z_ref])
        nodes['C3'] = np.array([-6.7947,  11.7688, 52.89 - z_ref])
        nodes['C4'] = np.array([-5.8197,  10.08,   74.0  - z_ref])

        # Hinge nodes (Level 1)
        nodes['HAB1'] = np.array([4.2657, -7.3884, 15.291 - z_ref])
        nodes['HBC1'] = np.array([4.2657,  7.3884, 15.291 - z_ref])
        nodes['HCA1'] = np.array([-8.5313, 0.0,    15.291 - z_ref])

        # Hinge nodes (Level 2)
        nodes['HAB2'] = np.array([3.6583, -6.3364, 41.5902 - z_ref])
        nodes['HBC2'] = np.array([3.6583,  6.3364, 41.5902 - z_ref])
        nodes['HCA2'] = np.array([-7.3166, 0.0,    41.5902 - z_ref])

        # Hinge nodes (Level 3)
        nodes['HAB3'] = np.array([3.1348, -5.4296, 64.2608 - z_ref])
        nodes['HBC3'] = np.array([3.1348,  5.4296, 64.2608 - z_ref])
        nodes['HCA3'] = np.array([-6.2695, 0.0,    64.2608 - z_ref])

        return nodes

    def _define_members(self) -> List[Dict]:
        """Define all members."""
        members: List[Dict] = []

        # Legs
        for leg in ['A', 'B', 'C']:
            for i in [1, 2, 3]:
                members.append({
                    'name': f'Leg_{leg}{i}-{leg}{i+1}',
                    'node1': f'{leg}{i}', 'node2': f'{leg}{i+1}',
                    'section': self.section_leg, 'type': 'leg'
                })

        # Bottom horizontal braces
        for n1, n2 in [('A1', 'B1'), ('B1', 'C1'), ('C1', 'A1')]:
            members.append({'name': f'HBrace_{n1}-{n2}', 'node1': n1, 'node2': n2,
                           'section': self.section_brace, 'type': 'h_brace'})

        # Level 2 horizontal braces
        for n1, n2 in [('A2', 'B2'), ('B2', 'C2'), ('C2', 'A2')]:
            members.append({'name': f'HBrace_{n1}-{n2}', 'node1': n1, 'node2': n2,
                           'section': self.section_brace, 'type': 'h_brace'})

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
                members.append({'name': f'XBr_{n1}-{n2}', 'node1': n1, 'node2': n2,
                               'section': self.section_brace, 'type': 'x_brace'})

        return members

    def get_member_geometry(self, member: Dict) -> Dict:
        """Compute basic geometry of a member."""
        coord1 = self.nodes[member['node1']]
        coord2 = self.nodes[member['node2']]
        dL = coord2 - coord1
        L = np.linalg.norm(dL)
        unit_vec = dL / L if L > 0 else np.array([1.0, 0.0, 0.0])
        return {'coord1': coord1, 'coord2': coord2, 'dL': dL,
                'L': L, 'L_mm': L * 1000.0, 'unit_vec': unit_vec}

    def get_top_nodes(self) -> List[str]:
        return ['A4', 'B4', 'C4']

    def get_bottom_nodes(self) -> List[str]:
        return ['A1', 'B1', 'C1']


# =============================================================================
# FEM BEAM ELEMENT CLASS
# =============================================================================
class BeamElement3D:
    """3D Timoshenko beam element."""

    def __init__(self, node1_coords: np.ndarray, node2_coords: np.ndarray,
                 section: TubularSection, E: float = 210000, G: float = 80769,
                 include_shear: bool = True):
        self.node1 = node1_coords
        self.node2 = node2_coords
        self.section = section
        self.E = E
        self.G = G
        self.include_shear = include_shear

        self.dL = node2_coords - node1_coords
        self.L = np.linalg.norm(self.dL)
        self.L_mm = self.L * 1000.0

        self.T = self._compute_transformation_matrix()
        self.K_local = self._compute_local_stiffness()
        self.K_global = self.T.T @ self.K_local @ self.T

    def _compute_transformation_matrix(self) -> np.ndarray:
        """Compute 12x12 transformation matrix."""
        lx = self.dL / self.L
        global_z = np.array([0.0, 0.0, 1.0])

        if abs(np.dot(lx, global_z)) > 0.999:
            global_y = np.array([0.0, 1.0, 0.0])
            ly = np.cross(global_z, lx)
            ly_norm = np.linalg.norm(ly)
            if ly_norm > 1e-10:
                ly = ly / ly_norm
            else:
                ly = global_y
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

    def _compute_local_stiffness(self) -> np.ndarray:
        """Compute 12x12 local stiffness matrix."""
        L = self.L_mm
        Ax = self.section.Ax_mm2
        Ay = self.section.Ay_mm2
        Az = self.section.Az_mm2
        Ix = self.section.Ix_mm4
        Iy = self.section.Iy_mm4
        Iz = self.section.Iz_mm4
        E = self.E
        G = self.G

        if self.include_shear and Ay > 0.0 and Az > 0.0:
            Phi_y = 12.0 * E * Iz / (G * Az * L**2)
            Phi_z = 12.0 * E * Iy / (G * Ay * L**2)
        else:
            Phi_y = 0.0
            Phi_z = 0.0

        alpha = E * Ax / L
        bz = E * Iz / ((1.0 + Phi_y) * L**3)
        by = E * Iy / ((1.0 + Phi_z) * L**3)
        t = G * Ix / L

        K = np.zeros((12, 12))

        # Axial
        K[0, 0] = K[6, 6] = alpha
        K[0, 6] = K[6, 0] = -alpha

        # Bending y-z plane
        K[1, 1] = K[7, 7] = 12.0 * bz
        K[1, 7] = K[7, 1] = -12.0 * bz
        K[1, 5] = K[5, 1] = K[1, 11] = K[11, 1] = 6.0 * bz * L
        K[7, 5] = K[5, 7] = K[7, 11] = K[11, 7] = -6.0 * bz * L
        K[5, 5] = K[11, 11] = (4.0 + Phi_y) * bz * L**2
        K[5, 11] = K[11, 5] = (2.0 - Phi_y) * bz * L**2

        # Bending x-z plane
        K[2, 2] = K[8, 8] = 12.0 * by
        K[2, 8] = K[8, 2] = -12.0 * by
        K[2, 4] = K[4, 2] = K[2, 10] = K[10, 2] = -6.0 * by * L
        K[8, 4] = K[4, 8] = K[8, 10] = K[10, 8] = 6.0 * by * L
        K[4, 4] = K[10, 10] = (4.0 + Phi_z) * by * L**2
        K[4, 10] = K[10, 4] = (2.0 - Phi_z) * by * L**2

        # Torsion
        K[3, 3] = K[9, 9] = t
        K[3, 9] = K[9, 3] = -t

        return K

    def get_internal_forces(self, u_global: np.ndarray) -> Dict:
        """Compute internal forces from global displacements."""
        u_local = self.T @ u_global
        F_local = self.K_local @ u_local

        return {
            'node1': {'Fx': -F_local[0], 'Fy': -F_local[1], 'Fz': -F_local[2],
                      'Mx': -F_local[3], 'My': -F_local[4], 'Mz': -F_local[5]},
            'node2': {'Fx': F_local[6], 'Fy': F_local[7], 'Fz': F_local[8],
                      'Mx': F_local[9], 'My': F_local[10], 'Mz': F_local[11]},
            'u_local': u_local, 'F_local': F_local
        }


# =============================================================================
# FEM SOLVER CLASS
# =============================================================================
class FEMSolver:
    """3D frame FEM solver."""

    def __init__(self, structure: JacketStructure, E: float = 210000, nu: float = 0.3):
        self.structure = structure
        self.n_dof = structure.n_dof
        self.E = E
        self.G = E / (2 * (1 + nu))

        self.K_global = np.zeros((self.n_dof, self.n_dof))
        self.F_global = np.zeros(self.n_dof)
        self.U_global = np.zeros(self.n_dof)

        self.elements: List[BeamElement3D] = []
        self._build_elements()
        self._assemble_global_stiffness()

    def _build_elements(self):
        """Create beam element objects."""
        for member in self.structure.members:
            coord1 = self.structure.nodes[member['node1']]
            coord2 = self.structure.nodes[member['node2']]
            section = member['section']
            element = BeamElement3D(coord1, coord2, section, self.E, self.G)
            self.elements.append(element)

    def _assemble_global_stiffness(self):
        """Assemble global stiffness matrix."""
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

    def apply_nodal_force(self, node_name: str, force_vector: np.ndarray):
        """Apply concentrated load at a node."""
        idx = self.structure.node_index[node_name]
        dofs = np.arange(6 * idx, 6 * idx + 6)
        self.F_global[dofs] += force_vector

    def apply_boundary_conditions(self, fixed_nodes: List[str]):
        """Define fixed DOFs at given nodes."""
        self.fixed_dofs: List[int] = []
        for node_name in fixed_nodes:
            idx = self.structure.node_index[node_name]
            dofs = np.arange(6 * idx, 6 * idx + 6)
            self.fixed_dofs.extend(dofs.tolist())
        self.fixed_dofs = np.array(self.fixed_dofs, dtype=int)
        self.free_dofs = np.setdiff1d(np.arange(self.n_dof), self.fixed_dofs)

    def solve(self) -> np.ndarray:
        """Solve global equilibrium system."""
        K_ff = self.K_global[np.ix_(self.free_dofs, self.free_dofs)]
        F_f = self.F_global[self.free_dofs]

        try:
            U_f = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            U_f = np.linalg.lstsq(K_ff, F_f, rcond=None)[0]

        self.U_global = np.zeros(self.n_dof)
        self.U_global[self.free_dofs] = U_f
        return self.U_global

    def get_reactions(self) -> Dict[str, np.ndarray]:
        """Compute reaction forces at fixed DOFs."""
        R = self.K_global @ self.U_global - self.F_global
        reactions: Dict[str, np.ndarray] = {}
        for dof in self.fixed_dofs:
            node_idx = dof // 6
            local_dof = dof % 6
            node_name = self.structure.node_list[node_idx]
            if node_name not in reactions:
                reactions[node_name] = np.zeros(6)
            reactions[node_name][local_dof] = R[dof]
        return reactions

    def get_member_internal_forces(self, fy: float = 355) -> List[Dict]:
        """Compute internal forces and utilization for all members."""
        results: List[Dict] = []

        for i, member in enumerate(self.structure.members):
            element = self.elements[i]
            idx1 = self.structure.node_index[member['node1']]
            idx2 = self.structure.node_index[member['node2']]
            dof1 = np.arange(6 * idx1, 6 * idx1 + 6)
            dof2 = np.arange(6 * idx2, 6 * idx2 + 6)
            u_elem = np.concatenate([self.U_global[dof1], self.U_global[dof2]])

            forces = element.get_internal_forces(u_elem)
            section = member['section']

            Fx_max = max(abs(forces['node1']['Fx']), abs(forces['node2']['Fx']))
            Fy_max = max(abs(forces['node1']['Fy']), abs(forces['node2']['Fy']))
            Fz_max = max(abs(forces['node1']['Fz']), abs(forces['node2']['Fz']))
            Mx_max = max(abs(forces['node1']['Mx']), abs(forces['node2']['Mx']))
            My_max = max(abs(forces['node1']['My']), abs(forces['node2']['My']))
            Mz_max = max(abs(forces['node1']['Mz']), abs(forces['node2']['Mz']))

            stresses = section.calc_stress_at_point(
                forces['node1']['Fx'], forces['node1']['Fy'], forces['node1']['Fz'],
                forces['node1']['Mx'], forces['node1']['My'], forces['node1']['Mz'], 'A1')

            max_vm = 0.0
            for pt in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']:
                st = section.calc_stress_at_point(
                    forces['node1']['Fx'], forces['node1']['Fy'], forces['node1']['Fz'],
                    forces['node1']['Mx'], forces['node1']['My'], forces['node1']['Mz'], pt)
                max_vm = max(max_vm, st['von_mises'])

            results.append({
                'member': member['name'], 'type': member['type'],
                'node1': member['node1'], 'node2': member['node2'],
                'length_m': element.L, 'section': section.name,
                'Fx_N1_kN': forces['node1']['Fx'] / 1000.0,
                'Fy_N1_kN': forces['node1']['Fy'] / 1000.0,
                'Fz_N1_kN': forces['node1']['Fz'] / 1000.0,
                'Mx_N1_kNm': forces['node1']['Mx'] / 1e6,
                'My_N1_kNm': forces['node1']['My'] / 1e6,
                'Mz_N1_kNm': forces['node1']['Mz'] / 1e6,
                'Fx_N2_kN': forces['node2']['Fx'] / 1000.0,
                'Fy_N2_kN': forces['node2']['Fy'] / 1000.0,
                'Fz_N2_kN': forces['node2']['Fz'] / 1000.0,
                'Mx_N2_kNm': forces['node2']['Mx'] / 1e6,
                'My_N2_kNm': forces['node2']['My'] / 1e6,
                'Mz_N2_kNm': forces['node2']['Mz'] / 1e6,
                'Fx_max_kN': Fx_max / 1000.0,
                'Fy_max_kN': Fy_max / 1000.0,
                'Fz_max_kN': Fz_max / 1000.0,
                'Mx_max_kNm': Mx_max / 1e6,
                'My_max_kNm': My_max / 1e6,
                'Mz_max_kNm': Mz_max / 1e6,
                'sigma_max_MPa': stresses['sigma_total'],
                'tau_max_MPa': stresses['tau_total'],
                'von_mises_max_MPa': max_vm,
                'utilization': max_vm / fy
            })
        return results


# =============================================================================
# MORISON FORCE CALCULATOR
# =============================================================================
class MorisonCalculator:
    """Morison force calculation integrated with FEM solver."""

    def __init__(self, structure: JacketStructure, wave: RaschiiWave,
                 wave_direction: float = 38.0, Cd: float = 0.7, Cm: float = 2.0,
                 rho_water: float = 1025):
        self.structure = structure
        self.wave = wave
        self.theta_dir = np.deg2rad(90.0 - wave_direction)
        self.wave_direction_compass = wave_direction
        self.Cd = Cd
        self.Cm = Cm
        self.rho = rho_water

    def get_wave_coords(self, x: float, y: float) -> float:
        """Transform global (x, y) into wave propagation coordinate."""
        return x * np.cos(self.theta_dir) + y * np.sin(self.theta_dir)

    def get_kinematics_3d(self, x: float, y: float, z: float, t: float) -> Dict:
        """Get 3D wave kinematics at a global point."""
        x_wave = self.get_wave_coords(x, y)
        kin2d = self.wave.get_kinematics(x_wave, z, t)

        if not kin2d['submerged']:
            return {'u': 0.0, 'v': 0.0, 'w': 0.0, 'du_dt': 0.0, 'dv_dt': 0.0,
                    'dw_dt': 0.0, 'submerged': False, 'eta': kin2d['eta']}

        cos_dir = np.cos(self.theta_dir)
        sin_dir = np.sin(self.theta_dir)

        return {
            'u': kin2d['u'] * cos_dir, 'v': kin2d['u'] * sin_dir, 'w': kin2d['w'],
            'du_dt': kin2d['du_dt'] * cos_dir, 'dv_dt': kin2d['du_dt'] * sin_dir,
            'dw_dt': kin2d['dw_dt'], 'submerged': True, 'eta': kin2d['eta']
        }

    def morison_force_per_length(self, U_vec: np.ndarray, A_vec: np.ndarray,
                                 unit_vec: np.ndarray, D: float) -> np.ndarray:
        """Calculate Morison force per unit length [N/m]."""
        U_perp = U_vec - np.dot(U_vec, unit_vec) * unit_vec
        A_perp = A_vec - np.dot(A_vec, unit_vec) * unit_vec
        U_perp_mag = np.linalg.norm(U_perp)

        A_cross = np.pi * D**2 / 4.0

        if U_perp_mag > 1e-10:
            F_drag = 0.5 * self.rho * self.Cd * D * U_perp_mag * U_perp
        else:
            F_drag = np.zeros(3)

        F_inertia = self.rho * self.Cm * A_cross * A_perp
        return F_drag + F_inertia

    def compute_member_morison_forces(self, member_idx: int, t: float = 0.0,
                                      n_gauss: int = 15) -> Dict:
        """Compute Morison forces on a specific member."""
        member = self.structure.members[member_idx]
        coord1 = self.structure.nodes[member['node1']]
        coord2 = self.structure.nodes[member['node2']]
        D = member['section'].D_outer / 1000.0

        dL = coord2 - coord1
        L = np.linalg.norm(dL)
        unit_vec = dL / L

        xi, weights = np.polynomial.legendre.leggauss(n_gauss)
        s_values = (xi + 1.0) / 2.0
        w_scaled = weights / 2.0

        F1 = np.zeros(3)
        F2 = np.zeros(3)
        total_force = np.zeros(3)
        submerged_length = 0.0

        for s, w in zip(s_values, w_scaled):
            pos = coord1 + s * dL
            x, y, z = pos
            kin = self.get_kinematics_3d(x, y, z, t)
            if not kin['submerged']:
                continue

            submerged_length += w * L
            U_vec = np.array([kin['u'], kin['v'], kin['w']])
            A_vec = np.array([kin['du_dt'], kin['dv_dt'], kin['dw_dt']])

            f_per_L = self.morison_force_per_length(U_vec, A_vec, unit_vec, D)
            f_integrated = f_per_L * L * w

            total_force += f_integrated
            F1 += (1.0 - s) * f_integrated
            F2 += s * f_integrated

        return {'member_idx': member_idx, 'member_name': member['name'],
                'length': L, 'submerged_length': submerged_length,
                'F_node1': F1, 'F_node2': F2, 'F_total': total_force}

    def compute_all_morison_forces(self, t: float = 0.0, n_gauss: int = 15) -> Dict:
        """Compute Morison forces for all members."""
        nodal_forces: Dict[str, np.ndarray] = {
            name: np.zeros(6) for name in self.structure.nodes.keys()}
        member_results: List[Dict] = []

        for i, member in enumerate(self.structure.members):
            result = self.compute_member_morison_forces(i, t, n_gauss)
            nodal_forces[member['node1']][:3] += result['F_node1']
            nodal_forces[member['node2']][:3] += result['F_node2']
            member_results.append(result)

        total_Fx = sum(r['F_total'][0] for r in member_results)
        total_Fy = sum(r['F_total'][1] for r in member_results)
        total_Fz = sum(r['F_total'][2] for r in member_results)

        return {'nodal_forces': nodal_forces, 'member_results': member_results,
                'total_force': np.array([total_Fx, total_Fy, total_Fz])}


# =============================================================================
# INTEGRATED ANALYSIS CLASS
# =============================================================================
class JacketAnalysis:
    """Complete structural analysis of a 3-legged OSP jacket."""

    def __init__(self, params: Dict):
        """Initialize with parameter dictionary."""
        self.params = params
        
        # Extract parameters
        self.H = params.get('H', 17.038)
        self.T = params.get('T', 9.4)
        self.d = params.get('d', 50.0)
        self.U_c = params.get('U_c', 1.7)
        self.wave_direction = params.get('wave_direction', 38.0)
        self.wave_model = params.get('wave_model', 'Fenton')
        self.N_harmonics = params.get('N_harmonics', 10)
        
        self.F_axial = params.get('F_axial', 25100.0)
        self.F_shear = params.get('F_shear', 2900.0)
        self.M_moment = params.get('M_moment', 0.0)
        self.M_torsion = params.get('M_torsion', 0.0)
        
        self.Cd = params.get('Cd', 0.7)
        self.Cm = params.get('Cm', 2.0)
        self.include_self_weight = params.get('include_self_weight', True)
        self.n_gauss = params.get('n_gauss', 15)
        
        self.E = params.get('E', 210000)
        self.nu = params.get('nu', 0.3)
        self.fy = params.get('fy', 355)
        self.rho_steel = params.get('rho_steel', 7850)
        self.rho_water = params.get('rho_water', 1025)
        self.g = params.get('g', 9.81)
        
        self.z_water_ref = params.get('z_water_ref', 47.0)
        self.D_leg = params.get('D_leg', 2000.0)
        self.t_leg = params.get('t_leg', 75.0)
        self.D_brace = params.get('D_brace', 800.0)
        self.t_brace = params.get('t_brace', 30.0)

        self.log_messages = []

    def log(self, msg):
        """Add message to log."""
        self.log_messages.append(msg)

    def initialize(self):
        """Initialize all components."""
        self.log("Creating jacket structure...")
        self.structure = JacketStructure(
            z_water_ref=self.z_water_ref,
            D_leg=self.D_leg,
            t_leg=self.t_leg,
            D_brace=self.D_brace,
            t_brace=self.t_brace,
            rho_steel=self.rho_steel
        )

        self.log("Creating wave model...")
        self.wave = RaschiiWave(
            self.H, self.T, self.d, self.U_c,
            self.wave_model, self.N_harmonics
        )

        self.morison = MorisonCalculator(
            self.structure, self.wave,
            self.wave_direction, self.Cd, self.Cm, self.rho_water
        )

        self.log("Creating FEM solver...")
        self.fem = FEMSolver(self.structure, self.E, self.nu)

    def apply_interface_loads(self):
        """Apply topside interface loads."""
        top_nodes = self.structure.get_top_nodes()
        n_legs = len(top_nodes)

        F_axial_N = self.F_axial * 1000.0
        F_shear_N = self.F_shear * 1000.0
        M_moment_Nmm = self.M_moment * 1e6
        M_torsion_Nmm = self.M_torsion * 1e6

        F_axial_per_leg = F_axial_N / n_legs
        theta = np.deg2rad(90.0 - self.wave_direction)
        F_shear_x = F_shear_N * np.cos(theta) / n_legs
        F_shear_y = F_shear_N * np.sin(theta) / n_legs

        for node in top_nodes:
            force = np.array([
                F_shear_x, F_shear_y, -F_axial_per_leg,
                M_torsion_Nmm / n_legs, M_moment_Nmm / n_legs, 0.0
            ])
            self.fem.apply_nodal_force(node, force)

        self.log(f"Applied interface loads: Axial={self.F_axial} kN, Shear={self.F_shear} kN, "
                f"Moment={self.M_moment} kNm, Torsion={self.M_torsion} kNm")

    def apply_morison_loads(self, t: float = 0.0):
        """Apply Morison wave loads."""
        morison_results = self.morison.compute_all_morison_forces(t, self.n_gauss)

        for node_name, force in morison_results['nodal_forces'].items():
            force_vector = np.zeros(6)
            force_vector[:3] = force[:3]
            self.fem.apply_nodal_force(node_name, force_vector)

        total = morison_results['total_force']
        self.log(f"Applied Morison loads at t={t:.2f}s: "
                f"Fx={total[0]/1000.0:.1f} kN, Fy={total[1]/1000.0:.1f} kN, "
                f"Fz={total[2]/1000.0:.1f} kN")

        return morison_results

    def apply_self_weight(self):
        """Apply self-weight of all members."""
        if not self.include_self_weight:
            return

        total_weight = 0.0
        for member in self.structure.members:
            section = member['section']
            geom = self.structure.get_member_geometry(member)
            w = section.mass_per_m * self.g
            member_weight = w * geom['L']
            total_weight += member_weight

            F_weight = member_weight / 2.0
            idx1 = self.structure.node_index[member['node1']]
            idx2 = self.structure.node_index[member['node2']]
            self.fem.F_global[6*idx1 + 2] -= F_weight
            self.fem.F_global[6*idx2 + 2] -= F_weight

        self.log(f"Applied self-weight: Total = {total_weight/1000.0:.1f} kN")

    def run_analysis(self, t: float = 0.0) -> Dict:
        """Run complete static analysis."""
        self.log("\n" + "="*60)
        self.log("JACKET STRUCTURAL ANALYSIS")
        self.log("="*60)

        self.log(f"\nStructure: {self.structure.n_nodes} nodes, "
                f"{self.structure.n_members} members")

        self.fem.F_global = np.zeros(self.fem.n_dof)

        self.log("\n" + "-"*40)
        self.log("APPLYING LOADS")
        self.log("-"*40)

        self.apply_interface_loads()
        morison_results = self.apply_morison_loads(t)
        self.apply_self_weight()

        bottom_nodes = self.structure.get_bottom_nodes()
        self.fem.apply_boundary_conditions(bottom_nodes)
        self.log(f"Fixed nodes: {bottom_nodes}")

        self.log("\n" + "-"*40)
        self.log("SOLVING FEM SYSTEM")
        self.log("-"*40)

        U = self.fem.solve()
        reactions = self.fem.get_reactions()
        internal_forces = self.fem.get_member_internal_forces(self.fy)

        self.log("\n" + "="*60)
        self.log("RESULTS SUMMARY")
        self.log("="*60)

        self.log("\nREACTION FORCES:")
        total_Rx = total_Ry = total_Rz = 0.0
        for node, R in reactions.items():
            self.log(f"  {node}: Rx={R[0]/1000.0:.1f} kN, Ry={R[1]/1000.0:.1f} kN, "
                    f"Rz={R[2]/1000.0:.1f} kN")
            total_Rx += R[0]
            total_Ry += R[1]
            total_Rz += R[2]

        self.log(f"\n  Total: Rx={total_Rx/1000.0:.1f} kN, Ry={total_Ry/1000.0:.1f} kN, "
                f"Rz={total_Rz/1000.0:.1f} kN")

        self.log("\nMAXIMUM DISPLACEMENTS:")
        max_disp = 0.0
        max_disp_node = ""
        for i, node in enumerate(self.structure.node_list):
            disp = np.linalg.norm(U[6*i:6*i+3])
            if disp > max_disp:
                max_disp = disp
                max_disp_node = node
        self.log(f"  Maximum displacement: {max_disp:.2f} mm at node {max_disp_node}")

        self.log("\nCRITICAL MEMBERS (Top 10 by utilization):")
        sorted_members = sorted(internal_forces, key=lambda x: x['utilization'], reverse=True)

        for m in sorted_members[:10]:
            self.log(f"  {m['member']:<25} VM={m['von_mises_max_MPa']:.1f} MPa, "
                    f"Util={m['utilization']:.2%}")

        max_util = max(m['utilization'] for m in internal_forces)
        if max_util > 1.0:
            self.log(f"\n  *** WARNING: Max utilization {max_util:.2%} exceeds yield! ***")
        else:
            self.log(f"\n  Maximum utilization: {max_util:.2%} (< 100%, OK)")

        return {
            'displacements': U,
            'reactions': reactions,
            'internal_forces': internal_forces,
            'morison_results': morison_results,
            'max_utilization': max_util,
            'log': '\n'.join(self.log_messages)
        }

    def run_time_history(self, t_array: np.ndarray, progress_callback=None) -> pd.DataFrame:
        """Run quasi-static time-history analysis."""
        results = []

        for i, t in enumerate(t_array):
            if progress_callback:
                progress_callback(i + 1, len(t_array))

            self.fem.F_global = np.zeros(self.fem.n_dof)
            self.apply_interface_loads()
            morison_res = self.morison.compute_all_morison_forces(t, self.n_gauss)

            for node_name, force in morison_res['nodal_forces'].items():
                force_vector = np.zeros(6)
                force_vector[:3] = force[:3]
                self.fem.apply_nodal_force(node_name, force_vector)

            self.apply_self_weight()
            self.fem.apply_boundary_conditions(self.structure.get_bottom_nodes())
            self.fem.solve()

            internal_forces = self.fem.get_member_internal_forces(self.fy)
            max_util = max(m['utilization'] for m in internal_forces)

            leg_forces = [m for m in internal_forces if m['type'] == 'leg']
            max_axial_leg = max(m['Fx_max_kN'] for m in leg_forces)
            max_shear_leg = max(np.sqrt(m['Fy_max_kN']**2 + m['Fz_max_kN']**2) for m in leg_forces)
            max_moment_leg = max(np.sqrt(m['My_max_kNm']**2 + m['Mz_max_kNm']**2) for m in leg_forces)

            results.append({
                't': t,
                'Morison_Fx_kN': morison_res['total_force'][0] / 1000.0,
                'Morison_Fy_kN': morison_res['total_force'][1] / 1000.0,
                'Morison_Fz_kN': morison_res['total_force'][2] / 1000.0,
                'Max_Axial_Leg_kN': max_axial_leg,
                'Max_Shear_Leg_kN': max_shear_leg,
                'Max_Moment_Leg_kNm': max_moment_leg,
                'Max_Utilization': max_util
            })

        return pd.DataFrame(results)


# =============================================================================
# GUI APPLICATION CLASS
# =============================================================================
class JacketAnalysisGUI:
    """Interactive GUI for Jacket Analysis."""

    def __init__(self, root):
        self.root = root
        self.root.title("3-Legged OSP Jacket Structural Analysis - Interactive GUI v7")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.params = {}
        self.analysis = None
        self.results = None
        
        self.create_widgets()
        self.load_default_values()

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.tab_wave = ttk.Frame(self.notebook)
        self.tab_material = ttk.Frame(self.notebook)
        self.tab_section = ttk.Frame(self.notebook)
        self.tab_loads = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)
        self.tab_results = ttk.Frame(self.notebook)
        self.tab_info = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_wave, text="Wave Parameters")
        self.notebook.add(self.tab_material, text="Material Properties")
        self.notebook.add(self.tab_section, text="Section Dimensions")
        self.notebook.add(self.tab_loads, text="Interface Loads")
        self.notebook.add(self.tab_analysis, text="Run Analysis")
        self.notebook.add(self.tab_results, text="Results")
        self.notebook.add(self.tab_info, text="Info & Assumptions")

        self._create_wave_tab()
        self._create_material_tab()
        self._create_section_tab()
        self._create_loads_tab()
        self._create_analysis_tab()
        self._create_results_tab()
        self._create_info_tab()

    def _create_labeled_entry(self, parent, label, row, default="", width=15, unit=""):
        """Helper to create a labeled entry field."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='e', padx=5, pady=2)
        entry = ttk.Entry(parent, width=width)
        entry.grid(row=row, column=1, sticky='w', padx=5, pady=2)
        entry.insert(0, str(default))
        if unit:
            ttk.Label(parent, text=unit).grid(row=row, column=2, sticky='w', padx=2, pady=2)
        return entry

    def _create_wave_tab(self):
        """Create wave parameters tab."""
        # Raschii status frame
        status_frame = ttk.Frame(self.tab_wave)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        if RASCHII_AVAILABLE:
            status_text = "✓ Raschii Library: INSTALLED - Nonlinear wave theories available (Fenton, Stokes, Airy)"
            status_color = "green"
        else:
            status_text = "⚠ Raschii Library: NOT INSTALLED - Using fallback linear Airy wave theory"
            status_color = "orange"
        
        status_label = ttk.Label(status_frame, text=status_text, font=('TkDefaultFont', 9, 'bold'))
        status_label.pack(side=tk.LEFT, padx=5)
        
        if not RASCHII_AVAILABLE:
            def install_raschii():
                """Attempt to install raschii."""
                import subprocess
                try:
                    messagebox.showinfo("Installing", "Installing raschii library...\nThis may take a moment.")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "raschii", "-q"])
                    messagebox.showinfo("Success", "Raschii installed successfully!\nPlease restart the application.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to install raschii:\n{e}\n\nTry manually: pip install raschii")
            
            ttk.Button(status_frame, text="Install Raschii", command=install_raschii).pack(side=tk.LEFT, padx=10)
        
        frame = ttk.LabelFrame(self.tab_wave, text="Wave and Environmental Parameters", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Wave parameters
        self.entry_H = self._create_labeled_entry(frame, "Wave Height (H):", 0, width=12, unit="m")
        self.entry_T = self._create_labeled_entry(frame, "Wave Period (T):", 1, width=12, unit="s")
        self.entry_d = self._create_labeled_entry(frame, "Water Depth (d):", 2, width=12, unit="m")
        self.entry_Uc = self._create_labeled_entry(frame, "Current Velocity (U_c):", 3, width=12, unit="m/s")
        self.entry_wave_dir = self._create_labeled_entry(frame, "Wave Direction:", 4, width=12, unit="° from North")

        # Wave model selection
        ttk.Label(frame, text="Wave Model:").grid(row=5, column=0, sticky='e', padx=5, pady=2)
        self.wave_model_var = tk.StringVar()
        self.combo_wave_model = ttk.Combobox(frame, textvariable=self.wave_model_var, width=12)
        if RASCHII_AVAILABLE:
            self.combo_wave_model['values'] = ('Fenton', 'Stokes', 'Airy', 'auto')
        else:
            self.combo_wave_model['values'] = ('Airy (fallback)',)
        self.combo_wave_model.grid(row=5, column=1, sticky='w', padx=5, pady=2)

        self.entry_N_harm = self._create_labeled_entry(frame, "N Harmonics:", 6, width=12, unit="(Fenton/Stokes)")

        # Morison coefficients
        ttk.Separator(frame, orient='horizontal').grid(row=7, column=0, columnspan=3, sticky='ew', pady=10)
        ttk.Label(frame, text="Morison Coefficients", font=('TkDefaultFont', 10, 'bold')).grid(
            row=8, column=0, columnspan=3, sticky='w', pady=5)

        self.entry_Cd = self._create_labeled_entry(frame, "Drag Coefficient (Cd):", 9, width=12, unit="[-]")
        self.entry_Cm = self._create_labeled_entry(frame, "Inertia Coefficient (Cm):", 10, width=12, unit="[-]")

        # Validation button
        ttk.Button(frame, text="Validate Wave Parameters", command=self.validate_wave).grid(
            row=11, column=0, columnspan=3, pady=20)

        # Validation result
        self.wave_validation_text = tk.Text(frame, height=10, width=60, state='disabled')
        self.wave_validation_text.grid(row=12, column=0, columnspan=3, pady=5)

    def _create_material_tab(self):
        """Create material properties tab."""
        frame = ttk.LabelFrame(self.tab_material, text="Material Properties - Steel", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.entry_E = self._create_labeled_entry(frame, "Young's Modulus (E):", 0, width=12, unit="N/mm² (MPa)")
        self.entry_nu = self._create_labeled_entry(frame, "Poisson's Ratio (ν):", 1, width=12, unit="[-]")
        self.entry_fy = self._create_labeled_entry(frame, "Yield Strength (fy):", 2, width=12, unit="N/mm² (MPa)")
        self.entry_rho_steel = self._create_labeled_entry(frame, "Steel Density:", 3, width=12, unit="kg/m³")

        ttk.Separator(frame, orient='horizontal').grid(row=4, column=0, columnspan=3, sticky='ew', pady=10)
        ttk.Label(frame, text="Environmental Properties", font=('TkDefaultFont', 10, 'bold')).grid(
            row=5, column=0, columnspan=3, sticky='w', pady=5)

        self.entry_rho_water = self._create_labeled_entry(frame, "Seawater Density:", 6, width=12, unit="kg/m³")
        self.entry_g = self._create_labeled_entry(frame, "Gravity (g):", 7, width=12, unit="m/s²")

        # Info box
        info_text = """
Material Property Notes:
------------------------
• Young's Modulus: Typical steel = 200,000 - 210,000 MPa
• Poisson's Ratio: Typical steel = 0.3
• Yield Strength: S235=235 MPa, S275=275 MPa, S355=355 MPa, S460=460 MPa
• Steel Density: Typical = 7,850 kg/m³
• Seawater Density: Typical = 1,025 kg/m³ (varies with salinity)
        """
        text_widget = tk.Text(frame, height=10, width=60, state='normal')
        text_widget.grid(row=8, column=0, columnspan=3, pady=10)
        text_widget.insert('1.0', info_text)
        text_widget.config(state='disabled')

    def _create_section_tab(self):
        """Create section dimensions tab."""
        frame = ttk.LabelFrame(self.tab_section, text="Tubular Section Dimensions", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Leg (Column) Section", font=('TkDefaultFont', 10, 'bold')).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=5)
        self.entry_D_leg = self._create_labeled_entry(frame, "Outer Diameter (D_leg):", 1, width=12, unit="mm")
        self.entry_t_leg = self._create_labeled_entry(frame, "Wall Thickness (t_leg):", 2, width=12, unit="mm")

        ttk.Separator(frame, orient='horizontal').grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)

        ttk.Label(frame, text="Brace Section", font=('TkDefaultFont', 10, 'bold')).grid(
            row=4, column=0, columnspan=3, sticky='w', pady=5)
        self.entry_D_brace = self._create_labeled_entry(frame, "Outer Diameter (D_brace):", 5, width=12, unit="mm")
        self.entry_t_brace = self._create_labeled_entry(frame, "Wall Thickness (t_brace):", 6, width=12, unit="mm")

        ttk.Separator(frame, orient='horizontal').grid(row=7, column=0, columnspan=3, sticky='ew', pady=10)

        ttk.Label(frame, text="Structure Reference", font=('TkDefaultFont', 10, 'bold')).grid(
            row=8, column=0, columnspan=3, sticky='w', pady=5)
        self.entry_z_water_ref = self._create_labeled_entry(frame, "MWL Reference Depth:", 9, width=12, unit="m")

        # Calculate button
        ttk.Button(frame, text="Calculate Section Properties", command=self.show_section_properties).grid(
            row=10, column=0, columnspan=3, pady=20)

        self.section_props_text = tk.Text(frame, height=15, width=60, state='disabled')
        self.section_props_text.grid(row=11, column=0, columnspan=3, pady=5)

    def _create_loads_tab(self):
        """Create interface loads tab."""
        frame = ttk.LabelFrame(self.tab_loads, text="Topside Interface Loads", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Forces at Interface Level", font=('TkDefaultFont', 10, 'bold')).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=5)

        self.entry_F_axial = self._create_labeled_entry(frame, "Axial Force (compression):", 1, width=12, unit="kN")
        self.entry_F_shear = self._create_labeled_entry(frame, "Horizontal Shear:", 2, width=12, unit="kN")

        ttk.Separator(frame, orient='horizontal').grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)

        ttk.Label(frame, text="Moments at Interface Level", font=('TkDefaultFont', 10, 'bold')).grid(
            row=4, column=0, columnspan=3, sticky='w', pady=5)

        self.entry_M_moment = self._create_labeled_entry(frame, "Overturning Moment:", 5, width=12, unit="kNm")
        self.entry_M_torsion = self._create_labeled_entry(frame, "Torsional Moment:", 6, width=12, unit="kNm")

        # Self-weight option
        ttk.Separator(frame, orient='horizontal').grid(row=7, column=0, columnspan=3, sticky='ew', pady=10)
        self.self_weight_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Include Self-Weight", variable=self.self_weight_var).grid(
            row=8, column=0, columnspan=3, sticky='w', pady=5)

        # Info
        info_text = """
Interface Load Notes:
---------------------
• Axial Force: Compression from topside weight and equipment
• Horizontal Shear: Wind/wave induced horizontal force at topside
• Overturning Moment: Global moment about horizontal axis (My)
• Torsional Moment: Global moment about vertical axis (Mx)

These loads are distributed equally among the three leg top nodes.
The shear force is applied in the wave direction.

NOTE: Moment and Torsion ARE implemented in this code!
Currently set to 0 but can be modified as needed.
        """
        text_widget = tk.Text(frame, height=12, width=60, state='normal')
        text_widget.grid(row=9, column=0, columnspan=3, pady=10)
        text_widget.insert('1.0', info_text)
        text_widget.config(state='disabled')

    def _create_analysis_tab(self):
        """Create analysis control tab."""
        frame = ttk.LabelFrame(self.tab_analysis, text="Analysis Control", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Analysis options
        ttk.Label(frame, text="Analysis Options", font=('TkDefaultFont', 10, 'bold')).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=5)

        self.entry_n_gauss = self._create_labeled_entry(frame, "Gauss Integration Points:", 1, width=12, unit="[-]")

        ttk.Separator(frame, orient='horizontal').grid(row=2, column=0, columnspan=3, sticky='ew', pady=10)

        # Static analysis
        ttk.Label(frame, text="Static Analysis (Single Time)", font=('TkDefaultFont', 10, 'bold')).grid(
            row=3, column=0, columnspan=3, sticky='w', pady=5)
        
        self.entry_t_static = self._create_labeled_entry(frame, "Time (t):", 4, "0.0", width=12, unit="s")
        ttk.Button(frame, text="Run Static Analysis", command=self.run_static_analysis).grid(
            row=5, column=0, columnspan=3, pady=10)

        ttk.Separator(frame, orient='horizontal').grid(row=6, column=0, columnspan=3, sticky='ew', pady=10)

        # Time history analysis
        ttk.Label(frame, text="Time History Analysis", font=('TkDefaultFont', 10, 'bold')).grid(
            row=7, column=0, columnspan=3, sticky='w', pady=5)

        self.entry_n_periods = self._create_labeled_entry(frame, "Number of Periods:", 8, "3", width=12, unit="[-]")
        self.entry_n_steps = self._create_labeled_entry(frame, "Time Steps:", 9, "31", width=12, unit="[-]")
        ttk.Button(frame, text="Run Time History Analysis", command=self.run_time_history).grid(
            row=10, column=0, columnspan=3, pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=11, column=0, columnspan=3, sticky='ew', padx=20, pady=10)

        # Status label
        self.status_label = ttk.Label(frame, text="Ready")
        self.status_label.grid(row=12, column=0, columnspan=3, pady=5)

        # Log output
        ttk.Label(frame, text="Analysis Log:").grid(row=13, column=0, sticky='w', padx=5, pady=5)
        self.log_text = tk.Text(frame, height=15, width=80)
        self.log_text.grid(row=14, column=0, columnspan=3, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.log_text.yview)
        scrollbar.grid(row=14, column=3, sticky='ns')
        self.log_text.config(yscrollcommand=scrollbar.set)

    def _create_results_tab(self):
        """Create results display tab."""
        frame = ttk.Frame(self.tab_results, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Buttons for different result views
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Show Summary", command=self.show_summary).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Plot Structure 3D", command=self.plot_structure).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Plot Internal Forces", command=self.plot_forces).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Export to CSV", command=self.export_results).pack(side=tk.LEFT, padx=5)

        # Results text area
        self.results_text = tk.Text(frame, height=35, width=100)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=10)

        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

    def _create_info_tab(self):
        """Create info and assumptions tab."""
        frame = ttk.Frame(self.tab_info, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        info_text = tk.Text(frame, height=45, width=100, wrap=tk.WORD)
        info_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        info_text.config(yscrollcommand=scrollbar.set)

        assumptions_text = """
================================================================================
                    CODE ASSUMPTIONS AND LIMITATIONS
================================================================================

1. STRUCTURAL ASSUMPTIONS
   ----------------------
   • Cross-section type: Circular tubular (pipe) sections
   • Section theory: THIN-WALL assumption (D/t > 10 recommended)
   • Beam theory: Timoshenko beam (includes shear deformation)
   • Material behavior: Linear elastic (no plasticity, no buckling)
   • Analysis type: Static or quasi-static (no dynamic effects)
   • Support conditions: Fixed at mudline (all 6 DOF restrained)

2. THIN-WALL CROSS-SECTION NOTES
   -----------------------------
   The code uses thin-walled tube formulas:
   • Area: A = π/4 * (D² - d²) ≈ π * D * t (for D >> t)
   • Second moment: I = π/64 * (D⁴ - d⁴)
   • Shear area: A_shear ≈ 0.5 * A
   
   VALIDITY: D/t ratio should be > 10 for thin-wall assumption
   Current leg: D/t = 2000/75 = 26.7 ✓
   Current brace: D/t = 800/30 = 26.7 ✓

3. MORISON EQUATION ASSUMPTIONS
   ----------------------------
   • Slender member assumption: D/L < 0.2 (member diameter << wavelength)
   • Only perpendicular velocity/acceleration components used
   • No end effects, no marine growth, no interference between members
   • Drag and inertia coefficients (Cd, Cm) are constant along member
   • Recommended values: Cd = 0.6-1.2, Cm = 1.5-2.0

4. WAVE THEORY LIMITATIONS (RASCHII LIBRARY)
   -----------------------------------------
   Model Orders:
   • AiryWave: 1st order (linear) - valid for small amplitude waves
   • StokesWave: Maximum N = 5 (5th order Stokes theory)
   • FentonWave: Typically N = 5-20 harmonics

   BREAKING WAVE LIMITS:
   • Maximum steepness: H/L < 0.142 (Miche criterion)
   • Deep water: H/(g*T²/(2π)) < 0.142
   • Shallow water: H/d < 0.78
   
   ERROR CONDITIONS:
   • Wave too steep → convergence failure
   • Invalid depth (d ≤ 0) → error
   • H > d → physically impossible
   
   RECOMMENDED RANGES:
   • Wave height: 0.5m < H < 30m
   • Wave period: 3s < T < 20s
   • Water depth: 10m < d < 300m
   • Wave steepness: H/L < 0.10 (for numerical stability)

5. INTERFACE LOADS IMPLEMENTATION
   ------------------------------
   The code DOES implement moment and torsion at interface:
   • F_axial: Distributed equally to 3 leg tops as vertical compression
   • F_shear: Applied in wave direction, distributed to 3 legs
   • M_moment: Overturning moment (about Y-axis), distributed to 3 legs
   • M_torsion: Torsional moment (about X-axis), distributed to 3 legs

6. COORDINATE SYSTEM
   -----------------
   • X: East (+)
   • Y: North (+)
   • Z: Up (+), z=0 at Mean Water Level (MWL)
   • Wave direction: Degrees from North, clockwise positive

7. STRESS CALCULATION
   ------------------
   • Von Mises stress evaluated at 8 points around circumference
   • Maximum von Mises used for utilization ratio
   • Utilization = σ_vm / f_y
   • NO buckling check - only elastic stress capacity

8. FEM FORMULATION
   ---------------
   • 3D beam elements with 12 DOF (6 per node)
   • DOFs: u, v, w (translations), θx, θy, θz (rotations)
   • Timoshenko beam formulation with shear correction
   • Global stiffness matrix assembly by direct stiffness method

9. UNITS CONVENTION
   ----------------
   Input:
   • Lengths: m (structure), mm (sections)
   • Forces: kN
   • Moments: kNm
   • Stresses: MPa (N/mm²)
   
   Internal calculations:
   • Lengths: mm
   • Forces: N
   • Moments: N·mm
   • Stresses: N/mm² = MPa

10. SIMPLIFICATIONS
    ---------------
    • Fixed geometry (node positions hardcoded for 3-leg jacket)
    • No joint flexibility (rigid connections)
    • No P-delta effects
    • No fatigue analysis
    • No corrosion allowance
    • No safety factors applied

================================================================================
                         RASCHII LIBRARY DETAILED NOTES
================================================================================

The raschii library (if available) provides nonlinear wave solutions:

FENTON WAVE:
• Higher-order stream function wave theory
• Best for steep waves in any water depth
• N parameter: number of Fourier components (5-20)
• Computationally intensive but most accurate

STOKES WAVE:
• Perturbation expansion solution
• Maximum N = 5 (5th order)
• Good for intermediate steepness waves
• Faster than Fenton

AIRY WAVE (Linear):
• First-order solution
• Valid only for small amplitude (H << L)
• Fastest computation
• Used as fallback if raschii unavailable

COMMON ERRORS AND SOLUTIONS:
• "Wave too steep": Reduce H or increase T
• "Convergence failed": Try different wave model
• "Invalid input": Check H > 0, T > 0, d > 0

================================================================================
"""
        info_text.insert('1.0', assumptions_text)
        info_text.config(state='disabled')

    def load_default_values(self):
        """Load default parameter values into entries."""
        # Wave parameters
        self.entry_H.delete(0, tk.END)
        self.entry_H.insert(0, str(DEFAULT_PARAMS['H']))
        self.entry_T.delete(0, tk.END)
        self.entry_T.insert(0, str(DEFAULT_PARAMS['T']))
        self.entry_d.delete(0, tk.END)
        self.entry_d.insert(0, str(DEFAULT_PARAMS['d']))
        self.entry_Uc.delete(0, tk.END)
        self.entry_Uc.insert(0, str(DEFAULT_PARAMS['U_c']))
        self.entry_wave_dir.delete(0, tk.END)
        self.entry_wave_dir.insert(0, str(DEFAULT_PARAMS['wave_direction']))
        self.wave_model_var.set('Fenton' if RASCHII_AVAILABLE else 'Airy (fallback)')
        self.entry_N_harm.delete(0, tk.END)
        self.entry_N_harm.insert(0, str(DEFAULT_PARAMS['N_harmonics']))
        self.entry_Cd.delete(0, tk.END)
        self.entry_Cd.insert(0, str(DEFAULT_PARAMS['Cd']))
        self.entry_Cm.delete(0, tk.END)
        self.entry_Cm.insert(0, str(DEFAULT_PARAMS['Cm']))

        # Material properties
        self.entry_E.delete(0, tk.END)
        self.entry_E.insert(0, str(DEFAULT_PARAMS['E']))
        self.entry_nu.delete(0, tk.END)
        self.entry_nu.insert(0, str(DEFAULT_PARAMS['nu']))
        self.entry_fy.delete(0, tk.END)
        self.entry_fy.insert(0, str(DEFAULT_PARAMS['fy']))
        self.entry_rho_steel.delete(0, tk.END)
        self.entry_rho_steel.insert(0, str(DEFAULT_PARAMS['rho_steel']))
        self.entry_rho_water.delete(0, tk.END)
        self.entry_rho_water.insert(0, str(DEFAULT_PARAMS['rho_water']))
        self.entry_g.delete(0, tk.END)
        self.entry_g.insert(0, str(DEFAULT_PARAMS['g']))

        # Section dimensions
        self.entry_D_leg.delete(0, tk.END)
        self.entry_D_leg.insert(0, str(DEFAULT_PARAMS['D_leg']))
        self.entry_t_leg.delete(0, tk.END)
        self.entry_t_leg.insert(0, str(DEFAULT_PARAMS['t_leg']))
        self.entry_D_brace.delete(0, tk.END)
        self.entry_D_brace.insert(0, str(DEFAULT_PARAMS['D_brace']))
        self.entry_t_brace.delete(0, tk.END)
        self.entry_t_brace.insert(0, str(DEFAULT_PARAMS['t_brace']))
        self.entry_z_water_ref.delete(0, tk.END)
        self.entry_z_water_ref.insert(0, str(DEFAULT_PARAMS['z_water_ref']))

        # Interface loads
        self.entry_F_axial.delete(0, tk.END)
        self.entry_F_axial.insert(0, str(DEFAULT_PARAMS['F_axial']))
        self.entry_F_shear.delete(0, tk.END)
        self.entry_F_shear.insert(0, str(DEFAULT_PARAMS['F_shear']))
        self.entry_M_moment.delete(0, tk.END)
        self.entry_M_moment.insert(0, str(DEFAULT_PARAMS['M_moment']))
        self.entry_M_torsion.delete(0, tk.END)
        self.entry_M_torsion.insert(0, str(DEFAULT_PARAMS['M_torsion']))

        # Analysis options
        self.entry_n_gauss.delete(0, tk.END)
        self.entry_n_gauss.insert(0, str(DEFAULT_PARAMS['n_gauss']))

    def get_parameters(self) -> Dict:
        """Collect all parameters from GUI entries."""
        try:
            params = {
                # Wave parameters
                'H': float(self.entry_H.get()),
                'T': float(self.entry_T.get()),
                'd': float(self.entry_d.get()),
                'U_c': float(self.entry_Uc.get()),
                'wave_direction': float(self.entry_wave_dir.get()),
                'wave_model': self.wave_model_var.get().split()[0],
                'N_harmonics': int(self.entry_N_harm.get()),
                'Cd': float(self.entry_Cd.get()),
                'Cm': float(self.entry_Cm.get()),

                # Material properties
                'E': float(self.entry_E.get()),
                'nu': float(self.entry_nu.get()),
                'fy': float(self.entry_fy.get()),
                'rho_steel': float(self.entry_rho_steel.get()),
                'rho_water': float(self.entry_rho_water.get()),
                'g': float(self.entry_g.get()),

                # Section dimensions
                'D_leg': float(self.entry_D_leg.get()),
                't_leg': float(self.entry_t_leg.get()),
                'D_brace': float(self.entry_D_brace.get()),
                't_brace': float(self.entry_t_brace.get()),
                'z_water_ref': float(self.entry_z_water_ref.get()),

                # Interface loads
                'F_axial': float(self.entry_F_axial.get()),
                'F_shear': float(self.entry_F_shear.get()),
                'M_moment': float(self.entry_M_moment.get()),
                'M_torsion': float(self.entry_M_torsion.get()),

                # Analysis options
                'include_self_weight': self.self_weight_var.get(),
                'n_gauss': int(self.entry_n_gauss.get()),
            }
            return params
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input value: {e}")
            return None

    def validate_wave(self):
        """Validate wave parameters and display results."""
        try:
            H = float(self.entry_H.get())
            T = float(self.entry_T.get())
            d = float(self.entry_d.get())

            is_valid, message = validate_wave_parameters(H, T, d)
            
            H_max = calculate_max_wave_height(T, d)
            message += f"\n\nMaximum allowable wave height for T={T}s, d={d}m:\n  H_max = {H_max:.2f} m"

            self.wave_validation_text.config(state='normal')
            self.wave_validation_text.delete('1.0', tk.END)
            self.wave_validation_text.insert('1.0', message)
            self.wave_validation_text.config(state='disabled')

            if not is_valid:
                messagebox.showwarning("Wave Validation", "Wave parameters are INVALID!\nSee details in validation box.")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")

    def show_section_properties(self):
        """Calculate and display section properties."""
        try:
            D_leg = float(self.entry_D_leg.get())
            t_leg = float(self.entry_t_leg.get())
            D_brace = float(self.entry_D_brace.get())
            t_brace = float(self.entry_t_brace.get())
            rho_steel = float(self.entry_rho_steel.get())

            leg = TubularSection(D_leg, t_leg, "Leg", rho_steel)
            brace = TubularSection(D_brace, t_brace, "Brace", rho_steel)

            text = f"""
LEG SECTION (D={D_leg}mm, t={t_leg}mm):
  D/t ratio = {leg.D_t_ratio:.1f} {'✓ OK' if leg.D_t_ratio > 10 else '⚠ THICK WALL'}
  Cross-sectional area A = {leg.Ax_cm2:.2f} cm²
  Second moment of area I = {leg.Iy_cm4:.2f} cm⁴
  Polar moment J = {leg.Ix_cm4:.2f} cm⁴
  Section modulus W = {leg.Wy_cm3:.2f} cm³
  Mass per meter = {leg.mass_per_m:.2f} kg/m

BRACE SECTION (D={D_brace}mm, t={t_brace}mm):
  D/t ratio = {brace.D_t_ratio:.1f} {'✓ OK' if brace.D_t_ratio > 10 else '⚠ THICK WALL'}
  Cross-sectional area A = {brace.Ax_cm2:.2f} cm²
  Second moment of area I = {brace.Iy_cm4:.2f} cm⁴
  Polar moment J = {brace.Ix_cm4:.2f} cm⁴
  Section modulus W = {brace.Wy_cm3:.2f} cm³
  Mass per meter = {brace.mass_per_m:.2f} kg/m

NOTE: Thin-wall assumption valid when D/t > 10
"""
            self.section_props_text.config(state='normal')
            self.section_props_text.delete('1.0', tk.END)
            self.section_props_text.insert('1.0', text)
            self.section_props_text.config(state='disabled')
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")

    def run_static_analysis(self):
        """Run static analysis."""
        params = self.get_parameters()
        if params is None:
            return

        # Validate wave parameters
        is_valid, msg = validate_wave_parameters(params['H'], params['T'], params['d'], False)
        if not is_valid:
            messagebox.showerror("Wave Error", f"Invalid wave parameters:\n{msg}")
            return

        self.status_label.config(text="Running static analysis...")
        self.log_text.delete('1.0', tk.END)
        self.root.update()

        try:
            t_static = float(self.entry_t_static.get())
            
            self.analysis = JacketAnalysis(params)
            self.analysis.initialize()
            self.results = self.analysis.run_analysis(t_static)

            self.log_text.insert('1.0', self.results['log'])
            self.status_label.config(text="Analysis complete!")

            messagebox.showinfo("Complete", "Static analysis completed successfully!")
            self.notebook.select(self.tab_results)
            self.show_summary()

        except Exception as e:
            self.status_label.config(text="Error!")
            messagebox.showerror("Analysis Error", f"Error during analysis:\n{str(e)}")
            import traceback
            self.log_text.insert('1.0', traceback.format_exc())

    def run_time_history(self):
        """Run time history analysis."""
        params = self.get_parameters()
        if params is None:
            return

        is_valid, msg = validate_wave_parameters(params['H'], params['T'], params['d'], False)
        if not is_valid:
            messagebox.showerror("Wave Error", f"Invalid wave parameters:\n{msg}")
            return

        self.status_label.config(text="Running time history analysis...")
        self.log_text.delete('1.0', tk.END)
        self.progress_var.set(0)
        self.root.update()

        try:
            n_periods = float(self.entry_n_periods.get())
            n_steps = int(self.entry_n_steps.get())
            T = params['T']

            t_array = np.linspace(0.0, n_periods * T, n_steps)

            self.analysis = JacketAnalysis(params)
            self.analysis.initialize()

            def progress_callback(current, total):
                self.progress_var.set(100 * current / total)
                self.status_label.config(text=f"Time step {current}/{total}")
                self.root.update()

            self.time_results = self.analysis.run_time_history(t_array, progress_callback)

            self.log_text.insert('1.0', f"Time history analysis complete!\n\n"
                               f"Summary:\n"
                               f"  Max Morison Fx: {self.time_results['Morison_Fx_kN'].max():.1f} kN\n"
                               f"  Max Morison Fy: {self.time_results['Morison_Fy_kN'].max():.1f} kN\n"
                               f"  Max Utilization: {self.time_results['Max_Utilization'].max():.2%}\n")

            self.status_label.config(text="Time history complete!")
            messagebox.showinfo("Complete", "Time history analysis completed!")

        except Exception as e:
            self.status_label.config(text="Error!")
            messagebox.showerror("Analysis Error", f"Error during analysis:\n{str(e)}")
            import traceback
            self.log_text.insert('1.0', traceback.format_exc())

    def show_summary(self):
        """Display results summary."""
        if self.results is None:
            messagebox.showwarning("No Results", "Please run analysis first!")
            return

        text = self.results['log']
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', text)

    def plot_structure(self):
        """Plot 3D structure."""
        if self.analysis is None:
            messagebox.showwarning("No Analysis", "Please run analysis first!")
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        structure = self.analysis.structure
        nodes = structure.nodes

        type_colors = {'leg': 'darkblue', 'h_brace': 'green', 'x_brace': 'gray'}
        type_lw = {'leg': 4.0, 'h_brace': 2.5, 'x_brace': 1.5}

        for member in structure.members:
            c1 = nodes[member['node1']]
            c2 = nodes[member['node2']]
            color = type_colors.get(member['type'], 'black')
            lw = type_lw.get(member['type'], 1.0)
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
                   color=color, linewidth=lw, alpha=0.8)

        for name, coords in nodes.items():
            z = coords[2]
            color = 'blue' if z <= 0.0 else 'red'
            size = 100 if name in ['A1', 'B1', 'C1', 'A4', 'B4', 'C4'] else 50
            ax.scatter(*coords, c=color, s=size, edgecolors='black', linewidth=1.5)

        ax.set_xlabel('X (East) [m]')
        ax.set_ylabel('Y (North) [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('3-Legged OSP Jacket Structure')

        plt.tight_layout()
        plt.show()

    def plot_forces(self):
        """Plot internal forces."""
        if self.results is None:
            messagebox.showwarning("No Results", "Please run analysis first!")
            return

        df = pd.DataFrame(self.results['internal_forces'])
        legs = df[df['type'] == 'leg']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Axial force
        ax = axes[0, 0]
        ax.barh(range(len(legs)), legs['Fx_max_kN'], color='steelblue')
        ax.set_yticks(range(len(legs)))
        ax.set_yticklabels(legs['member'])
        ax.set_xlabel('Axial Force [kN]')
        ax.set_title('Axial Force (Legs)')
        ax.grid(True, alpha=0.3)

        # Shear force
        ax = axes[0, 1]
        shear = np.sqrt(legs['Fy_max_kN']**2 + legs['Fz_max_kN']**2)
        ax.barh(range(len(legs)), shear, color='orange')
        ax.set_yticks(range(len(legs)))
        ax.set_yticklabels(legs['member'])
        ax.set_xlabel('Shear Force [kN]')
        ax.set_title('Shear Force (Legs)')
        ax.grid(True, alpha=0.3)

        # Bending moment
        ax = axes[1, 0]
        moment = np.sqrt(legs['My_max_kNm']**2 + legs['Mz_max_kNm']**2)
        ax.barh(range(len(legs)), moment, color='green')
        ax.set_yticks(range(len(legs)))
        ax.set_yticklabels(legs['member'])
        ax.set_xlabel('Bending Moment [kNm]')
        ax.set_title('Bending Moment (Legs)')
        ax.grid(True, alpha=0.3)

        # Utilization
        ax = axes[1, 1]
        colors = ['red' if u > 1.0 else 'steelblue' for u in df['utilization']]
        ax.barh(range(len(df)), df['utilization'], color=colors)
        ax.axvline(x=1.0, color='black', linestyle='--', label='Yield limit')
        ax.set_xlabel('Utilization [-]')
        ax.set_title('Utilization (All Members)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def export_results(self):
        """Export results to CSV."""
        if self.results is None:
            messagebox.showwarning("No Results", "Please run analysis first!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="internal_forces.csv"
        )
        if filename:
            df = pd.DataFrame(self.results['internal_forces'])
            df.to_csv(filename, index=False)
            messagebox.showinfo("Exported", f"Results saved to {filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  3-LEGGED OSP JACKET STRUCTURAL ANALYSIS")
    print("  Interactive GUI Version 7")
    print("=" * 60)
    
    if RASCHII_AVAILABLE:
        print("✓ Raschii wave library: AVAILABLE")
        print("  - Fenton wave theory (recommended)")
        print("  - Stokes wave theory (up to 5th order)")
        print("  - Airy wave theory (linear)")
    else:
        print("⚠ Raschii wave library: NOT AVAILABLE")
        print("  - Using fallback Airy (linear) wave theory")
        print("  - For nonlinear waves, install raschii manually:")
        print("    pip install raschii")
    
    print("\nStarting GUI...")
    print("=" * 60 + "\n")
    
    root = tk.Tk()
    
    # Set window icon (optional, won't fail if not available)
    try:
        root.iconbitmap(default='')
    except:
        pass
    
    # Center window on screen
    root.update_idletasks()
    width = 1400
    height = 900
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    app = JacketAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

