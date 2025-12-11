# -*- coding: utf-8 -*-
"""
Morison Equation + FEM Structural Analysis for 3-Legged OSP Jacket Structure
VERSION v6 - INTEGRATED WITH STEEL SOLVER METHODOLOGY

Features:
- Morison equation for hydrodynamic loads (using the raschii wave library if available)
- 3D beam FEM analysis based on STEEL Solver displacement method
- Internal forces: Axial, Shear (Fy, Fz), Bending (My, Mz), Torsion (Mx)
- Stress calculation at critical points for tubular sections
- Interface loads from topside (Axial: 25100 kN, Shear: 2900 kN)
- Steel yield strength: 355 MPa (elastic stress approach, no buckling)

Member Properties:
- Columns (Legs): D = 2000 mm, t = 75 mm
- Braces: D = 800 mm, t = 30 mm

Coordinate System:
- X: East (+)
- Y: North (+)
- Z: Up (+), z=0 at MWL

Reference: Bureau Veritas STEEL Solver Documentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import raschii for advanced wave models
try:
    import raschii
    RASCHII_AVAILABLE = True
except ImportError:
    RASCHII_AVAILABLE = False
    print("Warning: raschii not available, falling back to simplified linear wave model")

# =============================================================================
# CONSTANTS
# =============================================================================
g = 9.81       # Gravitational acceleration [m/s²]
rho = 1025     # Seawater density [kg/m³]

# Material Properties - Steel S355
E = 210000     # Young's modulus [N/mm²] = [MPa]
nu = 0.3       # Poisson's ratio [-]
G = E / (2 * (1 + nu))  # Shear modulus [N/mm²]
fy = 355       # Yield strength [N/mm²] = [MPa]
rho_steel = 7850  # Steel density [kg/m³]

# =============================================================================
# TUBULAR SECTION PROPERTIES CLASS
# =============================================================================
@dataclass
class TubularSection:
    """
    Tubular (pipe) cross-section properties.

    This class provides all geometrical and section parameters required by
    the FEM solver, and the stress calculation formulas for a closed
    tubular section, following the philosophy of STEEL Solver.

    All dimensional inputs are in mm. Internal derived values are kept
    in compatible units for FEM: areas in mm², inertias in mm⁴, etc.
    """
    D_outer: float  # Outer diameter [mm]
    t: float        # Wall thickness [mm]
    name: str = ""

    def __post_init__(self):
        # Inner diameter and radii
        self.D_inner = self.D_outer - 2 * self.t
        self.R_outer = self.D_outer / 2.0
        self.R_inner = self.D_inner / 2.0

        # Cross-sectional area [mm²] -> also in cm² and m² for convenience
        self.Ax_mm2 = np.pi / 4.0 * (self.D_outer**2 - self.D_inner**2)
        self.Ax_cm2 = self.Ax_mm2 / 100.0   # [cm²]
        self.Ax_m2 = self.Ax_mm2 / 1e6      # [m²]

        # Second moments of area about y and z [mm⁴] -> also cm⁴ and m⁴
        # Tube: Iy = Iz = π(D⁴ - d⁴)/64
        self.Iy_mm4 = np.pi / 64.0 * (self.D_outer**4 - self.D_inner**4)
        self.Iz_mm4 = self.Iy_mm4
        self.Iy_cm4 = self.Iy_mm4 / 1e4     # [cm⁴]
        self.Iz_cm4 = self.Iz_mm4 / 1e4     # [cm⁴]
        self.Iy_m4 = self.Iy_mm4 / 1e12     # [m⁴]
        self.Iz_m4 = self.Iz_mm4 / 1e12     # [m⁴]

        # Polar moment of inertia (torsional inertia) [mm⁴]
        # Tube: Ix = π(D⁴ - d⁴)/32 = 2 * Iy
        self.Ix_mm4 = np.pi / 32.0 * (self.D_outer**4 - self.D_inner**4)
        self.Ix_cm4 = self.Ix_mm4 / 1e4     # [cm⁴]
        self.Ix_m4 = self.Ix_mm4 / 1e12     # [m⁴]

        # Shear areas for Timoshenko beam theory [mm²]
        # For thin-walled tube: Ay ≈ Az ≈ Ax / 2 (common engineering approx.)
        self.Ay_mm2 = 0.5 * self.Ax_mm2
        self.Az_mm2 = 0.5 * self.Ax_mm2
        self.Ay_m2 = self.Ay_mm2 / 1e6
        self.Az_m2 = self.Az_mm2 / 1e6

        # Section moduli for bending [mm³] -> [cm³]
        # Wy = Wz = Iy / (D/2) = Iy / R_outer
        self.Wy_mm3 = self.Iy_mm4 / self.R_outer
        self.Wz_mm3 = self.Iz_mm4 / self.R_outer
        self.Wy_cm3 = self.Wy_mm3 / 1e3     # [cm³]
        self.Wz_cm3 = self.Wz_mm3 / 1e3     # [cm³]

        # Torsional section modulus [mm³] -> [cm³]
        # Wx = Ix / R_outer
        self.Wx_mm3 = self.Ix_mm4 / self.R_outer
        self.Wx_cm3 = self.Wx_mm3 / 1e3     # [cm³]

        # Effective areas for shear stress evaluation [mm²]
        # For tube under shear: take Sx,Sy,Sz = Ax as a conservative choice
        self.Sx_mm2 = self.Ax_mm2
        self.Sy_mm2 = self.Ax_mm2
        self.Sz_mm2 = self.Ax_mm2

        # Mass per unit length [kg/m]
        self.mass_per_m = self.Ax_m2 * rho_steel

    def get_stress_points(self) -> Dict[str, Tuple[float, float]]:
        """
        Return stress calculation points (y, z) relative to the centroid.

        For a tubular section we choose 8 points equally spaced around the
        outer circumference, labeled A1–A8.
        """
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
        """
        Calculate stresses at a given point on the section.

        Parameters
        ----------
        Fx, Fy, Fz : float
            Internal forces in local coordinates [N].
        Mx, My, Mz : float
            Internal moments in local coordinates [N·mm].
        point_id : str
            One of 'A1' ... 'A8', specifying which circumferential point.

        Returns
        -------
        dict
            Contains normal and shear stress components and von Mises stress
            in [N/mm²] = [MPa].

        The formulas follow the approach of STEEL Solver Section 1.3.4,
        specialized to a circular tube.
        """
        points = self.get_stress_points()
        y, z = points[point_id]

        # Axial normal stress
        sigma_Fx = Fx / self.Ax_mm2

        # Bending normal stress from My (bending about y-axis → stress in z)
        sigma_My = My * z / self.Iy_mm4 if self.Iy_mm4 > 0 else 0.0

        # Bending normal stress from Mz (bending about z-axis → stress in y)
        sigma_Mz = Mz * y / self.Iz_mm4 if self.Iz_mm4 > 0 else 0.0

        # Total normal stress
        sigma_total = sigma_Fx + sigma_My + sigma_Mz

        # Shear stress due to torsion Mx
        R = np.sqrt(y**2 + z**2)
        tau_Mx = Mx * R / self.Ix_mm4 if self.Ix_mm4 > 0 else 0.0

        # Shear stresses due to shear forces Fy and Fz
        tau_Fy = Fy / self.Ay_mm2 if self.Ay_mm2 > 0 else 0.0
        tau_Fz = Fz / self.Az_mm2 if self.Az_mm2 > 0 else 0.0

        # Total shear stress magnitude (vector combination of components)
        tau_total = np.sqrt(tau_Mx**2 + tau_Fy**2 + tau_Fz**2)

        # Von Mises equivalent stress
        sigma_vm = np.sqrt(sigma_total**2 + 3.0 * tau_total**2)

        return {
            'sigma_Fx': sigma_Fx,
            'sigma_My': sigma_My,
            'sigma_Mz': sigma_Mz,
            'sigma_total': sigma_total,
            'tau_Mx': tau_Mx,
            'tau_Fy': tau_Fy,
            'tau_Fz': tau_Fz,
            'tau_total': tau_total,
            'von_mises': sigma_vm
        }

    def print_properties(self):
        """Print basic section properties to the console."""
        print(f"\n{'='*50}")
        print(f"TUBULAR SECTION: {self.name}")
        print(f"{'='*50}")
        print(f"  Outer diameter D = {self.D_outer:.1f} mm")
        print(f"  Wall thickness t = {self.t:.1f} mm")
        print(f"  Inner diameter d = {self.D_inner:.1f} mm")
        print(f"\n  Stiffness Characteristics:")
        print(f"    Ax = {self.Ax_cm2:.2f} cm²")
        print(f"    Ay = {self.Ay_mm2/100.0:.2f} cm²")
        print(f"    Az = {self.Az_mm2/100.0:.2f} cm²")
        print(f"    Ix = {self.Ix_cm4:.2f} cm⁴ (torsional)")
        print(f"    Iy = {self.Iy_cm4:.2f} cm⁴")
        print(f"    Iz = {self.Iz_cm4:.2f} cm⁴")
        print(f"\n  Stress Characteristics:")
        print(f"    Wx = {self.Wx_cm3:.2f} cm³")
        print(f"    Wy = {self.Wy_cm3:.2f} cm³")
        print(f"    Wz = {self.Wz_cm3:.2f} cm³")
        print(f"\n  Mass per meter = {self.mass_per_m:.2f} kg/m")


# =============================================================================
# RASCHII WAVE WRAPPER CLASS
# =============================================================================
class RaschiiWave:
    """
    Wrapper class for raschii wave models.

    NOTE: raschii uses z = 0 at seabed and positive upward.
    In this code, z = 0 is at MWL, positive upward, so we adjust
    coordinates accordingly.
    """

    def __init__(self, H: float, T: float, d: float, U_c: float = 0.0,
                 wave_model: str = 'Fenton', N: int = 5, dt: float = 0.001):
        """
        Parameters
        ----------
        H : float
            Wave height [m].
        T : float
            Wave period [s].
        d : float
            Water depth [m].
        U_c : float
            Uniform current velocity [m/s] in wave direction.
        wave_model : str
            'Fenton', 'Stokes', 'Airy', or 'auto'.
        N : int
            Number of harmonics for higher-order waves.
        dt : float
            Time step for numerical time differentiation of velocities.
        """
        self.H = H
        self.T = T
        self.d = d
        self.U_c = U_c
        self.wave_model_name = wave_model
        self.N = N
        self.dt = dt

        if RASCHII_AVAILABLE:
            # Use raschii's non-linear wave models
            self.wave = self._create_wave(wave_model, N)
            self.omega = self.wave.omega
            self.k = self.wave.k
            self.L = self.wave.length
            self.c = self.wave.c
        else:
            # Fallback: Airy (linear) wave theory with dispersion relation
            self.omega = 2.0 * np.pi / T
            self.k = self._solve_dispersion(self.omega, d)
            self.L = 2.0 * np.pi / self.k
            self.c = self.L / T
            self.wave = None

        self.a = H / 2.0  # wave amplitude

    def _solve_dispersion(self, omega: float, d: float) -> float:
        """
        Solve the dispersion relation ω² = g k tanh(k d)
        using a Newton–Raphson iteration.
        """
        k = omega**2 / g  # initial guess
        for _ in range(50):
            f = omega**2 - g * k * np.tanh(k * d)
            df = -g * (np.tanh(k * d) + k * d / np.cosh(k * d)**2)
            k_new = k - f / df
            if abs(k_new - k) < 1e-10:
                break
            k = k_new
        return k

    def _create_wave(self, model: str, N: int):
        """
        Create a raschii wave object depending on the chosen wave model.
        For 'auto', a suitable model (Airy/Stokes/Fenton) is selected based
        on wave steepness.
        """
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
        """
        Free surface elevation η relative to MWL at horizontal position x.

        In raschii, surface elevation is given relative to seabed. We
        subtract the water depth to express it relative to MWL.
        """
        if self.wave is not None:
            eta_raschii = self.wave.surface_elevation(x, t=t)
            if hasattr(eta_raschii, '__len__'):
                eta_raschii = eta_raschii[0]
            return eta_raschii - self.d
        else:
            # Airy wave
            return self.a * np.cos(self.k * x - self.omega * t)

    def velocity(self, x: float, z_mwl: float, t: float = 0.0) -> Tuple[float, float]:
        """
        Horizontal and vertical water particle velocities at (x, z_mwl, t).

        z_mwl is the vertical coordinate relative to MWL (z=0 at MWL).
        Returns horizontal u and vertical w components [m/s].
        """
        eta_local = self.eta(x, t)
        # If the point is above the instantaneous free surface, velocity = 0
        if z_mwl > eta_local:
            return (0.0, 0.0)

        if self.wave is not None:
            # Convert MWL-based z to raschii convention (0 at seabed)
            z_raschii = z_mwl + self.d
            # Keep point within water column
            z_raschii = max(0.01, min(z_raschii, self.d + eta_local - 0.01))
            vel = self.wave.velocity(x, z_raschii, t=t)
            if hasattr(vel, 'shape'):
                u = float(vel[0, 0])
                w = float(vel[0, 1])
            else:
                u = float(vel[0])
                w = float(vel[1])
        else:
            # Airy wave theory velocities
            kd = self.k * self.d
            kz = self.k * (z_mwl + self.d)
            phase = self.k * x - self.omega * t
            u = (self.a * self.omega *
                 np.cosh(kz) / np.sinh(kd) * np.cos(phase))
            w = (self.a * self.omega *
                 np.sinh(kz) / np.sinh(kd) * np.sin(phase))

        # Add uniform current in x-direction (wave propagation direction)
        u += self.U_c
        return (u, w)

    def acceleration(self, x: float, z_mwl: float, t: float = 0.0) -> Tuple[float, float]:
        """
        Water particle accelerations (du/dt, dw/dt) at (x, z_mwl, t).

        Numerical time differentiation is used (forward difference).
        """
        eta_local = self.eta(x, t)
        if z_mwl > eta_local:
            return (0.0, 0.0)

        u0, w0 = self.velocity(x, z_mwl, t)
        u1, w1 = self.velocity(x, z_mwl, t + self.dt)
        return ((u1 - u0) / self.dt, (w1 - w0) / self.dt)

    def get_kinematics(self, x: float, z_mwl: float, t: float = 0.0) -> Dict:
        """
        Get wave kinematics at a point: velocities, accelerations and
        free surface elevation.
        """
        eta_local = self.eta(x, t)
        if z_mwl > eta_local:
            return {
                'u': 0.0, 'w': 0.0,
                'du_dt': 0.0, 'dw_dt': 0.0,
                'submerged': False,
                'eta': eta_local
            }

        u, w = self.velocity(x, z_mwl, t)
        du_dt, dw_dt = self.acceleration(x, z_mwl, t)
        return {
            'u': u, 'w': w,
            'du_dt': du_dt, 'dw_dt': dw_dt,
            'submerged': True,
            'eta': eta_local
        }

    def print_info(self):
        """Print basic wave properties."""
        print(f"\n  Wave model: {self.wave_model_name}")
        print(f"  Wave height H = {self.H} m")
        print(f"  Wave period T = {self.T:.4f} s")
        print(f"  Water depth d = {self.d} m")
        print(f"  Current U_c = {self.U_c} m/s")
        print(f"  Wave number k = {self.k:.6f} rad/m")
        print(f"  Wavelength L = {self.L:.2f} m")
        print(f"  Phase velocity c = {self.c:.4f} m/s")
        print(f"  Wave steepness H/L = {self.H/self.L:.4f}")


# =============================================================================
# STRUCTURE DEFINITION
# =============================================================================
@dataclass
class JacketStructure:
    """
    3-Legged OSP Jacket Structure with FEM definition.

    Coordinate system:
    - X: East (+)
    - Y: North (+)
    - Z: Up (+), z=0 at MWL
    """

    z_water_ref: float = 47.0  # MWL depth from original coordinate system

    # Section dimensions [mm]
    D_leg: float = 2000.0      # Leg outer diameter
    t_leg: float = 75.0        # Leg wall thickness
    D_brace: float = 800.0     # Brace outer diameter
    t_brace: float = 30.0      # Brace wall thickness

    def __post_init__(self):
        # Define section properties for legs and braces
        self.section_leg = TubularSection(self.D_leg, self.t_leg, "Column/Leg")
        self.section_brace = TubularSection(self.D_brace, self.t_brace, "Brace")

        # Define nodes and members
        self.nodes = self._define_nodes()
        self.node_list = list(self.nodes.keys())
        self.n_nodes = len(self.node_list)
        self.n_dof = 6 * self.n_nodes  # 6 DOF per node (u, v, w, θx, θy, θz)

        # Mapping from node name to index
        self.node_index = {name: i for i, name in enumerate(self.node_list)}

        # Define structural members
        self.members = self._define_members()
        self.n_members = len(self.members)

    def _define_nodes(self) -> Dict[str, np.ndarray]:
        """
        Define all nodes in MWL-based coordinates.

        The original jacket geometry is defined relative to seabed.
        Here we subtract z_water_ref so that z=0 corresponds to MWL.
        """
        z_ref = self.z_water_ref
        nodes: Dict[str, np.ndarray] = {}

        # LEG A (A1 bottom, A4 top)
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
        """
        Define all members (legs, horizontal braces, X-braces) with their
        end nodes, section properties and type.
        """
        members: List[Dict] = []

        # Legs (each leg split into 3 segments)
        for leg in ['A', 'B', 'C']:
            for i in [1, 2, 3]:
                members.append({
                    'name': f'Leg_{leg}{i}-{leg}{i+1}',
                    'node1': f'{leg}{i}',
                    'node2': f'{leg}{i+1}',
                    'section': self.section_leg,
                    'type': 'leg'
                })

        # Bottom horizontal braces (at mudline / z ~ -z_ref)
        for n1, n2 in [('A1', 'B1'), ('B1', 'C1'), ('C1', 'A1')]:
            members.append({
                'name': f'HBrace_{n1}-{n2}',
                'node1': n1,
                'node2': n2,
                'section': self.section_brace,
                'type': 'h_brace'
            })

        # Level 2 horizontal braces
        for n1, n2 in [('A2', 'B2'), ('B2', 'C2'), ('C2', 'A2')]:
            members.append({
                'name': f'HBrace_{n1}-{n2}',
                'node1': n1,
                'node2': n2,
                'section': self.section_brace,
                'type': 'h_brace'
            })

        # X-bracing configuration (3 levels, 12 braces per level)
        xbrace_config = [
            # Level 1
            [('A1', 'HAB1'), ('HAB1', 'B2'), ('B1', 'HAB1'), ('HAB1', 'A2'),
             ('B1', 'HBC1'), ('HBC1', 'C2'), ('C1', 'HBC1'), ('HBC1', 'B2'),
             ('C1', 'HCA1'), ('HCA1', 'A2'), ('A1', 'HCA1'), ('HCA1', 'C2')],
            # Level 2
            [('A2', 'HAB2'), ('HAB2', 'B3'), ('B2', 'HAB2'), ('HAB2', 'A3'),
             ('B2', 'HBC2'), ('HBC2', 'C3'), ('C2', 'HBC2'), ('HBC2', 'B3'),
             ('C2', 'HCA2'), ('HCA2', 'A3'), ('A2', 'HCA2'), ('HCA2', 'C3')],
            # Level 3
            [('A3', 'HAB3'), ('HAB3', 'B4'), ('B3', 'HAB3'), ('HAB3', 'A4'),
             ('B3', 'HBC3'), ('HBC3', 'C4'), ('C3', 'HBC3'), ('HBC3', 'B4'),
             ('C3', 'HCA3'), ('HCA3', 'A4'), ('A3', 'HCA3'), ('HCA3', 'C4')],
        ]
        for level_braces in xbrace_config:
            for n1, n2 in level_braces:
                members.append({
                    'name': f'XBr_{n1}-{n2}',
                    'node1': n1,
                    'node2': n2,
                    'section': self.section_brace,
                    'type': 'x_brace'
                })

        return members

    def get_member_geometry(self, member: Dict) -> Dict:
        """
        Compute basic geometry of a member:
        - End coordinates
        - Vector difference
        - Length (in m and mm)
        - Unit direction vector
        """
        coord1 = self.nodes[member['node1']]
        coord2 = self.nodes[member['node2']]

        dL = coord2 - coord1
        L = np.linalg.norm(dL)
        unit_vec = dL / L if L > 0 else np.array([1.0, 0.0, 0.0])

        return {
            'coord1': coord1,
            'coord2': coord2,
            'dL': dL,
            'L': L,
            'L_mm': L * 1000.0,
            'unit_vec': unit_vec
        }

    def get_top_nodes(self) -> List[str]:
        """Return the names of the top leg nodes (interface level)."""
        return ['A4', 'B4', 'C4']

    def get_bottom_nodes(self) -> List[str]:
        """Return the names of the bottom leg nodes (mudline support)."""
        return ['A1', 'B1', 'C1']


# =============================================================================
# FEM BEAM ELEMENT CLASS
# =============================================================================
class BeamElement3D:
    """
    3D Timoshenko beam element based on STEEL Solver methodology.

    Each element has 12 DOFs in local coordinates:
    [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]

    Local coordinate system:
    - x: along beam axis (from node1 to node2)
    - y: transverse direction
    - z: transverse direction forming a right-handed system with x and y
    """

    def __init__(self, node1_coords: np.ndarray, node2_coords: np.ndarray,
                 section: TubularSection, E: float = 210000, G: float = 80769,
                 include_shear: bool = True):
        """
        Parameters
        ----------
        node1_coords, node2_coords : np.ndarray
            Global coordinates [m].
        section : TubularSection
            Cross-section properties.
        E : float
            Young's modulus [N/mm²].
        G : float
            Shear modulus [N/mm²].
        include_shear : bool
            If True, includes shear deformation (Timoshenko beam).
        """
        self.node1 = node1_coords
        self.node2 = node2_coords
        self.section = section
        self.E = E
        self.G = G
        self.include_shear = include_shear

        # Basic geometry
        self.dL = node2_coords - node1_coords
        self.L = np.linalg.norm(self.dL)        # [m]
        self.L_mm = self.L * 1000.0             # [mm]

        # Compute transformation matrix (local ↔ global)
        self.T = self._compute_transformation_matrix()

        # Local stiffness matrix (12x12)
        self.K_local = self._compute_local_stiffness()

        # Global stiffness matrix of the element
        self.K_global = self.T.T @ self.K_local @ self.T

    def _compute_transformation_matrix(self) -> np.ndarray:
        """
        Compute the 12x12 transformation matrix from local to global DOFs.

        The same 3x3 rotation matrix is applied to translations and rotations
        at both ends (block-diagonal).
        """
        # Local x-axis (element axis)
        lx = self.dL / self.L

        # Try to align local z-axis as close as possible to global vertical
        global_z = np.array([0.0, 0.0, 1.0])

        # If element is almost vertical, use global Y as reference
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
            # General case: build orthogonal triad
            lz = np.cross(lx, global_z)
            lz = lz / np.linalg.norm(lz)
            ly = np.cross(lz, lx)

        # 3x3 rotation matrix from local to global
        R = np.array([lx, ly, lz])

        # Assemble 12x12 transformation matrix
        T = np.zeros((12, 12))
        for i in range(4):
            T[3*i:3*i+3, 3*i:3*i+3] = R

        return T

    def _compute_local_stiffness(self) -> np.ndarray:
        """
        Compute the 12x12 local stiffness matrix of the beam element.

        This follows the STEEL Solver documentation (Section 1.2):

        - Axial stiffness: EA / L
        - Bending stiffness in both principal planes with shear correction
        - Torsional stiffness: G Ix / L

        Units:
        - Length in mm
        - Forces in N
        - Moments in N·mm
        """
        L = self.L_mm

        # Section properties (mm², mm⁴)
        Ax = self.section.Ax_mm2
        Ay = self.section.Ay_mm2
        Az = self.section.Az_mm2
        Ix = self.section.Ix_mm4
        Iy = self.section.Iy_mm4
        Iz = self.section.Iz_mm4

        E = self.E
        G = self.G

        # Shear correction factors (Timoshenko)
        if self.include_shear and Ay > 0.0 and Az > 0.0:
            Phi_y = 12.0 * E * Iz / (G * Az * L**2)
            Phi_z = 12.0 * E * Iy / (G * Ay * L**2)
        else:
            Phi_y = 0.0
            Phi_z = 0.0

        # Basic stiffness coefficients
        alpha = E * Ax / L           # axial
        bz = E * Iz / ((1.0 + Phi_y) * L**3)  # bending about y-z plane
        by = E * Iy / ((1.0 + Phi_z) * L**3)  # bending about x-z plane
        t = G * Ix / L               # torsion

        # Initialize local stiffness matrix
        K = np.zeros((12, 12))

        # Axial terms
        K[0, 0] = alpha
        K[6, 6] = alpha
        K[0, 6] = -alpha
        K[6, 0] = -alpha

        # Bending in local y-z plane (shear Fy, moment Mz)
        K[1, 1] = 12.0 * bz
        K[7, 7] = 12.0 * bz
        K[1, 7] = -12.0 * bz
        K[7, 1] = -12.0 * bz

        K[1, 5] = 6.0 * bz * L
        K[5, 1] = 6.0 * bz * L
        K[1, 11] = 6.0 * bz * L
        K[11, 1] = 6.0 * bz * L
        K[7, 5] = -6.0 * bz * L
        K[5, 7] = -6.0 * bz * L
        K[7, 11] = -6.0 * bz * L
        K[11, 7] = -6.0 * bz * L

        K[5, 5] = (4.0 + Phi_y) * bz * L**2
        K[11, 11] = (4.0 + Phi_y) * bz * L**2
        K[5, 11] = (2.0 - Phi_y) * bz * L**2
        K[11, 5] = (2.0 - Phi_y) * bz * L**2

        # Bending in local x-z plane (shear Fz, moment My)
        K[2, 2] = 12.0 * by
        K[8, 8] = 12.0 * by
        K[2, 8] = -12.0 * by
        K[8, 2] = -12.0 * by

        K[2, 4] = -6.0 * by * L
        K[4, 2] = -6.0 * by * L
        K[2, 10] = -6.0 * by * L
        K[10, 2] = -6.0 * by * L
        K[8, 4] = 6.0 * by * L
        K[4, 8] = 6.0 * by * L
        K[8, 10] = 6.0 * by * L
        K[10, 8] = 6.0 * by * L

        K[4, 4] = (4.0 + Phi_z) * by * L**2
        K[10, 10] = (4.0 + Phi_z) * by * L**2
        K[4, 10] = (2.0 - Phi_z) * by * L**2
        K[10, 4] = (2.0 - Phi_z) * by * L**2

        # Torsion
        K[3, 3] = t
        K[9, 9] = t
        K[3, 9] = -t
        K[9, 3] = -t

        return K

    def get_internal_forces(self, u_global: np.ndarray) -> Dict:
        """
        Compute internal element forces from global nodal displacements.

        Parameters
        ----------
        u_global : np.ndarray
            12-component displacement vector in global coordinates:
            [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]

        Returns
        -------
        dict
            Internal forces at both ends in local coordinates (Fx, Fy, Fz, Mx, My, Mz)
            plus the local displacement and force vectors.
        """
        # Transform global displacements to local coordinates
        u_local = self.T @ u_global

        # Local internal force vector
        F_local = self.K_local @ u_local

        # Forces at node 1 (start, sign convention for element)
        Fx1 = -F_local[0]
        Fy1 = -F_local[1]
        Fz1 = -F_local[2]
        Mx1 = -F_local[3]
        My1 = -F_local[4]
        Mz1 = -F_local[5]

        # Forces at node 2 (end)
        Fx2 = F_local[6]
        Fy2 = F_local[7]
        Fz2 = F_local[8]
        Mx2 = F_local[9]
        My2 = F_local[10]
        Mz2 = F_local[11]

        return {
            'node1': {'Fx': Fx1, 'Fy': Fy1, 'Fz': Fz1,
                      'Mx': Mx1, 'My': My1, 'Mz': Mz1},
            'node2': {'Fx': Fx2, 'Fy': Fy2, 'Fz': Fz2,
                      'Mx': Mx2, 'My': My2, 'Mz': Mz2},
            'u_local': u_local,
            'F_local': F_local
        }


# =============================================================================
# FEM SOLVER CLASS
# =============================================================================
class FEMSolver:
    """
    3D frame FEM solver based on the displacement method.

    The solver assembles:
        K_global * U_global = F_global
    from all beam elements, and solves for U_global.
    """

    def __init__(self, structure: JacketStructure):
        self.structure = structure
        self.n_dof = structure.n_dof

        # Global matrices and vectors
        self.K_global = np.zeros((self.n_dof, self.n_dof))
        self.F_global = np.zeros(self.n_dof)
        self.U_global = np.zeros(self.n_dof)

        # Beam elements
        self.elements: List[BeamElement3D] = []

        # Build elements and assemble the global stiffness matrix
        self._build_elements()
        self._assemble_global_stiffness()

    def _build_elements(self):
        """Create beam element objects for all members."""
        for member in self.structure.members:
            coord1 = self.structure.nodes[member['node1']]
            coord2 = self.structure.nodes[member['node2']]
            section = member['section']

            element = BeamElement3D(coord1, coord2, section, E, G)
            self.elements.append(element)

    def _assemble_global_stiffness(self):
        """Assemble the global stiffness matrix K_global from all elements."""
        for i, member in enumerate(self.structure.members):
            element = self.elements[i]

            # DOF indices for this element
            idx1 = self.structure.node_index[member['node1']]
            idx2 = self.structure.node_index[member['node2']]

            dof1 = np.arange(6 * idx1, 6 * idx1 + 6)
            dof2 = np.arange(6 * idx2, 6 * idx2 + 6)
            dofs = np.concatenate([dof1, dof2])

            # Scatter-add element stiffness into global matrix
            for ii, di in enumerate(dofs):
                for jj, dj in enumerate(dofs):
                    self.K_global[di, dj] += element.K_global[ii, jj]

    def apply_nodal_force(self, node_name: str, force_vector: np.ndarray):
        """
        Apply a concentrated load at a node.

        Parameters
        ----------
        node_name : str
            Name of the node (e.g., 'A4').
        force_vector : np.ndarray
            6-component vector [Fx, Fy, Fz, Mx, My, Mz].
            Units: [N, N, N, N·mm, N·mm, N·mm].
        """
        idx = self.structure.node_index[node_name]
        dofs = np.arange(6 * idx, 6 * idx + 6)
        self.F_global[dofs] += force_vector

    def apply_member_distributed_load(self, member_idx: int,
                                      w_local: np.ndarray,
                                      L_submerged: float = None):
        """
        Apply a (uniform) distributed load on a member and convert it to
        equivalent nodal loads.

        This is a simplified helper function (not used in the main Morison
        workflow, where loads are integrated more explicitly).

        Parameters
        ----------
        member_idx : int
            Index of the member in structure.members.
        w_local : np.ndarray
            Distributed load [N/m] in global coordinates [wx, wy, wz].
        L_submerged : float, optional
            Submerged length [m]. If None, full member length is used.
        """
        member = self.structure.members[member_idx]
        element = self.elements[member_idx]

        L = element.L * 1000.0
        L_sub = L_submerged * 1000.0 if L_submerged else L

        # Convert distributed load to equivalent nodal forces (simplified)
        F_total = w_local * L_sub / 1000.0  # back to N (since w is N/m)

        idx1 = self.structure.node_index[member['node1']]
        idx2 = self.structure.node_index[member['node2']]

        dof1 = 6 * idx1
        dof2 = 6 * idx2

        # Apply half of total to each node in global translations
        self.F_global[dof1:dof1+3] += F_total / 2.0
        self.F_global[dof2:dof2+3] += F_total / 2.0

    def apply_boundary_conditions(self, fixed_nodes: List[str]):
        """
        Define boundary conditions (fixed DOFs) at the given nodes.

        All 6 DOFs of each specified node are fully restrained.
        """
        self.fixed_dofs: List[int] = []

        for node_name in fixed_nodes:
            idx = self.structure.node_index[node_name]
            dofs = np.arange(6 * idx, 6 * idx + 6)
            self.fixed_dofs.extend(dofs.tolist())

        self.fixed_dofs = np.array(self.fixed_dofs, dtype=int)
        self.free_dofs = np.setdiff1d(np.arange(self.n_dof), self.fixed_dofs)

    def solve(self) -> np.ndarray:
        """
        Solve the global equilibrium system F = K · U for nodal displacements.

        Returns
        -------
        U_global : np.ndarray
            Global displacement vector [mm for translations, mrad for rotations].
        """
        # Extract free DOF sub-system
        K_ff = self.K_global[np.ix_(self.free_dofs, self.free_dofs)]
        F_f = self.F_global[self.free_dofs]

        # Solve the linear system
        try:
            U_f = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            print("Warning: Singular global stiffness matrix, solving with least squares")
            U_f = np.linalg.lstsq(K_ff, F_f, rcond=None)[0]

        # Assemble full displacement vector
        self.U_global = np.zeros(self.n_dof)
        self.U_global[self.free_dofs] = U_f

        return self.U_global

    def get_reactions(self) -> Dict[str, np.ndarray]:
        """
        Compute reaction forces and moments at fixed DOFs.

        Returns
        -------
        dict
            Mapping node_name -> 6-component reaction vector [Rx, Ry, Rz, Mx, My, Mz].
        """
        # R = K U - F
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

    def get_member_internal_forces(self) -> List[Dict]:
        """
        Compute internal forces and member-level stress/utilization
        for all members.

        Returns
        -------
        list of dict
            One dictionary per member with internal forces, stresses and utilization.
        """
        results: List[Dict] = []

        for i, member in enumerate(self.structure.members):
            element = self.elements[i]

            # Collect global DOFs for this element
            idx1 = self.structure.node_index[member['node1']]
            idx2 = self.structure.node_index[member['node2']]
            dof1 = np.arange(6 * idx1, 6 * idx1 + 6)
            dof2 = np.arange(6 * idx2, 6 * idx2 + 6)

            u_elem = np.concatenate([self.U_global[dof1], self.U_global[dof2]])

            # Element internal forces in local coordinates
            forces = element.get_internal_forces(u_elem)

            section = member['section']

            # Maximum absolute internal forces between both ends
            Fx_max = max(abs(forces['node1']['Fx']), abs(forces['node2']['Fx']))
            Fy_max = max(abs(forces['node1']['Fy']), abs(forces['node2']['Fy']))
            Fz_max = max(abs(forces['node1']['Fz']), abs(forces['node2']['Fz']))
            Mx_max = max(abs(forces['node1']['Mx']), abs(forces['node2']['Mx']))
            My_max = max(abs(forces['node1']['My']), abs(forces['node2']['My']))
            Mz_max = max(abs(forces['node1']['Mz']), abs(forces['node2']['Mz']))

            # Stress at one reference point (A1) using node1 forces (signed)
            stresses = section.calc_stress_at_point(
                forces['node1']['Fx'],
                forces['node1']['Fy'],
                forces['node1']['Fz'],
                forces['node1']['Mx'],
                forces['node1']['My'],
                forces['node1']['Mz'],
                'A1'
            )

            # Evaluate von Mises stress at all eight circumferential points
            # and take the maximum as the governing stress
            max_vm = 0.0
            for pt in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']:
                st = section.calc_stress_at_point(
                    forces['node1']['Fx'],
                    forces['node1']['Fy'],
                    forces['node1']['Fz'],
                    forces['node1']['Mx'],
                    forces['node1']['My'],
                    forces['node1']['Mz'],
                    pt
                )
                max_vm = max(max_vm, st['von_mises'])

            results.append({
                'member': member['name'],
                'type': member['type'],
                'node1': member['node1'],
                'node2': member['node2'],
                'length_m': element.L,
                'section': section.name,

                # Internal forces at node 1 (kN / kNm)
                'Fx_N1_kN': forces['node1']['Fx'] / 1000.0,
                'Fy_N1_kN': forces['node1']['Fy'] / 1000.0,
                'Fz_N1_kN': forces['node1']['Fz'] / 1000.0,
                'Mx_N1_kNm': forces['node1']['Mx'] / 1e6,
                'My_N1_kNm': forces['node1']['My'] / 1e6,
                'Mz_N1_kNm': forces['node1']['Mz'] / 1e6,

                # Internal forces at node 2 (kN / kNm)
                'Fx_N2_kN': forces['node2']['Fx'] / 1000.0,
                'Fy_N2_kN': forces['node2']['Fy'] / 1000.0,
                'Fz_N2_kN': forces['node2']['Fz'] / 1000.0,
                'Mx_N2_kNm': forces['node2']['Mx'] / 1e6,
                'My_N2_kNm': forces['node2']['My'] / 1e6,
                'Mz_N2_kNm': forces['node2']['Mz'] / 1e6,

                # Maximum forces for convenience
                'Fx_max_kN': Fx_max / 1000.0,
                'Fy_max_kN': Fy_max / 1000.0,
                'Fz_max_kN': Fz_max / 1000.0,
                'Mx_max_kNm': Mx_max / 1e6,
                'My_max_kNm': My_max / 1e6,
                'Mz_max_kNm': Mz_max / 1e6,

                # Stresses at A1 (for reference)
                'sigma_max_MPa': stresses['sigma_total'],
                'tau_max_MPa': stresses['tau_total'],

                # Governing von Mises stress and utilization
                'von_mises_max_MPa': max_vm,
                'utilization': max_vm / fy
            })

        return results


# =============================================================================
# MORISON FORCE CALCULATOR
# =============================================================================
class MorisonCalculator:
    """
    Morison force calculation integrated with the FEM solver.

    For each member, the code:
    - samples wave kinematics along the member (Gaussian integration),
    - computes drag and inertia forces per unit length,
    - integrates to total element forces,
    - distributes these forces to the two end nodes as equivalent nodal loads.
    """

    def __init__(self, structure: JacketStructure, wave: RaschiiWave,
                 wave_direction: float = 38.0, Cd: float = 0.7, Cm: float = 2.0):
        """
        Parameters
        ----------
        structure : JacketStructure
        wave : RaschiiWave
        wave_direction : float
            Wave propagation direction (degrees from North).
        Cd : float
            Drag coefficient.
        Cm : float
            Inertia coefficient.
        """
        self.structure = structure
        self.wave = wave
        # Convert compass bearing (from North, clockwise) into math angle
        self.theta_dir = np.deg2rad(90.0 - wave_direction)
        self.wave_direction_compass = wave_direction
        self.Cd = Cd
        self.Cm = Cm

    def get_wave_coords(self, x: float, y: float) -> float:
        """
        Transform global (x, y) into wave propagation coordinate x_wave.

        x_wave is the coordinate along the wave direction.
        """
        return x * np.cos(self.theta_dir) + y * np.sin(self.theta_dir)

    def get_kinematics_3d(self, x: float, y: float, z: float, t: float) -> Dict:
        """
        Get 3D wave kinematics (u, v, w, du/dt, dv/dt, dw/dt) at a global
        point (x, y, z) and time t.

        The 2D wave solution is computed along the wave direction, then
        rotated into global x–y components.
        """
        x_wave = self.get_wave_coords(x, y)
        kin2d = self.wave.get_kinematics(x_wave, z, t)

        if not kin2d['submerged']:
            return {
                'u': 0.0, 'v': 0.0, 'w': 0.0,
                'du_dt': 0.0, 'dv_dt': 0.0, 'dw_dt': 0.0,
                'submerged': False, 'eta': kin2d['eta']
            }

        cos_dir = np.cos(self.theta_dir)
        sin_dir = np.sin(self.theta_dir)

        return {
            'u': kin2d['u'] * cos_dir,
            'v': kin2d['u'] * sin_dir,
            'w': kin2d['w'],
            'du_dt': kin2d['du_dt'] * cos_dir,
            'dv_dt': kin2d['du_dt'] * sin_dir,
            'dw_dt': kin2d['dw_dt'],
            'submerged': True,
            'eta': kin2d['eta']
        }

    def morison_force_per_length(self, U_vec: np.ndarray, A_vec: np.ndarray,
                                 unit_vec: np.ndarray, D: float) -> np.ndarray:
        """
        Calculate Morison force per unit length [N/m] on a cylindrical member.

        Only the velocity and acceleration components normal to the member
        axis are considered (standard Morison for slender members).
        """
        # Remove axial components (keep perpendicular to member axis)
        U_perp = U_vec - np.dot(U_vec, unit_vec) * unit_vec
        A_perp = A_vec - np.dot(A_vec, unit_vec) * unit_vec
        U_perp_mag = np.linalg.norm(U_perp)

        # Cross-sectional area of cylinder [m²]
        A_cross = np.pi * D**2 / 4.0

        # Drag force per unit length
        if U_perp_mag > 1e-10:
            F_drag = 0.5 * rho * self.Cd * D * U_perp_mag * U_perp
        else:
            F_drag = np.zeros(3)

        # Inertia force per unit length
        F_inertia = rho * self.Cm * A_cross * A_perp

        return F_drag + F_inertia

    def compute_member_morison_forces(self, member_idx: int, t: float = 0.0,
                                      n_gauss: int = 15) -> Dict:
        """
        Compute Morison forces on a specific member and convert them
        to equivalent nodal forces at its two ends.

        Gaussian quadrature is used along the member span.
        """
        member = self.structure.members[member_idx]
        coord1 = self.structure.nodes[member['node1']]
        coord2 = self.structure.nodes[member['node2']]
        D = member['section'].D_outer / 1000.0  # [m]

        dL = coord2 - coord1
        L = np.linalg.norm(dL)
        unit_vec = dL / L

        # Gauss-Legendre quadrature on [0, 1]
        xi, weights = np.polynomial.legendre.leggauss(n_gauss)
        s_values = (xi + 1.0) / 2.0
        w_scaled = weights / 2.0

        F1 = np.zeros(3)
        F2 = np.zeros(3)
        total_force = np.zeros(3)
        submerged_length = 0.0

        for s, w in zip(s_values, w_scaled):
            # Position along member: coord = coord1 + s * dL
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

            # Linear shape functions: (1 - s) at node1, s at node2
            F1 += (1.0 - s) * f_integrated
            F2 += s * f_integrated

        return {
            'member_idx': member_idx,
            'member_name': member['name'],
            'length': L,
            'submerged_length': submerged_length,
            'F_node1': F1,
            'F_node2': F2,
            'F_total': total_force
        }

    def compute_all_morison_forces(self, t: float = 0.0) -> Dict:
        """
        Compute Morison forces for all members and aggregate the resulting
        equivalent nodal loads.

        Returns
        -------
        dict
            - nodal_forces: mapping node_name -> 6-component force vector
              (only first 3 entries used: Fx, Fy, Fz).
            - member_results: list of per-member integration results.
            - total_force: total global wave force vector [Fx, Fy, Fz].
        """
        nodal_forces: Dict[str, np.ndarray] = {
            name: np.zeros(6) for name in self.structure.nodes.keys()
        }
        member_results: List[Dict] = []

        for i, member in enumerate(self.structure.members):
            result = self.compute_member_morison_forces(i, t)

            # Add equivalent nodal forces (translations only)
            nodal_forces[member['node1']][:3] += result['F_node1']
            nodal_forces[member['node2']][:3] += result['F_node2']

            member_results.append(result)

        # Sum total forces over all members
        total_Fx = sum(r['F_total'][0] for r in member_results)
        total_Fy = sum(r['F_total'][1] for r in member_results)
        total_Fz = sum(r['F_total'][2] for r in member_results)

        return {
            'nodal_forces': nodal_forces,
            'member_results': member_results,
            'total_force': np.array([total_Fx, total_Fy, total_Fz])
        }


# =============================================================================
# INTEGRATED ANALYSIS CLASS
# =============================================================================
class JacketAnalysis:
    """
    Complete structural analysis of a 3-legged OSP jacket.

    Includes:
    - Morison wave and current loads
    - Topside interface loads
    - Self-weight
    - 3D frame FEM analysis
    - Member stress and utilization checks
    """

    def __init__(self,
                 H: float = 17.038,      # Wave height [m]
                 T: float = 9.4,         # Wave period [s]
                 d: float = 50.0,        # Water depth [m]
                 U_c: float = 1.7,       # Current velocity [m/s]
                 wave_direction: float = 38.0,  # From North [deg]
                 wave_model: str = 'Fenton',
                 # Interface loads from topside [kN]
                 F_axial: float = 25100.0,
                 F_shear: float = 2900.0,
                 M_moment: float = 0.0,  # global overturning moment (if any) [kNm]
                 M_torsion: float = 0.0, # global torsional moment (if any) [kNm]
                 # Morison coefficients
                 Cd: float = 0.7,
                 Cm: float = 2.0,
                 include_self_weight: bool = True):

        self.H = H
        self.T = T
        self.d = d
        self.U_c = U_c
        self.wave_direction = wave_direction
        self.F_axial = F_axial
        self.F_shear = F_shear
        self.M_moment = M_moment
        self.M_torsion = M_torsion
        self.Cd = Cd
        self.Cm = Cm
        self.include_self_weight = include_self_weight

        # Create structure, wave model, Morison calculator and FEM solver
        print("Creating jacket structure...")
        self.structure = JacketStructure()

        print("Creating wave model...")
        self.wave = RaschiiWave(H, T, d, U_c, wave_model)

        self.morison = MorisonCalculator(self.structure, self.wave,
                                         wave_direction, Cd, Cm)

        print("Creating FEM solver...")
        self.fem = FEMSolver(self.structure)

    def apply_interface_loads(self):
        """
        Apply topside interface loads at the three top leg nodes (A4, B4, C4).

        Assumptions:
        - Axial load is equally shared by the three legs.
        - Horizontal shear is applied in the wave direction and equally
          distributed to the three legs.
        - Global overturning and torsional moments (if any) are equally
          shared as nodal moments on the three legs.
        """
        top_nodes = self.structure.get_top_nodes()
        n_legs = len(top_nodes)

        # Convert kN/kNm to N/N·mm
        F_axial_N = self.F_axial * 1000.0
        F_shear_N = self.F_shear * 1000.0
        M_moment_Nmm = self.M_moment * 1e6
        M_torsion_Nmm = self.M_torsion * 1e6

        # Axial load equally distributed in compression (negative z-direction)
        F_axial_per_leg = F_axial_N / n_legs

        # Shear load direction (same convention as wave direction)
        theta = np.deg2rad(90.0 - self.wave_direction)
        F_shear_x = F_shear_N * np.cos(theta) / n_legs
        F_shear_y = F_shear_N * np.sin(theta) / n_legs

        for node in top_nodes:
            # [Fx, Fy, Fz, Mx, My, Mz]
            force = np.array([
                F_shear_x,                 # Fx
                F_shear_y,                 # Fy
                -F_axial_per_leg,          # Fz (downwards)
                M_torsion_Nmm / n_legs,    # Mx
                M_moment_Nmm / n_legs,     # My
                0.0                        # Mz (not used here)
            ])
            self.fem.apply_nodal_force(node, force)

        print(f"Applied interface loads: Axial={self.F_axial} kN, Shear={self.F_shear} kN")

    def apply_morison_loads(self, t: float = 0.0):
        """
        Compute and apply Morison wave loads at time t to the FEM model.
        """
        morison_results = self.morison.compute_all_morison_forces(t)

        for node_name, force in morison_results['nodal_forces'].items():
            # Only translation DOFs (Fx, Fy, Fz) are used
            force_vector = np.zeros(6)
            force_vector[:3] = force[:3]
            self.fem.apply_nodal_force(node_name, force_vector)

        total = morison_results['total_force']
        print(f"Applied Morison loads at t={t:.2f}s: "
              f"Fx={total[0]/1000.0:.1f} kN, "
              f"Fy={total[1]/1000.0:.1f} kN, "
              f"Fz={total[2]/1000.0:.1f} kN")

        return morison_results

    def apply_self_weight(self):
        """
        Apply self-weight of all members as vertical distributed loads
        converted into equivalent nodal forces.
        """
        if not self.include_self_weight:
            return

        total_weight = 0.0

        for member in self.structure.members:
            section = member['section']
            geom = self.structure.get_member_geometry(member)

            # Weight per meter [N/m]
            w = section.mass_per_m * g
            member_weight = w * geom['L']
            total_weight += member_weight

            # Equivalent nodal loads: half at each node in -Z direction
            F_weight = member_weight / 2.0

            idx1 = self.structure.node_index[member['node1']]
            idx2 = self.structure.node_index[member['node2']]

            self.fem.F_global[6*idx1 + 2] -= F_weight
            self.fem.F_global[6*idx2 + 2] -= F_weight

        print(f"Applied self-weight: Total = {total_weight/1000.0:.1f} kN")

    def run_analysis(self, t: float = 0.0) -> Dict:
        """
        Run a complete static analysis at a given time t, including:
        - Interface loads
        - Morison loads
        - Self-weight
        - Fixed supports at mudline
        - FEM solution and post-processing
        """
        print("\n" + "="*80)
        print("JACKET STRUCTURAL ANALYSIS")
        print("="*80)

        # Basic structure information
        print(f"\nStructure: {self.structure.n_nodes} nodes, "
              f"{self.structure.n_members} members")

        self.structure.section_leg.print_properties()
        self.structure.section_brace.print_properties()

        # Wave summary
        print("\n" + "-"*50)
        print("WAVE CONDITIONS")
        print("-"*50)
        self.wave.print_info()

        # Reset global load vector
        self.fem.F_global = np.zeros(self.fem.n_dof)

        # Apply loads
        print("\n" + "-"*50)
        print("APPLYING LOADS")
        print("-"*50)

        self.apply_interface_loads()
        morison_results = self.apply_morison_loads(t)
        self.apply_self_weight()

        # Apply boundary conditions (fixed at bottom nodes)
        bottom_nodes = self.structure.get_bottom_nodes()
        self.fem.apply_boundary_conditions(bottom_nodes)
        print(f"Fixed nodes: {bottom_nodes}")

        # Solve FEM system
        print("\n" + "-"*50)
        print("SOLVING FEM SYSTEM")
        print("-"*50)

        U = self.fem.solve()

        # Compute reactions at fixed nodes
        reactions = self.fem.get_reactions()

        # Compute internal forces and member utilization
        internal_forces = self.fem.get_member_internal_forces()

        # Summary report
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)

        # Support reactions
        print("\nREACTION FORCES AT FIXED NODES:")
        total_Rx = 0.0
        total_Ry = 0.0
        total_Rz = 0.0
        for node, R in reactions.items():
            print(f"  {node}: Rx={R[0]/1000.0:.1f} kN, "
                  f"Ry={R[1]/1000.0:.1f} kN, "
                  f"Rz={R[2]/1000.0:.1f} kN")
            print(f"        Mx={R[3]/1e6:.1f} kNm, "
                  f"My={R[4]/1e6:.1f} kNm, "
                  f"Mz={R[5]/1e6:.1f} kNm")
            total_Rx += R[0]
            total_Ry += R[1]
            total_Rz += R[2]

        print(f"\n  Total reactions: Rx={total_Rx/1000.0:.1f} kN, "
              f"Ry={total_Ry/1000.0:.1f} kN, "
              f"Rz={total_Rz/1000.0:.1f} kN")

        # Maximum nodal displacement
        print("\nMAXIMUM DISPLACEMENTS:")
        max_disp = 0.0
        max_disp_node = ""
        for i, node in enumerate(self.structure.node_list):
            disp = np.linalg.norm(U[6*i:6*i+3])
            if disp > max_disp:
                max_disp = disp
                max_disp_node = node
        print(f"  Maximum displacement: {max_disp:.2f} mm at node {max_disp_node}")

        # Critical members by utilization
        print("\nCRITICAL MEMBERS (Top 10 by utilization):")
        sorted_members = sorted(internal_forces,
                                key=lambda x: x['utilization'],
                                reverse=True)

        print(f"  {'Member':<25} {'Type':<10} {'σ_max':<10} "
              f"{'τ_max':<10} {'VM':<10} {'Util':<8}")
        print(f"  {'-'*73}")

        for m in sorted_members[:10]:
            print(f"  {m['member']:<25} {m['type']:<10} "
                  f"{m['sigma_max_MPa']:<10.1f} {m['tau_max_MPa']:<10.1f} "
                  f"{m['von_mises_max_MPa']:<10.1f} {m['utilization']:<8.2%}")

        # Check maximum utilization against yield
        max_util = max(m['utilization'] for m in internal_forces)
        if max_util > 1.0:
            print(f"\n  *** WARNING: Maximum utilization {max_util:.2%} exceeds yield! ***")
        else:
            print(f"\n  Maximum utilization: {max_util:.2%} (< 100%, OK)")

        return {
            'displacements': U,
            'reactions': reactions,
            'internal_forces': internal_forces,
            'morison_results': morison_results,
            'max_utilization': max_util
        }

    def run_time_history(self, t_array: np.ndarray) -> pd.DataFrame:
        """
        Run a quasi-static time-history analysis:

        For each time instant t in t_array:
        - Recompute Morison loads,
        - Re-solve the static FEM system,
        - Extract maximum axial / shear / moment in legs,
        - Extract maximum utilization.

        Returns
        -------
        DataFrame
            One row per time step with aggregated response quantities.
        """
        results = []

        print("\nRunning time history analysis...")

        for i, t in enumerate(t_array):
            if (i + 1) % 5 == 0:
                print(f"  Time step {i+1}/{len(t_array)}")

            # Reset global load vector
            self.fem.F_global = np.zeros(self.fem.n_dof)

            # Re-apply loads at time t
            self.apply_interface_loads()
            morison_res = self.morison.compute_all_morison_forces(t)

            for node_name, force in morison_res['nodal_forces'].items():
                force_vector = np.zeros(6)
                force_vector[:3] = force[:3]
                self.fem.apply_nodal_force(node_name, force_vector)

            self.apply_self_weight()

            # Apply BCs and solve
            self.fem.apply_boundary_conditions(self.structure.get_bottom_nodes())
            self.fem.solve()

            # Internal forces and utilization
            internal_forces = self.fem.get_member_internal_forces()
            max_util = max(m['utilization'] for m in internal_forces)

            # Focus on legs for axial / shear / bending envelopes
            leg_forces = [m for m in internal_forces if m['type'] == 'leg']
            max_axial_leg = max(m['Fx_max_kN'] for m in leg_forces)
            max_shear_leg = max(
                np.sqrt(m['Fy_max_kN']**2 + m['Fz_max_kN']**2)
                for m in leg_forces
            )
            max_moment_leg = max(
                np.sqrt(m['My_max_kNm']**2 + m['Mz_max_kNm']**2)
                for m in leg_forces
            )

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
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_structure_3d(structure: JacketStructure, results: Dict = None,
                      wave_direction: float = 38.0, save_path: str = None):
    """
    Plot the 3D geometry of the jacket structure.

    If 'results' is provided, this function could be extended to color-code
    members by utilization or internal force (currently it just plots geometry).
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    nodes = structure.nodes

    # Colors and line widths by member type
    type_colors = {'leg': 'darkblue', 'h_brace': 'green', 'x_brace': 'gray'}
    type_lw = {'leg': 4.0, 'h_brace': 2.5, 'x_brace': 1.5}

    # Plot members as line segments
    for member in structure.members:
        c1 = nodes[member['node1']]
        c2 = nodes[member['node2']]
        color = type_colors.get(member['type'], 'black')
        lw = type_lw.get(member['type'], 1.0)
        ax.plot([c1[0], c2[0]],
                [c1[1], c2[1]],
                [c1[2], c2[2]],
                color=color, linewidth=lw, alpha=0.8)

    # Plot nodes as scatter points with labels
    for name, coords in nodes.items():
        z = coords[2]
        color = 'blue' if z <= 0.0 else 'red'
        size = 100 if name in ['A1', 'B1', 'C1', 'A4', 'B4', 'C4'] else 50
        ax.scatter(*coords, c=color, s=size,
                   edgecolors='black', linewidth=1.5, zorder=5)
        ax.text(coords[0] + 0.5, coords[1] + 0.5, coords[2] + 1.5,
                name, fontsize=8)

    # Water surface plane z = 0
    x_surf = np.linspace(-15.0, 25.0, 10)
    y_surf = np.linspace(-20.0, 20.0, 10)
    X, Y = np.meshgrid(x_surf, y_surf)
    ax.plot_surface(X, Y, np.zeros_like(X), alpha=0.15, color='cyan')

    # Wave direction arrow
    theta_math = np.deg2rad(90.0 - wave_direction)
    ax.quiver(0.0, 0.0, 5.0,
              10.0*np.cos(theta_math), 10.0*np.sin(theta_math), 0.0,
              color='green', arrow_length_ratio=0.2, linewidth=3)
    ax.text(12.0*np.cos(theta_math), 12.0*np.sin(theta_math), 5.0,
            f'Wave: {wave_direction}°', fontsize=10, color='green')

    ax.set_xlabel('X (East) [m]')
    ax.set_ylabel('Y (North) [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3-Legged OSP Jacket Structure\nFEM Structural Analysis',
                 fontsize=14)
    ax.set_xlim([-15.0, 25.0])
    ax.set_ylim([-20.0, 20.0])
    ax.set_zlim([-50.0, 30.0])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_internal_forces(internal_forces: List[Dict], save_path: str = None):
    """
    Plot bar charts of internal forces and stresses for the jacket legs.

    This function mainly focuses on:
    - Axial force
    - Resultant shear force
    - Resultant bending moment
    - Von Mises stress
    - Utilization of all members
    """
    df = pd.DataFrame(internal_forces)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Split legs vs braces
    legs = df[df['type'] == 'leg']
    braces = df[df['type'] != 'leg']

    # Axial force (legs)
    ax = axes[0, 0]
    ax.barh(range(len(legs)), legs['Fx_max_kN'],
            color='steelblue', label='Legs')
    ax.set_yticks(range(len(legs)))
    ax.set_yticklabels(legs['member'])
    ax.set_xlabel('Axial Force [kN]')
    ax.set_title('Axial Force (Legs)')
    ax.grid(True, alpha=0.3)

    # Shear force (legs, resultant)
    ax = axes[0, 1]
    shear = np.sqrt(legs['Fy_max_kN']**2 + legs['Fz_max_kN']**2)
    ax.barh(range(len(legs)), shear, color='orange', label='Legs')
    ax.set_yticks(range(len(legs)))
    ax.set_yticklabels(legs['member'])
    ax.set_xlabel('Shear Force [kN]')
    ax.set_title('Shear Force (Legs)')
    ax.grid(True, alpha=0.3)

    # Bending moment (legs, resultant)
    ax = axes[0, 2]
    moment = np.sqrt(legs['My_max_kNm']**2 + legs['Mz_max_kNm']**2)
    ax.barh(range(len(legs)), moment, color='green', label='Legs')
    ax.set_yticks(range(len(legs)))
    ax.set_yticklabels(legs['member'])
    ax.set_xlabel('Bending Moment [kNm]')
    ax.set_title('Bending Moment (Legs)')
    ax.grid(True, alpha=0.3)

    # Von Mises stress (legs)
    ax = axes[1, 0]
    ax.barh(range(len(legs)), legs['von_mises_max_MPa'],
            color='red', label='Legs')
    ax.axvline(x=fy, color='black', linestyle='--',
               label=f'fy = {fy} MPa')
    ax.set_yticks(range(len(legs)))
    ax.set_yticklabels(legs['member'])
    ax.set_xlabel('Von Mises Stress [MPa]')
    ax.set_title('Von Mises Stress (Legs)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Utilization (all members)
    ax = axes[1, 1]
    colors = ['red' if u > 1.0 else 'steelblue' for u in df['utilization']]
    ax.barh(range(len(df)), df['utilization'], color=colors)
    ax.axvline(x=1.0, color='black', linestyle='--', label='Yield limit')
    ax.set_xlabel('Utilization [-]')
    ax.set_title('Utilization (All Members)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
    ANALYSIS SUMMARY
    ================

    Total members: {len(df)}
    - Legs: {len(legs)}
    - Braces: {len(braces)}

    Maximum values (Legs):
    - Axial: {legs['Fx_max_kN'].max():.1f} kN
    - Shear: {shear.max():.1f} kN
    - Moment: {moment.max():.1f} kNm

    Maximum Von Mises: {df['von_mises_max_MPa'].max():.1f} MPa
    Maximum Utilization: {df['utilization'].max():.2%}

    Yield Strength: {fy} MPa
    """
    ax.text(0.1, 0.9, summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_time_history(time_df: pd.DataFrame, wave: RaschiiWave,
                      save_path: str = None):
    """
    Plot time-history results from run_time_history:

    - Total Morison forces vs time,
    - Maximum internal forces in legs vs time,
    - Maximum global utilization vs time.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Morison forces
    ax = axes[0]
    ax.plot(time_df['t'], time_df['Morison_Fx_kN'], 'b-', lw=1.5, label='Fx')
    ax.plot(time_df['t'], time_df['Morison_Fy_kN'], 'g-', lw=1.5, label='Fy')
    ax.plot(time_df['t'], time_df['Morison_Fz_kN'], 'c--', lw=1.0, label='Fz')
    ax.set_ylabel('Morison Force [kN]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Wave Forces: H = {wave.H} m, T = {wave.T} s')

    # Internal forces in legs (envelopes)
    ax = axes[1]
    ax.plot(time_df['t'], time_df['Max_Axial_Leg_kN'], 'r-', lw=2,
            label='Max Axial')
    ax.plot(time_df['t'], time_df['Max_Shear_Leg_kN'], 'b-', lw=2,
            label='Max Shear')
    ax.plot(time_df['t'], time_df['Max_Moment_Leg_kNm'], 'g-', lw=2,
            label='Max Moment')
    ax.set_ylabel('Force [kN] / Moment [kNm]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Maximum Internal Forces in Legs')

    # Utilization
    ax = axes[2]
    ax.plot(time_df['t'], time_df['Max_Utilization'] * 100.0, 'r-', lw=2)
    ax.fill_between(time_df['t'], 0.0,
                    time_df['Max_Utilization'] * 100.0,
                    alpha=0.3, color='red')
    ax.axhline(y=100.0, color='black', linestyle='--', lw=2,
               label='Yield limit')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Utilization [%]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Maximum Structural Utilization')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":

    print("=" * 80)
    print("3-LEGGED OSP JACKET STRUCTURAL ANALYSIS")
    print("VERSION v6: INTEGRATED MORISON + FEM (STEEL SOLVER METHODOLOGY)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # ANALYSIS PARAMETERS
    # -------------------------------------------------------------------------
    # Wave parameters
    H = 17.038        # Wave height [m]
    T = 9.4           # Wave period [s]
    d = 50.0          # Water depth [m]
    U_c = 1.7         # Current velocity [m/s]
    wave_direction = 38.0  # Wave direction (from North) [degrees]
    wave_model = 'Fenton' if RASCHII_AVAILABLE else 'Airy'

    # Topside interface loads (from design brief or figure) [kN]
    F_axial = 25100.0     # Axial compression
    F_shear = 2900.0      # Horizontal shear
    M_moment = 0.0        # Global overturning moment [kNm]
    M_torsion = 0.0       # Global torsional moment [kNm]

    # Morison coefficients
    Cd = 0.7
    Cm = 2.0

    # -------------------------------------------------------------------------
    # CREATE ANALYSIS OBJECT
    # -------------------------------------------------------------------------
    analysis = JacketAnalysis(
        H=H, T=T, d=d, U_c=U_c,
        wave_direction=wave_direction,
        wave_model=wave_model,
        F_axial=F_axial,
        F_shear=F_shear,
        M_moment=M_moment,
        M_torsion=M_torsion,
        Cd=Cd, Cm=Cm,
        include_self_weight=True
    )

    # -------------------------------------------------------------------------
    # RUN STATIC ANALYSIS AT t = 0
    # -------------------------------------------------------------------------
    results = analysis.run_analysis(t=0.0)

    # -------------------------------------------------------------------------
    # SAVE STATIC RESULTS
    # -------------------------------------------------------------------------
    internal_df = pd.DataFrame(results['internal_forces'])
    internal_df.to_csv('internal_forces_v6.csv', index=False)
    print("\n--- Saved internal_forces_v6.csv ---")

    print("\n" + "="*80)
    print("DETAILED MEMBER INTERNAL FORCES")
    print("="*80)
    print(internal_df[['member', 'type',
                       'Fx_N1_kN', 'Fy_N1_kN', 'Fz_N1_kN',
                       'My_N1_kNm', 'Mz_N1_kNm',
                       'von_mises_max_MPa', 'utilization']].to_string())

    # -------------------------------------------------------------------------
    # TIME HISTORY ANALYSIS (3 wave periods)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("TIME HISTORY ANALYSIS (3 periods)")
    print("="*80)

    t_array = np.linspace(0.0, 3.0 * T, 31)
    time_df = analysis.run_time_history(t_array)
    time_df.to_csv('time_history_v6.csv', index=False)
    print("\n--- Saved time_history_v6.csv ---")

    print(f"\nTime history summary:")
    print(f"  Max Morison Fx: {time_df['Morison_Fx_kN'].max():.1f} kN")
    print(f"  Max Morison Fy: {time_df['Morison_Fy_kN'].max():.1f} kN")
    print(f"  Max utilization: {time_df['Max_Utilization'].max():.2%}")

    # -------------------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------------------
    print("\n--- Generating plots ---")

    plot_structure_3d(analysis.structure, results, wave_direction,
                      'structure_3d_v6.png')

    plot_internal_forces(results['internal_forces'], 'internal_forces_v6.png')

    plot_time_history(time_df, analysis.wave, 'time_history_v6.png')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
