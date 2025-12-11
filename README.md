# 🌊 Offshore Jacket Structure Analysis Tool

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive **GUI-based structural analysis tool** for offshore jacket structures, combining **Morison equation** hydrodynamic loading with **3D Finite Element Method (FEM)** analysis.

![Structure Preview](docs/preview.png)

## ✨ Features

- 🏗️ **Customizable Geometry**: Define any multi-legged jacket structure with user-specified nodes and members
- 🌊 **Advanced Wave Models**: Supports Airy, Stokes (up to 5th order), and Fenton stream function via [raschii](https://github.com/torebutlin/raschii) library
- 📊 **Detailed Morison Force Output**: Separate drag, inertia, and total hydrodynamic forces
- 🔍 **Phase Scan Analysis**: Automatically find the critical wave phase for maximum loading
- 🎯 **FEM Analysis**: 3D Timoshenko beam elements with full internal force and stress output
- 📈 **Stress Utilization Check**: Von Mises stress calculation with yield strength comparison
- 🖼️ **3D Visualization**: Interactive structure plots with utilization color-coding
- 💾 **Export Results**: CSV export for post-processing

## 📋 Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Tkinter (usually included with Python)
- Raschii (optional, for nonlinear wave theories)

## 🚀 Quick Start

### Option 1: Double-click to run (Windows)
```
Double-click: RunGUI_v2.bat
```

### Option 2: Command line
```bash
cd DesignProject
pip install -r requirements.txt
python JacketAnalysisGUI_v2.py
```

## 📐 Coordinate System

```
        Z (Up)
        ↑
        |    
        |      
        +--------→ X (East)
       /
      /
     ↓
    Y (North)
```

| Axis | Direction |
|------|-----------|
| **X** | East (+) / West (-) |
| **Y** | North (+) / South (-) |
| **Z** | Up (+) / Down (-) |
| **Z=0** | Mean Water Level (MWL) |

## 🌊 Wave Theory Limitations

| Model | Max Order | Best For |
|-------|-----------|----------|
| Airy | 1st | Small amplitude (H/L < 0.01) |
| Stokes | 5th | Moderate waves (H/L < 0.06) |
| Fenton | 5-20 | Steep waves (H/L > 0.06) |
| Auto | Varies | Automatic best selection |

**Breaking Wave Limits:**
- Deep water: H/L < 0.142
- Shallow water: H/d < 0.78

## 📊 Output Data

### Morison Forces (Pure Hydrodynamic)
- **Drag Force**: Proportional to |u|×u, in phase with velocity
- **Inertia Force**: Proportional to du/dt, 90° phase lead
- **Total Morison**: Vector sum of drag and inertia

### FEM Results
- Support reactions
- Nodal displacements
- Member internal forces (axial, shear, bending, torsion)
- Von Mises stress and utilization ratio

## ⚠️ Assumptions & Limitations

### Structural
- Circular tubular (thin-wall) sections only
- Linear elastic material behavior
- Static/quasi-static analysis
- Rigid connections
- Fixed supports at mudline

### Analysis Limitations
This tool does **NOT** consider:
- ❌ Buckling (Euler, local, lateral-torsional)
- ❌ P-delta effects
- ❌ Dynamic response / DAF
- ❌ Fatigue analysis
- ❌ Corrosion allowance
- ❌ Marine growth
- ❌ Partial safety factors

**For final design, use certified software and follow applicable codes (API RP 2A, ISO 19902, NORSOK, etc.)**

## 📁 Project Structure

```
DesignProject/
├── JacketAnalysisGUI_v2.py   # Main GUI application (v2 - recommended)
├── JacketAnalysisGUI.py      # Original GUI version
├── UserDefinedAPP.py         # Core analysis module
├── requirements.txt          # Python dependencies
├── RunGUI_v2.bat            # Windows launcher (v2)
├── RunAnalysisGUI.bat       # Windows launcher (original)
└── README.md                # This file
```

## 🎓 Theory Reference

### Morison Equation
```
F = ½ρCdD|u|u + ρCmA(du/dt)
```
Where:
- ρ = seawater density (1025 kg/m³)
- Cd = drag coefficient (0.6-1.2)
- Cm = inertia coefficient (1.5-2.0)
- D = member diameter
- u = water particle velocity (perpendicular to member)

### FEM Formulation
- 3D Timoshenko beam elements
- 12 DOF per element (6 per node)
- Includes shear deformation effects

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Disclaimer**: This tool is for educational and preliminary design purposes only. Always verify results with certified engineering software for actual structural design.

