# PULSE: Primary-Frequency UPS Local Sub-Second Event-Triggered Control

Official simulation, verification, and figure-generation repository for the paper:

> **PULSE: Fast Event-Triggered Grid Frequency and Synthetic Inertia Support from Distributed Data Center UPS Fleets**  
> *Seyed Mahmoud Sajjadi Mohammadabadi*  
> Department of Computer Science and Engineering, University of Nevada, Reno  
> Target Journal: *Sustainable Energy, Grids and Networks (SEGAN)*, Elsevier

---

## Overview

**PULSE** (**P**rimary-frequency **U**PS **L**ocal **S**ub-second **E**vent-triggered control) is a communication-aware decentralized architecture that enables distributed rack-level uninterruptible power supply (UPS) batteries ($N=200$ agents, aggregated $12\,\text{MW}$) to provide synthetic inertia and primary frequency containment without requiring a central dispatcher.

The name **PULSE** embodies both its physical mechanism—injecting an ultra-fast sub-second pulse of active power into the grid to arrest frequency decline immediately following a contingency—and its algorithmic structure:
- **P**rimary-frequency: Autonomous primary regulation arresting frequency nadir.
- **U**PS: Distributed rack-level energy storage assets in Tier III/IV data center facilities.
- **L**ocal: $100\%$ local sensing and actuation requiring zero inter-rack communication.
- **S**ub-second: Rapid initial active power dispatch within tens of milliseconds.
- **E**vent-triggered: Communication-minimizing adaptive threshold triggering mechanism.

![Framework Overview](figs/fig0_overview.png)

### Key Highlights
- **Realistic Filtering & Inertial Support**: Operates seamlessly under realistic P-class PMU ($2$-cycle, $\tau_f=33.3\,\text{ms}$) and inverter PLL ($1$-cycle, $\tau_f=16.7\,\text{ms}$) measurement filtering with $\sigma_f=0.5\,\text{mHz}$ Gaussian noise and a $\pm 15\,\text{mHz}$ deadband. Evaluates synthetic inertia and RoCoF over standardized $100\,\text{ms}$ and $500\,\text{ms}$ observation windows, providing an equivalent virtual inertia constant $H_v = 0.40\,\text{s}$.
- **Decentralized Adaptive Event Triggering**: Completely eliminates idle-state communication traffic during normal operation (0 msgs/s under ambient frequency drift) and reduces post-contingency message count by up to $16.1\%$ compared to fixed-threshold event-triggered control (FETC).
- **Communication Imperfection Resilience**: Maintains grid frequency nadir within $28\,\text{mHz}$ across up to $30\%$ packet loss and log-normal network jitter (capped at $50\,\text{ms}$).
- **Energy Model & IT Reliability Protection**: Enforces an 80% minimum state-of-charge ($\text{SoC}_{\min}$) threshold with a 10% linear taper to safeguard critical data center runtime, fully preserving the required 40-minute IT backup duration under Tier III/IV standards.
- **Rigorous Verification & Monte Carlo Robustness**: Validated against benchmark baselines (unsupported grid, local droop/PD, and periodic polling) across 50 Monte Carlo seeds, extensive $3\%\text{--}8\%$ generation trip sweeps, and heterogeneous battery fleets.

---

## Repository Structure

| File / Folder | Description |
| :--- | :--- |
| `sim.py` | Complete numerical simulation framework reproducing all dynamical responses, parametric sweeps, battery degradation calculations, and Figures 1–5. |
| `fig_overview.py` | Generates system architectural block diagrams (`figs/fig0_overview.png`, `figs/graphical_abstract.png`). |
| `test_pulse_e2e.py` | Comprehensive test suite containing 152 automated end-to-end tests validating physics, numerical bounds, and stability. |
| `results_summary.txt` | Machine-readable log of simulated metrics across baseline, contingency, and packet loss regimes. |
| `figs/` | High-resolution publication-quality PNG and vector PDF figures. |

---

## Reproducing the Results

### Requirements
- Python 3.9+
- `numpy`, `matplotlib`, `pytest`

Install dependencies:
```bash
pip install numpy matplotlib pytest
```

### Running Simulations & Tests
To regenerate the system overview and graphical abstract:
```bash
python fig_overview.py
```

To run the full simulation and regenerate all figures:
```bash
python sim.py
```

To run the automated verification test suite:
```bash
pytest test_pulse_e2e.py
```

---

## Citation

```bibtex
@article{sajjadi2026pulse,
  title   = {{PULSE}: Fast Event-Triggered Grid Frequency and Synthetic Inertia Support from Distributed Data Center {UPS} Fleets},
  author  = {Sajjadi Mohammadabadi, Seyed Mahmoud},
  journal = {Sustainable Energy, Grids and Networks},
  year    = {2026},
  url     = {https://github.com/mahmoudsajjadi/PULSE}
}
```
