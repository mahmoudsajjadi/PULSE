import sim
import numpy as np

print("=== EXPERIMENT A: Multi-Contingency Inertia Sensitivity Sweep ===")
results_a = {}
for H_val in [2.0, 3.2, 5.0]:
    for frac in [0.03, 0.05, 0.08]:
        r_pulse = sim.run_sfr_simulation(horizon=6.0, contingency_frac=frac, controller_mode="pulse_adaptive", H_sys=H_val, seed=42)
        r_base = sim.run_sfr_simulation(horizon=6.0, contingency_frac=frac, controller_mode="none", H_sys=H_val, seed=42)
        r_poll = sim.run_sfr_simulation(horizon=6.0, contingency_frac=frac, controller_mode="periodic_poll", poll_period=0.100, H_sys=H_val, seed=42)
            
        lift_mhz = (r_pulse["nadir_hz"] - r_base["nadir_hz"]) * 1000.0
        results_a[(H_val, frac)] = {
            "nadir_pulse": r_pulse["nadir_hz"],
            "nadir_base": r_base["nadir_hz"],
            "nadir_poll": r_poll["nadir_hz"],
            "rocof_100_pulse": r_pulse["peak_rocof_100ms"],
            "rocof_100_base": r_base["peak_rocof_100ms"],
            "msgs_pulse": r_pulse["msg_count_per_agent"],
            "lift_mhz": lift_mhz
        }
        print(f"H={H_val:.1f}s | Trip={frac*100:.0f}% | Base Nadir: {r_base['nadir_hz']:.4f} Hz | PULSE Nadir: {r_pulse['nadir_hz']:.4f} Hz | Lift: +{lift_mhz:.1f} mHz | Msgs: {r_pulse['msg_count_per_agent']:.2f}")

print("\n=== EXPERIMENT B: Heterogeneous Rack-Level Battery Fleet Analysis ===")
# Degraded capacities E_nom ~ U(25, 30) kWh, initial SoC ~ U(75%, 95%)
rng = np.random.default_rng(42)
E_nom_het = rng.uniform(25.0, 30.0, 200)
soc_init_het = rng.uniform(75.0, 95.0, 200)

fleet_het = sim.BatteryFleet(200, E_nom_kWh=E_nom_het, soc_init=soc_init_het)
p_avail_initial = fleet_het.get_available_power_MW()
r_het = sim.run_sfr_simulation(horizon=30.0, contingency_frac=0.05, controller_mode="pulse_adaptive", filter_mode="pll_1cycle", kd=4.0, kp=9.0, battery_model="dynamic_kwh", soc_seed=42, seed=42)

het_runtime = (fleet_het.energy_kWh * fleet_het.eta_dis / fleet_het.P_IT_kW) * 60.0
fade_metrics = fleet_het.compute_capacity_fade()

print(f"Heterogeneous Fleet: E_nom mean={E_nom_het.mean():.2f} kWh (range [{E_nom_het.min():.2f}, {E_nom_het.max():.2f}])")
print(f"Initial SoC mean={soc_init_het.mean():.2f}% (range [{soc_init_het.min():.2f}%, {soc_init_het.max():.2f}%])")
print(f"PULSE Nadir: {r_het['nadir_hz']:.4f} Hz | Final Mean Runtime: {r_het['runtime_min_final_mean']:.2f} min | Min Runtime: {r_het['runtime_min_final_min']:.2f} min")
print(f"Tier III Compliance: {r_het['tier3_compliance_pct']:.1f}% | Tier IV Compliance: {r_het['tier4_compliance_pct']:.1f}%")
print(f"Energy throughput per rack: {r_het['energy_throughput_per_rack_kwh']:.4f} kWh | 10-yr Capacity fade: {r_het['degradation']['fade_10yr_pct_mean']:.5f}%")
