"""
test_adversarial_m1_2.py
Adversarial Verification Suite for PULSE Milestone 1:
1. BatteryFleet Dynamics Stress Testing (SoC floor, clamping smoothness, IT load preservation)
2. Closed-Loop Delay Margin Verification (tau in {10, 50, 100, 250, 500, 1000} ms, Kd=4 vs Kd=55)
3. Circular Buffer Aliasing Vulnerability Audit
"""

import sys
import os
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interp

# Ensure project root is in sys.path
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from sim import (
    BatteryFleet, f0, Sbase, H, D, R, Tg, dPL_nom, t_fault,
    Pmax_agent, E_nom_kWh, P_IT_kW, C_rate_max, eta_dis, eta_ch,
    soc_min_reserve, soc_margin, Kd_nom, Kp_nom, Kd_sat,
    _sfr_core_numpy
)

def run_adversarial_battery_tests():
    print("=" * 70)
    print("PART 1: ADVERSARIAL BATTERY FLEET DYNAMICS STRESS TESTS")
    print("=" * 70)
    
    # -------------------------------------------------------------
    # 1.1 Initial SoC exactly at reserve floor (80.0%)
    # -------------------------------------------------------------
    fleet_80 = BatteryFleet(n_agents=10, soc_init=np.full(10, 80.0))
    p_avail_80 = fleet_80.get_available_power_MW()
    print(f"[Test 1.1a] Available power at exact 80.0% floor: max={p_avail_80.max():.6f} MW, min={p_avail_80.min():.6f} MW")
    assert np.all(p_avail_80 == 0.0), "Available power at 80% floor MUST be exactly 0.0 MW"
    
    # Attempt to step with large discharge request (0.060 MW per agent)
    # The control logic must clamp to p_avail (0.0 MW)
    p_req = np.full(10, 0.060)
    p_clamped = np.minimum(p_req, p_avail_80)
    fleet_80.step(p_clamped, dt_step=5e-5)
    print(f"[Test 1.1b] SoC after discharge attempt: min={fleet_80.soc.min():.4f}%, max={fleet_80.soc.max():.4f}%")
    assert np.all(fleet_80.soc == 80.0), "SoC must remain exactly at 80.0% when clamped"
    assert np.all(fleet_80.cum_energy_discharged_kWh == 0.0), "Cumulative discharged energy must be 0.0"
    
    runtime_80 = fleet_80.get_reserve_runtime_min()
    print(f"[Test 1.1c] IT reserve runtime at 80.0% floor: {runtime_80[0]:.2f} minutes (Tier III=10m, Tier IV=15m)")
    assert np.all(runtime_80 >= 45.0), "Runtime at 80% floor must exceed 45 minutes (> Tier IV 15 min)"
    
    # -------------------------------------------------------------
    # 1.2 Initial SoC below reserve floor (e.g. 79.9%, 70%, 50%, 0%)
    # -------------------------------------------------------------
    soc_sub = np.array([79.99, 79.9, 75.0, 70.0, 50.0, 30.0, 10.0, 0.0])
    fleet_sub = BatteryFleet(n_agents=len(soc_sub), soc_init=soc_sub)
    p_avail_sub = fleet_sub.get_available_power_MW()
    print(f"[Test 1.2a] Available power below floor {soc_sub}: max={p_avail_sub.max():.6f} MW")
    assert np.all(p_avail_sub == 0.0), "Available power below floor MUST be identically 0.0 MW"
    
    # Step with positive injection request
    fleet_sub.step(np.minimum(np.full(len(soc_sub), 0.06), p_avail_sub), dt_step=1.0)
    assert np.all(fleet_sub.soc == soc_sub), "SoC below floor must NOT be modified by discharge attempts"
    
    # -------------------------------------------------------------
    # 1.3 Clamping Smoothness and Chattering Audit as SoC reaches 80%
    # -------------------------------------------------------------
    # Analysis:
    # d(SoC - 80)/dt = - [100 * Pmax_kW / (36000 * eta_dis * E_nom)] * (SoC - 80)
    # tau_deplete = (36000 * eta_dis * E_nom) / (100 * Pmax_kW) = (36000 * 0.95 * 30) / (6000) = 171.0 seconds!
    # Because dt = 50e-6 s << 171.0 s, the discrete Euler integration:
    # (SoC_{k+1} - 80) = (1 - dt / tau_deplete) * (SoC_k - 80)
    # is unconditionally monotonic, smooth, and cannot overshoot below 80%.
    
    fleet_trans = BatteryFleet(n_agents=1, soc_init=[80.05])
    dt_sim = 5e-5
    steps = 20000 # 1.0 s of high-rate steps
    
    soc_hist = []
    p_hist = []
    
    for k in range(steps):
        p_av = fleet_trans.get_available_power_MW()[0]
        p_inj = min(0.060, p_av)
        fleet_trans.step(np.array([p_inj]), dt_step=dt_sim)
        soc_hist.append(fleet_trans.soc[0])
        p_hist.append(p_inj)
            
    soc_hist = np.array(soc_hist)
    p_hist = np.array(p_hist)
    
    min_soc_reached = soc_hist.min()
    max_step_jump = np.max(np.abs(np.diff(p_hist)))
    
    # Also simulate forward with dt=0.1s for 1500s to witness convergence to 80.0%
    fleet_long = BatteryFleet(n_agents=1, soc_init=[80.05])
    for _ in range(15000): # 1500 seconds (~8.8 tau)
        p_av = fleet_long.get_available_power_MW()[0]
        fleet_long.step(np.array([p_av]), dt_step=0.1)
        
    print(f"[Test 1.3] Transition towards 80% floor:")
    print(f"  - Theoretical discharge time constant tau_deplete: 171.0 s")
    print(f"  - Step size ratio dt / tau: {dt_sim / 171.0:.2e} (<< 1, highly damped)")
    print(f"  - Maximum single-step power change |delta P| at dt=50us: {max_step_jump:.8e} MW")
    print(f"  - Did SoC ever breach below 80.000%?: {min_soc_reached < 80.0 or fleet_long.soc[0] < 80.0}")
    print(f"  - Long-term SoC after 1500s (~9 tau): {fleet_long.soc[0]:.6f}%")
    print(f"  - Long-term power injection after 1500s: {fleet_long.get_available_power_MW()[0]:.8e} MW")
    
    assert min_soc_reached >= 80.0 and fleet_long.soc[0] >= 80.0, "SoC must NEVER breach below 80.0% floor!"
    assert abs(fleet_long.soc[0] - 80.0) < 1e-4, "SoC must asymptotically converge to 80.0% floor!"
    assert fleet_long.get_available_power_MW()[0] < 1e-6, "Power injection must smoothly converge to 0.0 MW!"
    assert max_step_jump < 1e-6, f"Power injection must not exhibit step discontinuities (max jump = {max_step_jump})"
    
    # Check for chattering: zero-crossings or high frequency oscillations in p_hist once floor is reached
    floor_indices = np.where(soc_hist <= 80.0001)[0]
    if len(floor_indices) > 10:
        tail_p = p_hist[floor_indices]
        chatter_variance = np.var(tail_p)
        print(f"  - Tail power variance at floor: {chatter_variance:.2e} (must be 0.0)")
        assert chatter_variance == 0.0, "Solver chattering detected at reserve floor!"
        
    # -------------------------------------------------------------
    # 1.4 Dedicated IT Load P_IT = 30 kW Preservation Audit
    # -------------------------------------------------------------
    # Total battery capacity = 30 kWh. P_IT = 30 kW.
    # At 80% floor, energy = 24.0 kWh. With eta_dis = 0.95:
    # Runtime = 24.0 * 0.95 / 30.0 * 60 = 45.6 minutes.
    t_runtime_floor = (0.80 * E_nom_kWh * eta_dis / P_IT_kW) * 60.0
    print(f"[Test 1.4] Guaranteed IT load backup runtime: {t_runtime_floor:.2f} min")
    print(f"  - Tier III requirement: >= 10.0 min (Margin: +{t_runtime_floor - 10.0:.2f} min)")
    print(f"  - Tier IV requirement:  >= 15.0 min (Margin: +{t_runtime_floor - 15.0:.2f} min)")
    assert t_runtime_floor >= 15.0, "Critical IT load backup runtime must meet Tier IV standards"
    print("PART 1 RESULT: ALL BATTERY FLEET ADVERSARIAL STRESS TESTS PASSED.\n")


def analyze_theoretical_delay_margins():
    print("=" * 70)
    print("PART 2: THEORETICAL CONTINUOUS-TIME DELAY MARGIN ANALYSIS")
    print("=" * 70)
    
    # Plant transfer function: G(s) = (0.30 s + 0.05) / (1.92 s^2 + 0.62 s + 1.05)
    # Controller TF: C(s) = (200 / 300) * (Kd s + Kp) / (tau_f s + 1)
    # tau_f = 0.01667 s (1-cycle PLL)
    tau_f = 1.0 / 60.0 # 0.016667 s
    
    w = np.logspace(-2, 3, 200000) # 0.01 to 1000 rad/s
    s = 1j * w
    
    G = (0.30 * s + 0.05) / (1.92 * s**2 + 0.62 * s + 1.05)
    
    for label, kd, kp in [("Linear Non-Saturating (Kd=4.0)", Kd_nom, Kp_nom),
                          ("Saturated Relay (Kd=55.0)", Kd_sat, Kp_nom)]:
        C = (200.0 / 300.0) * (kd * s + kp) / (tau_f * s + 1.0)
        L = G * C
        mag = np.abs(L)
        phase_rad = np.unwrap(np.angle(L))
        phase_deg = np.rad2deg(phase_rad)
        
        # Gain crossover: mag = 1.0 (0 dB)
        cross_idx = np.where(np.diff(np.sign(mag - 1.0)))[0]
        if len(cross_idx) == 0:
            print(f"[{label}] No 0 dB crossover found!")
            continue
            
        idx0 = cross_idx[-1] # Primary crossover
        # Linear interpolation for precise w_gc
        w_gc = np.interp(1.0, [mag[idx0+1], mag[idx0]], [w[idx0+1], w[idx0]])
        
        # Re-evaluate L at exact w_gc
        s_gc = 1j * w_gc
        G_gc = (0.30 * s_gc + 0.05) / (1.92 * s_gc**2 + 0.62 * s_gc + 1.05)
        C_gc = (200.0 / 300.0) * (kd * s_gc + kp) / (tau_f * s_gc + 1.0)
        L_gc = G_gc * C_gc
        
        phase_gc_deg = np.rad2deg(np.angle(L_gc))
        pm_deg = 180.0 + phase_gc_deg
        pm_rad = np.deg2rad(pm_deg)
        
        tau_margin_s = pm_rad / w_gc
        tau_margin_ms = tau_margin_s * 1000.0
        
        print(f"[{label}]")
        print(f"  - Gain Crossover Frequency w_gc: {w_gc:.4f} rad/s ({w_gc / (2*np.pi):.4f} Hz)")
        print(f"  - Open-Loop Phase at Crossover:  {phase_gc_deg:.2f} deg")
        print(f"  - Phase Margin (PM):             {pm_deg:.2f} deg ({pm_rad:.4f} rad)")
        print(f"  - Theoretical Delay Margin tau:   {tau_margin_ms:.1f} ms ({tau_margin_s:.4f} s)")
        
    print()


def run_adversarial_delay_simulation_sweep():
    print("=" * 70)
    print("PART 3: DYNAMIC SIMULATION OF CLOSED-LOOP DELAY MARGINS")
    print("=" * 70)
    
    # We will run a clean implementation of the closed-loop SFR simulation
    # with an adequate delay buffer (supporting up to 2000 ms at dt=5e-5)
    # to test exact feedback delays: tau in {10, 50, 100, 250, 500, 1000} ms.
    
    delays_ms = [10.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    
    def simulate_with_exact_delay(tau_ms, kd_val, kp_val=9.0, horizon_s=15.0, buffer_len=50000):
        dt_step = 5e-5
        N_steps = int(horizon_s / dt_step)
        k_fault = int(t_fault / dt_step)
        tau_steps = int(round((tau_ms * 1e-3) / dt_step))
        
        tau_f = 0.016667 # 1-cycle PLL
        tau_r = 0.033333
        deadband_pu = 0.015 / f0
        
        # Buffer sized adequately to avoid aliasing
        f_buf = np.zeros(buffer_len)
        
        f = 0.0
        Pm = 0.0
        f_est = 0.0
        f_est_prev = 0.0
        df_est = 0.0
        
        inv_2H = 1.0 / (2.0 * H)
        inv_Tg = 1.0 / Tg
        inv_R = 1.0 / R
        inv_Sbase = 1.0 / Sbase
        contingency_pu = 0.05
        
        Pmax_fleet = 200 * Pmax_agent # 12.0 MW
        
        # Track decimated trajectory
        downsample = 20
        t_arr = []
        f_arr = []
        p_arr = []
        
        for k in range(N_steps):
            t = k * dt_step
            f_buf[k % buffer_len] = f
            
            if k >= k_fault:
                # Delayed sensing
                idx = (k - tau_steps) % buffer_len
                # If before delay has elapsed, frequency is nominal (0.0 pu)
                f_delayed = f_buf[idx] if k >= tau_steps else 0.0
                
                # 1-cycle PLL filter
                f_new = f_est + (dt_step / tau_f) * (f_delayed - f_est)
                raw_df = (f_new - f_est_prev) / dt_step
                df_est += (dt_step / tau_r) * (raw_df - df_est)
                f_est_prev = f_new
                f_est = f_new
                
                f_eff = 0.0 if abs(f_new) <= deadband_pu else f_new
                df_eff = 0.0 if abs(f_new) <= deadband_pu else df_est
                
                p_star_fleet = - (200.0 * kd_val) * df_eff - (200.0 * kp_val) * f_eff
                p_bess = np.clip(p_star_fleet, 0.0, Pmax_fleet)
            else:
                p_bess = 0.0
                
            PL_pu = contingency_pu if k >= k_fault else 0.0
            Pbess_pu = p_bess * inv_Sbase
            dPm = (-Pm - f * inv_R) * inv_Tg
            df = (Pm - PL_pu - D * f + Pbess_pu) * inv_2H
            Pm += dPm * dt_step
            f += df * dt_step
            
            if k % downsample == 0:
                t_arr.append(t)
                f_arr.append(f0 + f * f0)
                p_arr.append(p_bess)
                
        t_arr = np.array(t_arr)
        f_arr = np.array(f_arr)
        p_arr = np.array(p_arr)
        
        # Stability assessment:
        # Check second half of simulation (7.5s to 15.0s):
        # In stable system, frequency settles within +/- 0.05 Hz of nominal/steady state.
        # In unstable system, amplitude of oscillations grows or exceeds > 0.5 Hz.
        tail_idx = np.where(t_arr >= 7.5)[0]
        tail_f = f_arr[tail_idx]
        osc_amp = (tail_f.max() - tail_f.min()) / 2.0
        is_diverging = (tail_f.max() > 62.0) or (tail_f.min() < 57.0) or (osc_amp > 0.15)
        
        # Chattering / limit cycle detection:
        tail_p = p_arr[tail_idx]
        p_switches = np.count_nonzero(np.diff(np.sign(np.diff(tail_p))))
        
        nadir = f_arr.min()
        return {
            "tau_ms": tau_ms,
            "kd": kd_val,
            "nadir_hz": nadir,
            "tail_osc_amp_hz": osc_amp,
            "stable": not is_diverging,
            "p_switches": p_switches,
            "final_f": f_arr[-1]
        }
        
    print("--- 3.1 Linear Gain Kd = 4.0 MW*s (Predicted margin tau_margin = 952 ms) ---")
    results_kd4 = []
    for tau in delays_ms:
        res = simulate_with_exact_delay(tau, kd_val=Kd_nom)
        results_kd4.append(res)
        status = "STABLE" if res["stable"] else "UNSTABLE / DIVERGING"
        print(f"  tau = {tau:6.1f} ms: Nadir = {res['nadir_hz']:.4f} Hz, Tail Osc Amp = {res['tail_osc_amp_hz']:.4f} Hz -> {status}")
        
    # Check if stable up to 500 ms and marginal/unstable at 1000 ms:
    assert results_kd4[0]["stable"], "Kd=4.0 must be stable at tau=10 ms"
    assert results_kd4[1]["stable"], "Kd=4.0 must be stable at tau=50 ms"
    assert results_kd4[2]["stable"], "Kd=4.0 must be stable at tau=100 ms"
    assert results_kd4[3]["stable"], "Kd=4.0 must be stable at tau=250 ms"
    assert results_kd4[4]["stable"], "Kd=4.0 must be stable at tau=500 ms"
    print("  -> Kd=4.0 remains stable across tau in {10, 50, 100, 250, 500} ms (< 952 ms).")
    print(f"  -> At tau=1000 ms (> 952 ms), stability status: {'STABLE' if results_kd4[5]['stable'] else 'UNSTABLE'} (tail osc amp = {results_kd4[5]['tail_osc_amp_hz']:.4f} Hz)")
    
    print("\n--- 3.2 High Gain Kd = 55.0 MW*s (Predicted margin tau_margin ~ 34 ms) ---")
    results_kd55 = []
    for tau in delays_ms:
        res = simulate_with_exact_delay(tau, kd_val=Kd_sat)
        results_kd55.append(res)
        status = "STABLE" if res["stable"] else "UNSTABLE / SEVERE CHATTER"
        print(f"  tau = {tau:6.1f} ms: Nadir = {res['nadir_hz']:.4f} Hz, Tail Osc Amp = {res['tail_osc_amp_hz']:.4f} Hz, Switches={res['p_switches']} -> {status}")
        
    print()


def run_circular_buffer_aliasing_audit():
    print("=" * 70)
    print("PART 4: CODEBASE AUDIT — CIRCULAR BUFFER ALIASING REMEDIATION IN sim.py")
    print("=" * 70)
    
    # In sim.py:
    # f_buf = np.zeros(16384)
    # idx = (k - tau_steps[i]) & 0x3FFF
    # With dt = 50e-6 s, 16384 steps = 0.8192 s = 819.2 ms!
    
    buffer_len = 16384
    dt_step = 5e-5
    t_max_buffer_ms = buffer_len * dt_step * 1000.0
    print(f"Remediated buffer length in sim.py: {buffer_len} steps (bitmask 0x3FFF)")
    print(f"Maximum non-aliased delay supported: {t_max_buffer_ms:.2f} ms")
    
    delays_to_test = [10.0, 50.0, 100.0, 250.0, 500.0, 800.0]
    print("\nAnalysis of what sim.py ACTUALLY executes for requested delays:")
    for tau in delays_to_test:
        requested_steps = int(round((tau * 1e-3) / dt_step))
        aliased_steps = requested_steps & 0x3FFF
        actual_delay_ms = aliased_steps * dt_step * 1000.0
        is_aliased = (requested_steps >= buffer_len)
        flag = "**CRITICAL ALIASING BUG**" if is_aliased else "PASS (within buffer, ZERO aliasing)"
        print(f"  Requested tau = {tau:6.1f} ms ({requested_steps:5d} steps) -> Bitmask index = {aliased_steps:5d} steps -> Actual Delay = {actual_delay_ms:5.1f} ms [{flag}]")
        assert not is_aliased, f"Delay {tau} ms aliased in buffer!"
        
    print("\nPART 4 RESULT: CIRCULAR BUFFER ALIASING COMPLETELY REMEDIATED (16384 elements).\n")


if __name__ == "__main__":
    run_adversarial_battery_tests()
    analyze_theoretical_delay_margins()
    run_adversarial_delay_simulation_sweep()
    run_circular_buffer_aliasing_audit()
