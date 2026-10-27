"""
Independent Challenger Stress Testing Harness for PULSE Simulation Core
========================================================================
Empirical stress tests for:
1. Severe contingencies (up to 20% generation loss, 60 MW)
2. Latency jitter and packet drop rates (up to 50%)
3. Variable fleet sizing (N = 50, 100, 200, 400)
4. Numerical stability (Euler dt limits, NaN/Inf checks, state divergence, high gains)
5. Analytical swing equation equivalence for virtual inertia H_v
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import numpy as np

# Ensure repository root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import sim

def print_banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def test_severe_contingencies():
    print_banner("TEST 1: Severe Contingencies (3% to 25% Generation Loss)")
    contingencies = [0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.25]
    results = []
    
    for c in contingencies:
        loss_mw = c * sim.Sbase
        t0 = time.time()
        res = sim.run_sfr_simulation(
            horizon=30.0,
            contingency_frac=c,
            controller_mode="pulse_adaptive",
            filter_mode="pll_1cycle",
            battery_model="dynamic_kwh",
            seed=42
        )
        dur = time.time() - t0
        
        # Check for numerical validity
        f_nan = np.isnan(res["f_hz"]).any() or np.isinf(res["f_hz"]).any()
        p_nan = np.isnan(res["p_mw"]).any() or np.isinf(res["p_mw"]).any()
        soc_nan = np.isnan(res["soc_pct"]).any() or np.isinf(res["soc_pct"]).any()
        
        # Recovery check: final frequency at 30s should be stable (governor recovers)
        f_final = res["f_hz"][-1]
        p_final = res["p_mw"][-1]
        p_peak = res["p_mw"].max()
        soc_min = res["soc_pct"].min()
        runtime_min = res["runtime_min"].min()
        
        status = "PASS" if not (f_nan or p_nan or soc_nan) else "FAIL_NAN"
        if f_final < 55.0 or f_final > 65.0:
            status = "FAIL_UNSTABLE"
            
        print(f"Contingency {c*100:4.1f}% ({loss_mw:5.1f} MW) | Nadir: {res['nadir_hz']:6.3f} Hz | "
              f"Peak P: {p_peak:5.2f} MW | f_final: {f_final:6.3f} Hz | "
              f"Min SoC: {soc_min:5.1f}% | Min Runtime: {runtime_min:5.1f}m | [{status}] ({dur:.2f}s)")
        
        results.append({
            "contingency": c, "loss_mw": loss_mw, "nadir_hz": res["nadir_hz"],
            "p_peak": p_peak, "f_final": f_final, "min_soc": soc_min,
            "min_runtime": runtime_min, "status": status
        })
    return results

def test_packet_loss_and_jitter():
    print_banner("TEST 2: Latency Jitter and High Packet Drop Rates (0% to 50%)")
    packet_drops = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    jitters = [0.0, 10.0, 50.0, 100.0]
    
    print("\n--- Subtest 2A: Fixed Seed Varying Packet Drop ---")
    base_res = sim.run_sfr_simulation(
        horizon=10.0, contingency_frac=0.05, controller_mode="pulse_adaptive",
        p_drop=0.0, jitter_scale_ms=0.0, seed=42
    )
    base_nadir = base_res["nadir_hz"]
    base_msgs = base_res["msg_count_total"]
    print(f"Baseline (p_drop=0.0, jitter=0.0): Nadir = {base_nadir:.6f} Hz, Msgs = {base_msgs}")
    
    drop_results = []
    for pd in packet_drops:
        res = sim.run_sfr_simulation(
            horizon=10.0, contingency_frac=0.05, controller_mode="pulse_adaptive",
            p_drop=pd, jitter_scale_ms=0.0, seed=42
        )
        diff_nadir = res["nadir_hz"] - base_nadir
        diff_msgs = res["msg_count_total"] - base_msgs
        drop_results.append((pd, res["nadir_hz"], res["msg_count_total"], diff_nadir, diff_msgs))
        print(f"p_drop = {pd:4.2f} | Nadir = {res['nadir_hz']:9.6f} Hz (delta = {diff_nadir:+.6f}) | "
              f"Msgs = {res['msg_count_total']:6.0f} (delta = {diff_msgs:+6.0f})")

    print("\n--- Subtest 2B: Varying Jitter Scale ---")
    jitter_results = []
    for jm in jitters:
        res = sim.run_sfr_simulation(
            horizon=10.0, contingency_frac=0.05, controller_mode="pulse_adaptive",
            p_drop=0.0, jitter_scale_ms=jm, seed=42
        )
        diff_nadir = res["nadir_hz"] - base_nadir
        diff_msgs = res["msg_count_total"] - base_msgs
        jitter_results.append((jm, res["nadir_hz"], res["msg_count_total"], diff_nadir, diff_msgs))
        print(f"jitter = {jm:5.1f} ms | Nadir = {res['nadir_hz']:9.6f} Hz (delta = {diff_nadir:+.6f}) | "
              f"Msgs = {res['msg_count_total']:6.0f} (delta = {diff_msgs:+6.0f})")
              
    return drop_results, jitter_results

def test_variable_fleet_sizing():
    print_banner("TEST 3: Variable Fleet Sizing (N = 50, 100, 200, 400)")
    fleet_sizes = [50, 100, 200, 400]
    
    # Check if run_sfr_simulation accepts n_agents parameter
    import inspect
    sig = inspect.signature(sim.run_sfr_simulation)
    has_n_agents_param = "n_agents" in sig.parameters
    print(f"API Check: run_sfr_simulation() has 'n_agents' parameter? {has_n_agents_param}")
    
    results = []
    for N in fleet_sizes:
        print(f"\nTesting N = {N}:")
        # Try direct parameter if supported
        if has_n_agents_param:
            try:
                res = sim.run_sfr_simulation(horizon=10.0, n_agents=N, seed=42)
                print(f"  Passed via n_agents={N} parameter: Nadir = {res['nadir_hz']:.4f} Hz, H_v = {res['h_v_s']:.4f} s")
                results.append((N, "param_pass", res["nadir_hz"], res["h_v_s"]))
            except Exception as e:
                print(f"  Error with n_agents={N} parameter: {e}")
                results.append((N, "param_error", str(e), 0.0))
        else:
            # Test monkey-patching sim.n_agents
            orig_n = sim.n_agents
            try:
                sim.n_agents = N
                # Also need to re-evaluate H_v_nom
                res = sim.run_sfr_simulation(horizon=10.0, seed=42)
                print(f"  Monkey-patched sim.n_agents={N}: Nadir = {res['nadir_hz']:.4f} Hz, H_v = {res['h_v_s']:.4f} s, Peak P = {res['p_mw'].max():.2f} MW")
                results.append((N, "monkeypatch_pass", res["nadir_hz"], res["h_v_s"], res["p_mw"].max()))
            except Exception as e:
                print(f"  Monkey-patching failed for N={N}: {e}")
                results.append((N, "monkeypatch_error", str(e), 0.0, 0.0))
            finally:
                sim.n_agents = orig_n
                
    return has_n_agents_param, results

def test_numerical_stability_and_solvers():
    print_banner("TEST 4: Numerical Stability, Euler dt Margins, High Gains, & Boundary Conditions")
    
    print("\n--- Subtest 4A: dt Step-Size Stability Limit ---")
    dt_candidates = [1e-5, 2.5e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    dt_results = []
    for dt_val in dt_candidates:
        try:
            # Use small downsample_factor so array length doesn't overflow
            ds = max(1, int(1e-3 / dt_val))
            res = sim.run_sfr_simulation(
                horizon=5.0, dt=dt_val, downsample_factor=ds,
                controller_mode="pulse_adaptive", filter_mode="pll_1cycle", seed=42
            )
            has_nan = np.isnan(res["f_hz"]).any() or np.isinf(res["f_hz"]).any()
            f_min = res["nadir_hz"]
            f_max = res["f_hz"].max()
            status = "STABLE" if (not has_nan and 55.0 < f_min < 65.0 and 55.0 < f_max < 65.0) else "UNSTABLE/NAN"
            print(f"  dt = {dt_val*1e6:7.1f} us ({dt_val*1000:5.2f} ms) | Nadir: {f_min:6.3f} Hz | Max f: {f_max:6.3f} Hz | [{status}]")
            dt_results.append((dt_val, status, f_min, f_max))
        except Exception as e:
            print(f"  dt = {dt_val*1e6:7.1f} us | CRASH: {e}")
            dt_results.append((dt_val, f"CRASH: {e}", 0, 0))

    print("\n--- Subtest 4B: Extreme Control Gains (Kd, Kp) ---")
    gain_cases = [
        (4.0, 9.0, "nominal linear"),
        (55.0, 9.0, "saturated relay"),
        (100.0, 9.0, "extreme Kd=100"),
        (200.0, 9.0, "ultra Kd=200"),
        (4.0, 50.0, "extreme Kp=50"),
        (55.0, 50.0, "extreme Kd=55, Kp=50"),
        (0.0, 0.0, "zero gain")
    ]
    gain_results = []
    for kd_v, kp_v, desc in gain_cases:
        try:
            res = sim.run_sfr_simulation(
                horizon=10.0, kd=kd_v, kp=kp_v,
                controller_mode="pulse_adaptive", filter_mode="pll_1cycle", seed=42
            )
            has_nan = np.isnan(res["f_hz"]).any()
            peak_p = res["p_mw"].max()
            print(f"  Kd={kd_v:5.1f}, Kp={kp_v:5.1f} ({desc:22s}) | Nadir: {res['nadir_hz']:6.3f} Hz | Peak P: {peak_p:5.2f} MW | NaN: {has_nan}")
            gain_results.append((kd_v, kp_v, desc, res['nadir_hz'], peak_p, has_nan))
        except Exception as e:
            print(f"  Kd={kd_v:5.1f}, Kp={kp_v:5.1f} | CRASH: {e}")
            gain_results.append((kd_v, kp_v, desc, 0, 0, True))

    print("\n--- Subtest 4C: Boundary Contingencies (0%, Negative Load Drop, 50% Extreme Trip) ---")
    boundary_cases = [0.0, -0.05, 0.50]
    for bc in boundary_cases:
        res = sim.run_sfr_simulation(horizon=5.0, contingency_frac=bc, seed=42)
        has_nan = np.isnan(res["f_hz"]).any()
        print(f"  Contingency frac = {bc:+.2f} | Nadir: {res['nadir_hz']:6.3f} Hz | Peak P: {res['p_mw'].max():.2f} MW | Min P: {res['p_mw'].min():.2f} MW | NaN: {has_nan}")

    return dt_results, gain_results

def test_synthetic_inertia_analytical_fidelity():
    print_banner("TEST 5: Analytical Swing Equation Equivalence for H_v")
    
    # Analytical derivation
    # df/dt at t_fault+ before governor and damping act:
    # df_pu/dt = - dPL_pu / (2 * H_eff)
    # in Hz/s: RoCoF_0 = - (dPL / (2 * H_eff * Sbase)) * f0
    Sbase = sim.Sbase
    f0 = sim.f0
    H = sim.H
    dPL = 0.05 * Sbase # 15 MW
    
    # Theoretical unsupported:
    rocof_unsupported_theory = - (dPL / (2.0 * H * Sbase)) * f0 # -0.46875 Hz/s
    
    # Linear virtual inertia:
    # Kd = 4.0, N = 200 -> H_v = (200 * 4.0) / (2 * 300) = 1.33333 s
    H_v = (sim.n_agents * 4.0) / (2.0 * Sbase)
    H_eff = H + H_v
    rocof_linear_theory = - (dPL / (2.0 * H_eff * Sbase)) * f0 # -0.33088 Hz/s
    theoretical_reduction = 100.0 * (1.0 - abs(rocof_linear_theory) / abs(rocof_unsupported_theory))
    
    print(f"Analytical Reference Values:")
    print(f"  H_base = {H:.3f} s, H_v = {H_v:.4f} s, H_eff = {H_eff:.4f} s (+{H_v/H*100:.2f}%)")
    print(f"  Theoretical RoCoF_0 Unsupported: {rocof_unsupported_theory:.5f} Hz/s")
    print(f"  Theoretical RoCoF_0 Linear:      {rocof_linear_theory:.5f} Hz/s")
    print(f"  Theoretical Reduction:           {theoretical_reduction:.2f}%")
    
    # Now simulate with idealized filter and zero/minimal latency
    print("\n--- Empirical Verification with sim.py ---")
    
    # Run unsupported baseline
    res_unsupported = sim.run_sfr_simulation(
        horizon=2.0, dt=5e-5, contingency_frac=0.05,
        controller_mode="none", filter_mode="idealized", seed=42
    )
    
    # Run continuous linear droop with zero deadband, zero filter
    res_linear = sim.run_sfr_simulation(
        horizon=2.0, dt=5e-5, contingency_frac=0.05,
        controller_mode="industry_droop", filter_mode="idealized",
        kd=4.0, kp=9.0, deadband_hz=0.0, seed=42
    )
    
    # Run pulse adaptive with idealized filter
    res_pulse = sim.run_sfr_simulation(
        horizon=2.0, dt=5e-5, contingency_frac=0.05,
        controller_mode="pulse_adaptive", filter_mode="idealized",
        kd=4.0, kp=9.0, deadband_hz=0.0, seed=42
    )

    # Compute immediate slope right after actuation starts
    # Since idealized has network delay tau ~ 1-8 ms, let's observe when actuation begins
    t_fault = 0.100
    
    # In zoom slice:
    t_zoom_un = res_unsupported["t_zoom"]
    f_zoom_un = res_unsupported["f_zoom"]
    t_zoom_lin = res_linear["t_zoom"]
    f_zoom_lin = res_linear["f_zoom"]
    p_zoom_lin = res_linear["p_zoom"]
    
    # Calculate RoCoF in zoom slice over [t_act, t_act + 50ms]
    # Find when power exceeds 5 MW in linear mode
    idx_act = np.where(p_zoom_lin > 1.0)[0]
    if len(idx_act) > 0:
        k_act = idx_act[0]
        t_act = t_zoom_lin[k_act]
        # measure slope over next 20 ms
        k_eval = np.where(t_zoom_lin >= t_act + 0.020)[0][0]
        dt_eval = t_zoom_lin[k_eval] - t_zoom_lin[k_act]
        df_un = (f_zoom_un[k_eval] - f_zoom_un[k_act]) / dt_eval
        df_lin = (f_zoom_lin[k_eval] - f_zoom_lin[k_act]) / dt_eval
        emp_reduction = 100.0 * (1.0 - abs(df_lin) / abs(df_un))
        print(f"Zoom Window Evaluation [t={t_act*1000:.1f}ms to {(t_act+dt_eval)*1000:.1f}ms]:")
        print(f"  Empirical Unsupported RoCoF: {df_un:.5f} Hz/s (Theory: {rocof_unsupported_theory:.5f})")
        print(f"  Empirical Linear RoCoF:      {df_lin:.5f} Hz/s (Theory: {rocof_linear_theory:.5f})")
        print(f"  Empirical Reduction:         {emp_reduction:.2f}% (Theory: {theoretical_reduction:.2f}%)")
    
    print(f"\n100 ms Sliding Window Metrics:")
    print(f"  Unsupported peak RoCoF_100ms: {res_unsupported['peak_rocof_100ms']:.5f} Hz/s")
    print(f"  Linear peak RoCoF_100ms:      {res_linear['peak_rocof_100ms']:.5f} Hz/s")
    print(f"  Pulse peak RoCoF_100ms:       {res_pulse['peak_rocof_100ms']:.5f} Hz/s")
    
    print(f"\n500 ms Sliding Window Metrics:")
    print(f"  Unsupported peak RoCoF_500ms: {res_unsupported['peak_rocof_500ms']:.5f} Hz/s")
    print(f"  Linear peak RoCoF_500ms:      {res_linear['peak_rocof_500ms']:.5f} Hz/s")
    print(f"  Pulse peak RoCoF_500ms:       {res_pulse['peak_rocof_500ms']:.5f} Hz/s")

if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("STARTING EMPIRICAL CHALLENGER STRESS HARNESS")
    print("=" * 70)
    
    test_severe_contingencies()
    test_packet_loss_and_jitter()
    test_variable_fleet_sizing()
    test_numerical_stability_and_solvers()
    test_synthetic_inertia_analytical_fidelity()
    
    print("\n" + "=" * 70)
    print(f"ALL CHALLENGER STRESS TESTS COMPLETED IN {time.time() - t_start:.2f} s")
    print("=" * 70)
