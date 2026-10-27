"""
test_challenger_m1_it2_harness.py
Empirical Challenger 2 Verification Suite (Milestone 1, Iteration 2)
Focus Areas:
1. Battery Physics & Reserve Dynamics across seeds and contingency fractions (dPL = 0.03, 0.05, 0.08)
2. Pareto Sweep Monotonicity and Trade-off Validity (Nadir vs Messages across eta)
3. Trajectory Matrix Integrity & Sum Conservation (p_individual shape, non-triviality, sum ≈ p_fleet)
4. Comprehensive Verification of Integrity & Edge Cases
"""

import sys
import os
import time
import numpy as np

# Ensure project root is in sys.path
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from sim import (
    run_sfr_simulation, run_soc_constrained,
    Pmax_agent, E_nom_kWh, P_IT_kW, eta_dis, soc_min_reserve, soc_margin,
    Kd_nom, Kp_nom, f0, Sbase
)

def banner(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)

# =====================================================================
# TEST 1: Battery Physics, Dynamic SoC Availability & Reserve Runtime
# =====================================================================
def test_battery_physics_adversarial():
    banner("TEST 1: Battery Physics, Dynamic SoC Availability & Reserve Runtime")
    seeds = [42, 101, 202, 999]
    dpls = [0.03, 0.05, 0.08]
    
    print("\n--- 1.1 Multi-Seed & Multi-Contingency Sweep ---")
    results = {}
    
    for s in seeds:
        for dpl in dpls:
            key = (s, dpl)
            t0 = time.time()
            res = run_soc_constrained(
                dPL_frac=dpl,
                seed=s,
                horizon=10.0,
                Kd_val=4.0,
                Kp_val=9.0
            )
            dt_sim = time.time() - t0
            
            avail_init = res["soc_mean_avail_pct"]
            frac_res_init = res["soc_frac_fully_reserved_pct"]
            t3 = res["tier3_compliance_pct"]
            t4 = res["tier4_compliance_pct"]
            
            # Runtime metrics
            rt_final = res["runtime_min"][-1, :].copy()
            rt_mean = float(np.mean(rt_final))
            rt_min = float(np.min(rt_final))
            
            # SoC metrics
            soc_init = res["soc_pct"][0, :].copy()
            soc_final = res["soc_pct"][-1, :].copy()
            soc_drop_mean = float(np.mean(soc_init - soc_final))
            soc_min = float(np.min(soc_final))
            
            # Discharged energy
            e_dis_mean = res["degradation"]["energy_discharged_mean_kWh"]
            e_dis_total = res["degradation"]["energy_discharged_total_kWh"]
            nadir_val = res["nadir_hz"]
            
            del res
            import gc
            gc.collect()
            
            results[key] = {
                "avail_init": avail_init,
                "frac_res_init": frac_res_init,
                "t3": t3,
                "t4": t4,
                "rt_mean": rt_mean,
                "rt_min": rt_min,
                "soc_drop_mean": soc_drop_mean,
                "soc_min": soc_min,
                "soc_init": soc_init,
                "soc_final": soc_final,
                "e_dis_mean": e_dis_mean,
                "e_dis_total": e_dis_total,
                "nadir": nadir_val
            }
            
            print(f"Seed {s:4d} | dPL {dpl:.2f} ({(dpl*300):4.1f} MW) | Avail={avail_init:6.3f}% | "
                  f"Drop={soc_drop_mean:6.4f}% | RtMean={rt_mean:6.2f}m | RtMin={rt_min:6.2f}m | "
                  f"E_dis={e_dis_total:7.2f}kWh | Nadir={nadir_val:.4f}Hz ({dt_sim:.2f}s)")
            
    print("\n--- 1.2 Verification of Dynamic Variation Across Seeds ---")
    avail_by_seed = [results[(s, 0.05)]["avail_init"] for s in seeds]
    print(f"Initial SoC Availability across seeds {seeds}: {[f'{a:.3f}%' for a in avail_by_seed]}")
    # Verify that availability is not static across seeds
    avail_std = np.std(avail_by_seed)
    print(f"Standard deviation across seeds: {avail_std:.4f}%")
    assert avail_std > 0.1, f"FAIL: soc_mean_avail_pct does not vary dynamically across seeds! std={avail_std}"
    print("PASS: soc_mean_avail_pct varies dynamically with random initial seed distribution.")

    print("\n--- 1.3 Verification of Dynamic Physical Depletion with Contingency Fraction (dPL) ---")
    for s in seeds:
        drops = [results[(s, dpl)]["soc_drop_mean"] for dpl in dpls]
        e_totals = [results[(s, dpl)]["e_dis_total"] for dpl in dpls]
        rt_means = [results[(s, dpl)]["rt_mean"] for dpl in dpls]
        rt_mins = [results[(s, dpl)]["rt_min"] for dpl in dpls]
        
        print(f"Seed {s:4d}:")
        print(f"  dPL = [0.03, 0.05, 0.08]")
        print(f"  SoC Mean Drop (%):    {[f'{d:.4f}' for d in drops]}")
        print(f"  Energy Discharged (kWh): {[f'{e:.2f}' for e in e_totals]}")
        print(f"  Final Mean Runtime (min): {[f'{r:.2f}' for r in rt_means]}")
        print(f"  Final Min Runtime (min):  {[f'{r:.2f}' for r in rt_mins]}")
        
        # Physical law: higher contingency must draw strictly more total discharged energy
        assert drops[0] < drops[1] < drops[2], (
            f"FAIL: Seed {s} SoC drop not strictly monotonic with dPL! {drops}"
        )
        assert e_totals[0] < e_totals[1] < e_totals[2], (
            f"FAIL: Seed {s} discharged energy not strictly increasing with dPL! {e_totals}"
        )
        # Physical law: higher contingency must result in lower final runtime
        assert rt_means[0] > rt_means[1] > rt_means[2], (
            f"FAIL: Seed {s} mean runtime not strictly decreasing with dPL! {rt_means}"
        )
        assert rt_mins[0] >= rt_mins[1] >= rt_mins[2], (
            f"FAIL: Seed {s} min runtime not non-increasing with dPL! {rt_mins}"
        )

    print("PASS: Battery SoC depletion, discharged energy, and reserve runtime vary strictly monotonically with contingency fraction dPL.")

    print("\n--- 1.4 Verification of Reserve Clamping & Tier Standards Compliance ---")
    for s in seeds:
        for dpl in dpls:
            res_item = results[(s, dpl)]
            soc_init = res_item["soc_init"]
            soc_final = res_item["soc_final"]
            
            # Check racks that started above reserve floor (80%)
            above_floor_mask = (soc_init >= 80.0)
            if np.any(above_floor_mask):
                min_soc_above = float(np.min(soc_final[above_floor_mask]))
                assert min_soc_above >= 80.0, (
                    f"FAIL: Reserve breached! Rack starting at {soc_init[above_floor_mask]} discharged to {min_soc_above:.4f}% < 80.0% for seed {s}, dpl {dpl}"
                )
            
            # Check racks that started below reserve floor (< 80%)
            below_floor_mask = (soc_init < 80.0)
            if np.any(below_floor_mask):
                max_change_below = float(np.max(np.abs(soc_final[below_floor_mask] - soc_init[below_floor_mask])))
                assert max_change_below < 1e-6, (
                    f"FAIL: Rack below reserve floor discharged! Max change = {max_change_below:.2e}% for seed {s}, dpl {dpl}"
                )
                
            assert res_item["rt_min"] >= 15.0, (
                f"FAIL: Tier IV compliance breached! Min runtime {res_item['rt_min']:.2f}m < 15.0 min"
            )
            assert res_item["t3"] == 100.0, f"FAIL: Tier III compliance not 100% ({res_item['t3']}%)"
            assert res_item["t4"] == 100.0, f"FAIL: Tier IV compliance not 100% ({res_item['t4']}%)"
    print("PASS: Reserve floor (80.0%) and Tier IV compliance (>= 15 min) strictly preserved under all conditions.")
    return True

# =====================================================================
# TEST 2: Pareto Sweep Monotonicity & Trade-Off Validity
# =====================================================================
def test_pareto_monotonicity():
    banner("TEST 2: Pareto Sweep Monotonicity & Trade-Off Validity (Nadir vs Messages)")
    eta_vals = [0.005, 0.010, 0.020, 0.040, 0.080, 0.150]
    
    print("\n--- 2.1 Fixed-Seed Pareto Sweep (seed = 42) ---")
    # Using fixed seed isolates the pure control threshold sensitivity from stochastic noise
    fetc_fixed_seed = []
    pulse_fixed_seed = []
    
    for ev in eta_vals:
        rf = run_sfr_simulation(
            horizon=10.0, controller_mode="fetc_fixed", filter_mode="pll_1cycle",
            kd=Kd_nom, eta=ev, eta_nom=ev, seed=42
        )
        rp = run_sfr_simulation(
            horizon=10.0, controller_mode="pulse_adaptive", filter_mode="pll_1cycle",
            kd=Kd_nom, eta=ev, eta_nom=ev, eta_rel=ev * 4.5, seed=42
        )
        fetc_fixed_seed.append((ev, rf["nadir_hz"], rf["msg_count_per_agent"]))
        pulse_fixed_seed.append((ev, rp["nadir_hz"], rp["msg_count_per_agent"]))
        del rf, rp
        import gc
        gc.collect()

    print(f"{'eta':>6s} | {'FETC Nadir':>10s} {'FETC Msgs':>10s} | {'PULSE Nadir':>11s} {'PULSE Msgs':>11s}")
    print("-" * 55)
    for i, ev in enumerate(eta_vals):
        fn, fm = fetc_fixed_seed[i][1], fetc_fixed_seed[i][2]
        pn, pm = pulse_fixed_seed[i][1], pulse_fixed_seed[i][2]
        print(f"{ev:6.3f} | {fn:10.4f} {fm:10.2f} | {pn:11.4f} {pm:11.2f}")

    print("\n--- 2.2 Varying-Seed Pareto Sweep (as executed in sim.py line 953) ---")
    fetc_var_seed = []
    pulse_var_seed = []
    for ev in eta_vals:
        ev_seed = int(ev * 1000) + 1
        rf = run_sfr_simulation(
            horizon=10.0, controller_mode="fetc_fixed", filter_mode="pll_1cycle",
            kd=Kd_nom, eta=ev, eta_nom=ev, seed=ev_seed
        )
        rp = run_sfr_simulation(
            horizon=10.0, controller_mode="pulse_adaptive", filter_mode="pll_1cycle",
            kd=Kd_nom, eta=ev, eta_nom=ev, eta_rel=ev * 4.5, seed=ev_seed
        )
        fetc_var_seed.append((ev, rf["nadir_hz"], rf["msg_count_per_agent"]))
        pulse_var_seed.append((ev, rp["nadir_hz"], rp["msg_count_per_agent"]))
        del rf, rp
        import gc
        gc.collect()

    print(f"{'eta':>6s} | {'FETC Nadir':>10s} {'FETC Msgs':>10s} | {'PULSE Nadir':>11s} {'PULSE Msgs':>11s}")
    print("-" * 55)
    for i, ev in enumerate(eta_vals):
        fn, fm = fetc_var_seed[i][1], fetc_var_seed[i][2]
        pn, pm = pulse_var_seed[i][1], pulse_var_seed[i][2]
        print(f"{ev:6.3f} | {fn:10.4f} {fm:10.2f} | {pn:11.4f} {pm:11.2f}")

    print("\n--- 2.3 Monotonicity Audit ---")
    # For both FETC and PULSE, as eta increases:
    # 1. Message counts must be strictly decreasing or non-increasing.
    # 2. Nadir must be non-increasing (as event triggers become sparser, nadir is slightly lower or equal).
    for name, data in [("FETC (Fixed Seed)", fetc_fixed_seed),
                       ("PULSE (Fixed Seed)", pulse_fixed_seed),
                       ("FETC (Varying Seed)", fetc_var_seed),
                       ("PULSE (Varying Seed)", pulse_var_seed)]:
        nadirs = [d[1] for d in data]
        msgs = [d[2] for d in data]
        
        # Check message count monotonicity
        msg_diffs = np.diff(msgs)
        is_msg_monotonic = np.all(msg_diffs <= 1e-4) # allow floating precision
        print(f"{name}:")
        print(f"  Messages: {[f'{m:.2f}' for m in msgs]} -> Monotonically decreasing? {is_msg_monotonic}")
        assert is_msg_monotonic, f"FAIL: {name} message counts are not monotonically decreasing! diffs={msg_diffs}"
        
        # Check nadir vs messages:
        # A valid Pareto relationship means higher message counts yield higher or equal nadir (i.e. more communication improves control)
        # So sorting by message count ascending, nadir should be non-decreasing
        nadir_diffs = np.diff(nadirs)
        print(f"  Nadirs:   {[f'{n:.4f}' for n in nadirs]}")
        # Note: with larger eta, nadir decreases (or stays flat within noise threshold)
        nadir_degradation = nadirs[0] - nadirs[-1]
        print(f"  Nadir span (tightest eta=0.005 vs loosest eta=0.150): {nadir_degradation:+.4f} Hz")
        assert nadir_degradation >= 0.0, f"FAIL: Loosest eta had better nadir than tightest eta! {nadirs}"
        
    print("\n--- 2.4 Pareto Dominance Audit: PULSE vs FETC ---")
    # For identical eta or comparable nadir, does PULSE communicate significantly fewer messages?
    for i, ev in enumerate(eta_vals):
        f_msg = fetc_fixed_seed[i][2]
        p_msg = pulse_fixed_seed[i][2]
        f_nad = fetc_fixed_seed[i][1]
        p_nad = pulse_fixed_seed[i][1]
        msg_reduction_pct = 100.0 * (1.0 - p_msg / f_msg) if f_msg > 0 else 0.0
        nadir_delta_mhz = (p_nad - f_nad) * 1000.0
        print(f"eta={ev:.3f}: PULSE msg reduction = {msg_reduction_pct:5.1f}% | Nadir delta = {nadir_delta_mhz:+5.2f} mHz")
        # PULSE must reduce messages compared to FETC while preserving near-identical nadir
        assert p_msg <= f_msg, f"FAIL: PULSE used more messages than FETC at eta={ev}! ({p_msg} > {f_msg})"
        assert abs(nadir_delta_mhz) < 40.0, f"FAIL: PULSE nadir diverged from FETC by > 40 mHz ({nadir_delta_mhz} mHz)"
        assert p_nad >= 59.50, f"FAIL: PULSE nadir breached 59.50 Hz limit! ({p_nad} Hz)"

    print("PASS: Pareto sweep demonstrates authentic strictly monotonic message reduction and valid Pareto optimality.")
    return True

# =====================================================================
# TEST 3: Trajectory Matrix & Individual Rack Injection Integrity
# =====================================================================
def test_trajectory_matrix_integrity():
    banner("TEST 3: Trajectory Matrix & Individual Rack Injection Integrity (p_individual)")
    
    controllers = ["pulse_adaptive", "fetc_fixed", "industry_droop"]
    
    for ctrl in controllers:
        print(f"\n--- 3.1 Controller: {ctrl} (10.0s horizon) ---")
        t0 = time.time()
        res = run_sfr_simulation(
            horizon=10.0,
            controller_mode=ctrl,
            filter_mode="pll_1cycle",
            battery_model="dynamic_kwh",
            seed=42,
            downsample_factor=20
        )
        dt_sim = time.time() - t0
        
        p_ind = res["p_individual"]
        p_fleet = res["p_mw"]
        t_s = res["t_s"]
        
        N_rec = len(t_s)
        n_ag = 200
        
        print(f"Simulation completed in {dt_sim:.2f}s. N_rec = {N_rec}, n_agents = {n_ag}")
        print(f"p_individual ndim: {p_ind.ndim}, shape: {p_ind.shape}, dtype: {p_ind.dtype}")
        
        # 1. Dimension and shape check
        assert p_ind.ndim == 2, f"FAIL: p_individual ndim must be 2, got {p_ind.ndim}"
        assert p_ind.shape == (N_rec, n_ag), f"FAIL: p_individual shape must be ({N_rec}, {n_ag}), got {p_ind.shape}"
        
        # 2. Non-triviality check
        max_val = np.max(p_ind)
        min_val = np.min(p_ind)
        mean_val = np.mean(p_ind)
        active_elements = np.count_nonzero(p_ind > 1e-6)
        total_elements = p_ind.size
        active_pct = 100.0 * active_elements / total_elements
        
        print(f"Min power: {min_val:.6f} MW, Max power: {max_val:.6f} MW, Mean: {mean_val:.6f} MW")
        print(f"Active elements (> 1e-6 MW): {active_elements}/{total_elements} ({active_pct:.2f}%)")
        
        assert max_val > 0.010, f"FAIL: p_individual has max power {max_val} MW (trivial/zeros)"
        assert min_val >= -1e-9, f"FAIL: p_individual contains negative power injection! min = {min_val}"
        assert max_val <= Pmax_agent + 1e-6, f"FAIL: p_individual exceeds rack Pmax {Pmax_agent} MW! max = {max_val}"
        assert active_elements > 0.1 * total_elements, f"FAIL: p_individual has insufficient active entries ({active_pct:.1f}%)"
        
        # 3. Aggregation and conservation check: sum(p_individual, axis=1) == p_fleet
        p_fleet_reconstructed = np.sum(p_ind, axis=1)
        abs_err = np.abs(p_fleet_reconstructed - p_fleet)
        max_abs_err = np.max(abs_err)
        mean_abs_err = np.mean(abs_err)
        
        # Relative error on active steps (p_fleet > 0.1 MW)
        active_idx = np.where(p_fleet > 0.1)[0]
        if len(active_idx) > 0:
            rel_err_active = abs_err[active_idx] / p_fleet[active_idx]
            max_rel_err = np.max(rel_err_active)
            mean_rel_err = np.mean(rel_err_active)
        else:
            max_rel_err = 0.0
            mean_rel_err = 0.0
            
        print(f"Fleet Power Conservation Check: sum(p_individual) vs p_fleet:")
        print(f"  Max absolute error:  {max_abs_err:.2e} MW")
        print(f"  Mean absolute error: {mean_abs_err:.2e} MW")
        print(f"  Max relative error:  {max_rel_err:.2e} ({max_rel_err*100:.6f}%)")
        print(f"  Mean relative error: {mean_rel_err:.2e} ({mean_rel_err*100:.6f}%)")
        
        assert max_abs_err < 1e-9, f"FAIL: sum(p_individual) does NOT match p_fleet! max error = {max_abs_err}"
        assert max_rel_err < 1e-9, f"FAIL: relative discrepancy in p_individual sum! max rel err = {max_rel_err}"
        
        # 4. Individual Agent Heterogeneity / Diversity Check
        # In event-triggered mode (PULSE/FETC), different agents trigger at slightly different times due to latency
        std_across_agents = np.std(p_ind, axis=1)
        max_agent_std = np.max(std_across_agents)
        print(f"  Max standard deviation across agents at any step: {max_agent_std:.6f} MW")
        if ctrl != "industry_droop":
            assert max_agent_std > 1e-4, f"FAIL: All agents actuated identically in event-triggered control! std={max_agent_std}"
            print("  PASS: Individual agents actuate asynchronously with authentic rack-level diversity.")
            
        del res, p_ind, p_fleet, p_fleet_reconstructed
        import gc
        gc.collect()

    print("\nPASS: p_individual is an authentic 2D trajectory matrix matching fleet power with machine-precision conservation (< 1e-9 MW).")
    return True

# =====================================================================
# TEST 4: Forensic Re-Verification of Auditor Findings
# =====================================================================
def test_forensic_remediation():
    banner("TEST 4: Forensic Re-Verification of Auditor Findings")
    
    # 4.1 No hardcoded strings
    sim_content = open(os.path.join(REPO_DIR, "sim.py")).read()
    assert "39.39" not in sim_content, "FAIL: Static literal '39.39' found in sim.py"
    assert "39.50" not in sim_content, "FAIL: Static literal '39.50' found in sim.py"
    print("PASS: Zero hardcoded metric literals ('39.39', '39.50') in sim.py.")
    
    # 4.2 Network impairment sensitivity (p_drop and jitter)
    r_base = run_sfr_simulation(horizon=5.0, contingency_frac=0.05, p_drop=0.0, jitter_scale_ms=0.0, seed=42)
    r_drop = run_sfr_simulation(horizon=5.0, contingency_frac=0.05, p_drop=0.30, jitter_scale_ms=0.0, seed=42)
    r_jit = run_sfr_simulation(horizon=5.0, contingency_frac=0.05, p_drop=0.0, jitter_scale_ms=50.0, seed=42)
    
    delta_drop_msgs = r_drop["msg_count_total"] - r_base["msg_count_total"]
    delta_jit_msgs = r_jit["msg_count_total"] - r_base["msg_count_total"]
    
    print(f"Network Impairment Check:")
    print(f"  Base (p_drop=0, jitter=0):    Msgs = {r_base['msg_count_total']:.0f}, Nadir = {r_base['nadir_hz']:.4f} Hz")
    print(f"  p_drop=0.30:                  Msgs = {r_drop['msg_count_total']:.0f} (delta = {delta_drop_msgs:+.0f}), Nadir = {r_drop['nadir_hz']:.4f} Hz")
    print(f"  jitter=50ms:                  Msgs = {r_jit['msg_count_total']:.0f} (delta = {delta_jit_msgs:+.0f}), Nadir = {r_jit['nadir_hz']:.4f} Hz")
    
    assert abs(delta_drop_msgs) > 0, "FAIL: p_drop produced zero difference in message count (facade implementation)!"
    assert abs(delta_jit_msgs) > 0, "FAIL: jitter produced zero difference in message count (facade implementation)!"
    print("PASS: Network impairments produce authentic dynamic variations.")
    
    # 4.3 16384 Buffer Aliasing Check
    from test_adversarial_m1_2 import run_circular_buffer_aliasing_audit
    run_circular_buffer_aliasing_audit()
    print("PASS: Circular buffer aliasing completely resolved (16384 elements).")
    return True

if __name__ == "__main__":
    t_start = time.time()
    banner("STARTING CHALLENGER 2 EMPIRICAL HARNESS")
    
    t1_pass = test_battery_physics_adversarial()
    t2_pass = test_pareto_monotonicity()
    t3_pass = test_trajectory_matrix_integrity()
    t4_pass = test_forensic_remediation()
    
    total_time = time.time() - t_start
    banner(f"ALL CHALLENGER 2 VERIFICATION SUITES PASSED ({total_time:.2f} s)")
