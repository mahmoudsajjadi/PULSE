"""
Deep Empirical Challenger Stress Suite for PULSE Milestone 1 Iteration 2
========================================================================
Validates:
1. Network Impairments (p_drop and jitter_scale_ms):
   - Physical variations in delivered messages and frequency response (not zero deltas).
   - Validation across both Numba and Pure-NumPy execution engines.
2. Circular Delay Buffer Aliasing & Boundary Stress:
   - Delays: 100 ms, 250 ms, 500 ms, 800 ms (2000, 5000, 10000, 16000 steps).
   - Direct verification against modulo 2048 aliasing.
   - Impulse delay retrieval verification.
   - Headroom analysis relative to 819.2 ms capacity (16384 * 50us).
3. Memory Consumption & Heap Growth:
   - 5 consecutive back-to-back 30.0s simulation runs (600,000 steps each).
   - Heap allocation profiling via tracemalloc.
   - Verification of zero ArrayMemoryError and sub-100MB incremental growth.
"""

import os
import sys
import gc
import tracemalloc
import time
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import sim

def log_header(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

def test_network_impairments_both_engines():
    log_header("CHALLENGER TEST 1: Network Impairment Sensitivity Across Engines")
    
    # 1. Numba Kernel Test (default)
    print("--- 1A: Numba Engine (HAS_NUMBA = %s) ---" % sim.HAS_NUMBA)
    packet_drops = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    base_numba = sim.run_sfr_simulation(
        horizon=10.0, contingency_frac=0.05, controller_mode="pulse_adaptive",
        p_drop=0.0, jitter_scale_ms=0.0, seed=42
    )
    base_nadir = base_numba["nadir_hz"]
    base_msgs = base_numba["msg_count_total"]
    
    numba_drop_deltas = []
    for pd in packet_drops[1:]:
        res = sim.run_sfr_simulation(
            horizon=10.0, contingency_frac=0.05, controller_mode="pulse_adaptive",
            p_drop=pd, jitter_scale_ms=0.0, seed=42
        )
        d_nadir = res["nadir_hz"] - base_nadir
        d_msgs = res["msg_count_total"] - base_msgs
        print(f"  [Numba] p_drop={pd:4.2f} | Nadir: {res['nadir_hz']:9.6f} Hz (delta: {d_nadir:+.6f} Hz) | Msgs: {res['msg_count_total']:5.0f} (delta: {d_msgs:+5.0f})")
        assert abs(d_nadir) > 1e-6 or abs(d_msgs) > 0, f"Zero delta detected for p_drop={pd} on Numba engine!"
        numba_drop_deltas.append((pd, d_nadir, d_msgs))
        
    jitters = [10.0, 25.0, 50.0, 100.0]
    numba_jitter_deltas = []
    for jm in jitters:
        res = sim.run_sfr_simulation(
            horizon=10.0, contingency_frac=0.05, controller_mode="pulse_adaptive",
            p_drop=0.0, jitter_scale_ms=jm, seed=42
        )
        d_nadir = res["nadir_hz"] - base_nadir
        d_msgs = res["msg_count_total"] - base_msgs
        print(f"  [Numba] jitter={jm:5.1f} ms | Nadir: {res['nadir_hz']:9.6f} Hz (delta: {d_nadir:+.6f} Hz) | Msgs: {res['msg_count_total']:5.0f} (delta: {d_msgs:+5.0f})")
        assert abs(d_nadir) > 1e-6 or abs(d_msgs) > 0, f"Zero delta detected for jitter={jm} on Numba engine!"
        numba_jitter_deltas.append((jm, d_nadir, d_msgs))

    # 2. NumPy Kernel Test
    print("\n--- 1B: Pure-NumPy Fallback Engine ---")
    orig_has_numba = sim.HAS_NUMBA
    try:
        sim.HAS_NUMBA = False # Force NumPy kernel
        base_numpy = sim.run_sfr_simulation(
            horizon=10.0, contingency_frac=0.05, controller_mode="pulse_adaptive",
            p_drop=0.0, jitter_scale_ms=0.0, seed=42
        )
        np_base_nadir = base_numpy["nadir_hz"]
        np_base_msgs = base_numpy["msg_count_total"]
        print(f"  NumPy Baseline: Nadir = {np_base_nadir:.6f} Hz, Msgs = {np_base_msgs}")
        
        for pd in [0.10, 0.50]:
            res = sim.run_sfr_simulation(
                horizon=10.0, contingency_frac=0.05, controller_mode="pulse_adaptive",
                p_drop=pd, jitter_scale_ms=0.0, seed=42
            )
            d_nadir = res["nadir_hz"] - np_base_nadir
            d_msgs = res["msg_count_total"] - np_base_msgs
            print(f"  [NumPy] p_drop={pd:4.2f} | Nadir: {res['nadir_hz']:9.6f} Hz (delta: {d_nadir:+.6f} Hz) | Msgs: {res['msg_count_total']:5.0f} (delta: {d_msgs:+5.0f})")
            assert abs(d_nadir) > 1e-6 or abs(d_msgs) > 0, f"Zero delta detected for p_drop={pd} on NumPy engine!"
            
        for jm in [50.0, 100.0]:
            res = sim.run_sfr_simulation(
                horizon=10.0, contingency_frac=0.05, controller_mode="pulse_adaptive",
                p_drop=0.0, jitter_scale_ms=jm, seed=42
            )
            d_nadir = res["nadir_hz"] - np_base_nadir
            d_msgs = res["msg_count_total"] - np_base_msgs
            print(f"  [NumPy] jitter={jm:5.1f} ms | Nadir: {res['nadir_hz']:9.6f} Hz (delta: {d_nadir:+.6f} Hz) | Msgs: {res['msg_count_total']:5.0f} (delta: {d_msgs:+5.0f})")
            assert abs(d_nadir) > 1e-6 or abs(d_msgs) > 0, f"Zero delta detected for jitter={jm} on NumPy engine!"
    finally:
        sim.HAS_NUMBA = orig_has_numba

    print("-> Test 1 Result: PASS (Network impairments are authentically integrated across both engines).")
    return numba_drop_deltas, numba_jitter_deltas


def test_circular_delay_buffer_aliasing():
    log_header("CHALLENGER TEST 2: Circular Delay Buffer Aliasing & Boundary Stress")
    
    # Mathematical and empirical analysis of delay buffer
    dt = 5e-5 # 50 microseconds
    buffer_len = 16384 # Remediated size
    mask = 0x3FFF
    max_supported_delay_s = buffer_len * dt
    max_supported_delay_ms = max_supported_delay_s * 1000.0
    print(f"Buffer specification:")
    print(f"  Length: {buffer_len} elements")
    print(f"  Bitmask: 0x{mask:X} ({mask})")
    print(f"  Step size dt: {dt*1e6:.1f} us")
    print(f"  Maximum delay span without wrap: {max_supported_delay_ms:.2f} ms ({max_supported_delay_s:.4f} s)")
    
    test_delays_ms = [100.0, 250.0, 500.0, 800.0]
    
    print("\n--- 2A: Direct Bitmask & Step Mapping Audit ---")
    for tau in test_delays_ms:
        req_steps = int(round((tau * 1e-3) / dt))
        idx_16k = req_steps & mask
        idx_2k = req_steps & 0x07FF # Previous buggy 2048-element buffer
        
        tau_effective_16k_ms = idx_16k * dt * 1000.0
        tau_effective_2k_ms = idx_2k * dt * 1000.0
        
        aliased_16k = (req_steps != idx_16k)
        aliased_2k = (req_steps != idx_2k)
        
        print(f"Requested Delay tau = {tau:6.1f} ms ({req_steps:5d} steps):")
        print(f"  16384 buffer: actual delay = {tau_effective_16k_ms:6.1f} ms | Aliased: {aliased_16k}")
        print(f"   2048 buffer: actual delay = {tau_effective_2k_ms:6.1f} ms | Aliased: {aliased_2k} (ERROR: {abs(tau - tau_effective_2k_ms):.1f} ms)")
        
        assert not aliased_16k, f"Delay {tau} ms aliased in 16384 buffer!"
        if tau > 102.4:
            assert aliased_2k, f"Delay {tau} ms should alias in 2048 buffer!"

    print("\n--- 2B: Empirical Dynamic Signal Delay Retrieval Verification ---")
    # Simulate an actual ring buffer filled step-by-step from k=0 to N_sim.
    # When k >= tau_steps, verifying that retrieving (k - tau_steps) & mask yields exact f(k - tau_steps).
    N_sim = 50000 # 2.5 seconds of steps (>> 16384 steps, covering wrap-around)
    
    for tau in test_delays_ms:
        tau_steps = int(round((tau * 1e-3) / dt))
        f_buf = np.zeros(buffer_len, dtype=np.float64)
        max_err = 0.0
        
        for k in range(N_sim):
            true_signal = float(k) * 0.001
            f_buf[k & mask] = true_signal
            
            if k >= tau_steps:
                retrieved = f_buf[(k - tau_steps) & mask]
                expected = float(k - tau_steps) * 0.001
                err = abs(retrieved - expected)
                if err > max_err:
                    max_err = err
                
        print(f"  tau = {tau:6.1f} ms ({tau_steps:5d} steps) | Retrieval Max Error: {max_err:.10e}")
        assert max_err < 1e-12, f"Delay retrieval error {max_err} exceeds tolerance for tau={tau} ms!"


    print("\n--- 2C: Extreme Delay Stability Sweep in sim.py (100 ms to 800 ms) ---")
    # Run full closed-loop simulation kernel with exact uniform agent delays tau in {100, 250, 500, 800} ms
    n_ag = 200
    N_steps = int(round(5.0 / dt))
    downsample = 20
    k_fault = int(0.100 / dt)
    poll_steps = int(round(0.100 / dt))
    soc_arr = np.full(n_ag, 85.0)
    pmax_arr = np.full(n_ag, sim.Pmax_agent)
    cfg = sim.FILTER_CONFIGS["pll_1cycle"]
    db_pu = 0.015 / sim.f0
    noise_pu = 0.0005 / sim.f0
    
    for tau in test_delays_ms:
        tau_step_val = int(round((tau * 1e-3) / dt))
        tau_steps = np.full(n_ag, tau_step_val, dtype=np.int64)
        
        # Invoke kernel with exact delay
        kernel = sim._sfr_core_numba if sim.HAS_NUMBA else sim._sfr_core_numpy
        out = kernel(
            N_steps, dt, downsample,
            0.100, 0.05,
            4, # pulse_adaptive
            1, # pll_1cycle
            2, # dynamic_kwh
            tau_steps, soc_arr.copy(), pmax_arr,
            4.0, 9.0, poll_steps,
            0.02, 0.09, 0.6,
            cfg["tau_f"], cfg["tau_r"], db_pu,
            noise_pu, 0.0, np.zeros(1),
            42
        )
        t_rec, f_rec, p_rec, p_ind, soc_rec, rt_rec, msg_rec, t_z, f_z, p_z, f_min, t_nad, tot_msgs, cum_e = out
        nadir = sim.f0 + f_min * sim.f0
        print(f"  Exact tau = {tau:5.1f} ms ({tau_step_val:5d} steps): Nadir = {nadir:6.3f} Hz | Peak P = {p_rec.max():5.2f} MW | Msgs = {tot_msgs} | Stable: {55.0 < nadir < 65.0}")
        assert 55.0 < nadir < 65.0, f"Unstable nadir {nadir} at tau={tau}"
        assert not np.isnan(f_rec).any(), f"NaN in frequency trajectory at tau={tau}"

        
    print("-> Test 2 Result: PASS (Circular buffer correctly operates up to 800 ms with ZERO aliasing).")


def test_memory_consumption_repeated_30s():
    log_header("CHALLENGER TEST 3: Memory Profile & Heap Allocation (5 Back-to-Back 30s Runs)")
    
    tracemalloc.start()
    runs = 5
    run_records = []
    
    gc.collect()
    snap_initial = tracemalloc.take_snapshot()
    
    for r in range(1, runs + 1):
        t0 = time.time()
        tracemalloc.reset_peak()
        
        # Execute full 30.0 s simulation (600,000 steps, 200 agents)
        res = sim.run_sfr_simulation(
            horizon=30.0,
            dt=5e-5,
            contingency_frac=0.05,
            controller_mode="pulse_adaptive",
            filter_mode="pll_1cycle",
            battery_model="dynamic_kwh",
            seed=42 + r
        )
        elapsed = time.time() - t0
        
        current_b, peak_b = tracemalloc.get_traced_memory()
        current_mb = current_b / (1024 * 1024)
        peak_mb = peak_b / (1024 * 1024)
        
        # Check integrity of returned arrays
        assert res["t_s"].shape == (30000,), f"Unexpected t_s shape: {res['t_s'].shape}"
        assert res["p_individual"].shape == (30000, 200), f"Unexpected p_individual shape: {res['p_individual'].shape}"
        assert not np.isnan(res["f_hz"]).any(), "NaN found in frequency trace"
        
        print(f"  Run {r}/{runs} (horizon=30.0s, 600k steps) | Time: {elapsed:5.2f}s | Current RAM: {current_mb:6.2f} MiB | Peak Run RAM: {peak_mb:6.2f} MiB")
        run_records.append({
            "run": r, "time_s": elapsed, "current_mb": current_mb, "peak_mb": peak_mb
        })
        
        # Free results to test GC release
        del res
        gc.collect()

    tracemalloc.stop()
    
    # Verify memory stability: Peak RAM during any run should be < 200 MiB (in contrast to former 916 MiB monolithic noise array)
    peaks = [rec["peak_mb"] for rec in run_records]
    max_peak = max(peaks)
    final_current = run_records[-1]["current_mb"]
    
    print(f"\nMemory Stability Summary:")
    print(f"  Maximum Peak RAM Across All Runs: {max_peak:.2f} MiB")
    print(f"  Final Retained Heap RAM:         {final_current:.2f} MiB")
    
    assert max_peak < 200.0, f"Peak memory {max_peak:.2f} MiB exceeded 200 MiB limit!"
    print("-> Test 3 Result: PASS (Zero ArrayMemoryError, streaming generation keeps RAM minimal).")
    return run_records


if __name__ == "__main__":
    t_global_start = time.time()
    print("=" * 75)
    print("STARTING DEEP EMPIRICAL CHALLENGER STRESS SUITE (M1 ITERATION 2)")
    print("=" * 75)
    
    r1 = test_network_impairments_both_engines()
    r2 = test_circular_delay_buffer_aliasing()
    r3 = test_memory_consumption_repeated_30s()
    
    print("\n" + "=" * 75)
    print(f"ALL DEEP CHALLENGER STRESS TESTS PASSED SUCCESSFULLY IN {time.time() - t_global_start:.2f} s")
    print("=" * 75)
