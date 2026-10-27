import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import sim

def analyze_unconstrained_swing():
    print("=" * 70)
    print("DEEP DIVE: Swing Equation & Virtual Inertia H_v with Unconstrained Battery")
    print("=" * 70)
    
    # 1. Unconstrained battery (all 200 agents active)
    # Controller = industry_droop, idealized filter (tau_f=0, tau_r=0, noise=0, deadband=0)
    res_unsupported = sim.run_sfr_simulation(
        horizon=2.0, dt=5e-5, contingency_frac=0.05,
        controller_mode="none", filter_mode="idealized",
        battery_model="unconstrained", seed=42
    )
    
    res_linear_unconstrained = sim.run_sfr_simulation(
        horizon=2.0, dt=5e-5, contingency_frac=0.05,
        controller_mode="industry_droop", filter_mode="idealized",
        kd=4.0, kp=0.0, deadband_hz=0.0, # pure synthetic inertia (Kd=4, Kp=0)
        battery_model="unconstrained", seed=42
    )

    t_zoom_un = res_unsupported["t_zoom"]
    f_zoom_un = res_unsupported["f_zoom"]
    t_zoom_lin = res_linear_unconstrained["t_zoom"]
    f_zoom_lin = res_linear_unconstrained["f_zoom"]
    p_zoom_lin = res_linear_unconstrained["p_zoom"]
    
    # Theoretical predictions:
    Sbase = sim.Sbase
    f0 = sim.f0
    H = sim.H
    dPL = 0.05 * Sbase
    H_v = (sim.n_agents * 4.0) / (2.0 * Sbase) # 1.3333 s
    H_eff = H + H_v # 4.5333 s
    
    rocof_un_theory = - (dPL / (2.0 * H * Sbase)) * f0 # -0.46875 Hz/s
    rocof_lin_theory = - (dPL / (2.0 * H_eff * Sbase)) * f0 # -0.33088 Hz/s
    theory_reduction = 100.0 * (1.0 - abs(rocof_lin_theory) / abs(rocof_un_theory))
    
    print(f"Theory: H_base={H:.3f}s, H_v={H_v:.4f}s, H_eff={H_eff:.4f}s")
    print(f"Theoretical RoCoF Unsupported: {rocof_un_theory:.5f} Hz/s")
    print(f"Theoretical RoCoF Linear:      {rocof_lin_theory:.5f} Hz/s (pure synthetic inertia Kd=4, Kp=0)")
    print(f"Theoretical Reduction:         {theory_reduction:.2f}%")
    
    # Check zoom slope once actuation is active
    # Transport delay tau_s is uniform[1ms, 8ms]
    # By t = t_fault + 8ms = 0.108 s, all agents have received the signal
    k1 = np.where(t_zoom_lin >= 0.108)[0][0]
    k2 = np.where(t_zoom_lin >= 0.128)[0][0] # 20 ms window where all agents are fully active
    dt_win = t_zoom_lin[k2] - t_zoom_lin[k1]
    
    df_un = (f_zoom_un[k2] - f_zoom_un[k1]) / dt_win
    df_lin = (f_zoom_lin[k2] - f_zoom_lin[k1]) / dt_win
    p_mean = np.mean(p_zoom_lin[k1:k2])
    reduction_emp = 100.0 * (1.0 - abs(df_lin) / abs(df_un))
    
    print(f"\nZoom Window [0.108s to 0.128s] (All agents active, mean P = {p_mean:.3f} MW):")
    print(f"  Empirical Unsupported RoCoF: {df_un:.5f} Hz/s (Theory: {rocof_un_theory:.5f})")
    print(f"  Empirical Linear RoCoF:      {df_lin:.5f} Hz/s (Theory: {rocof_lin_theory:.5f})")
    print(f"  Empirical Reduction:         {reduction_emp:.2f}% (Theory: {theory_reduction:.2f}%)")
    print(f"  Discrepancy vs Theory:       {abs(reduction_emp - theory_reduction):.2f} percentage points")

    # Now let's test why there's any residual difference:
    # In sim.py line 386:
    # df_eff = (f_delayed - f_buf[idxm]) / dt_step
    # Note that df_eff is the numerical backward difference over dt_step (50 us)
    # And p_star = -kd * df_eff - kp * f_eff
    # If df_eff ~ df/dt, then P_bess = - (N * Kd / f0) * (df / dt)?
    # WAIT! Check units of Kd in sim.py line 389:
    # p_star = -kd * df_eff - kp * f_eff
    # df_eff is in pu/s! (f is in pu, so df_eff is in pu/s)
    # If df_eff is in pu/s, then kd has units of MW / (pu/s) = MW*s/pu.
    # So P_star is in MW.
    # Total P_bess = sum(p_hold) = N * p_star (if all identical).
    # Then P_bess_pu = P_bess / Sbase = (N * kd / Sbase) * (-df_pu/dt).
    # In swing equation:
    # 2*H * df_pu/dt = - PL_pu + P_bess_pu
    # = - PL_pu - (N * kd / Sbase) * df_pu/dt
    # (2*H + N * kd / Sbase) * df_pu/dt = - PL_pu
    # 2 * (H + N * kd / (2 * Sbase)) * df_pu/dt = - PL_pu
    # So H_v = (N * kd) / (2 * Sbase).
    # Exactly matching!
    
if __name__ == "__main__":
    analyze_unconstrained_swing()
