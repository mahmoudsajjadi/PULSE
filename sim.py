import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# =============================================================
# 1. Reduced-order system-frequency-response (SFR) model
#    2H df/dt = dPm - dPL - D*f + Pbess
#    Tg dPm/dt = -Pm - f/R
# =============================================================
Sbase   = 300.0          # MVA, local substation aggregation capacity
H       = 3.2            # s, equivalent inertia constant (weak-inertia area)
D       = 1.0            # pu load damping
R       = 0.05           # pu droop
Tg      = 6.0             # s governor lag
dPL     = 0.05 * Sbase    # MW step loss (15 MW generator trip)
t_fault = 0.100           # s, contingency onset

dt   = 5e-5                # 50 microsecond step
Tend = 6.0
N    = int(Tend/dt)
time = np.arange(N) * dt

# =============================================================
# 2. Distributed rack-level battery agents (event-triggered control)
# =============================================================
n_agents   = 200
Pmax_agent = 0.06         # MW per rack-level bank -> 12 MW aggregate (0.04 pu)
tau        = rng.uniform(1e-3, 8e-3, n_agents)      # local sensing/comm latency [s]
tau_steps  = np.round(tau/dt).astype(int)
eta        = 0.02                                    # relative event-trigger threshold
dwell      = 1.5e-3                                   # min inter-event time (anti-Zeno) [s]
Kd         = 55.0         # MW s / (pu/s) -- synthetic-inertia gain
Kp         = 9.0          # MW / pu       -- droop / primary-response gain


def run(active, event_triggered=True, poll_period=0.100, return_cum_msgs=False):
    f = 0.0
    Pm = 0.0
    f_trace = np.zeros(N)
    pbess_trace = np.zeros(N)
    cum_msgs_trace = np.zeros(N)
    p_hold = np.zeros(n_agents)
    last_evt = np.full(n_agents, -1.0)
    if not event_triggered:
        last_evt = np.full(n_agents, -poll_period)
    f_prev = 0.0
    total_msgs = 0.0

    for k in range(N):
        t = time[k]
        PL = dPL if t >= t_fault else 0.0

        Pbess_MW = 0.0
        if active:
            # each agent only "sees" the state as it was tau_i seconds ago
            idx  = np.clip(k - tau_steps,     0, k)
            idxm = np.clip(k - tau_steps - 1, 0, k)
            valid = time[idx] >= t_fault
            f_meas   = np.where(valid, f_trace[idx], 0.0)
            df_meas  = np.where(valid, (f_trace[idx] - f_trace[idxm]) / dt, 0.0)
            p_star = np.clip(-Kd * df_meas - Kp * f_meas, -Pmax_agent, Pmax_agent)

            if event_triggered:
                trig = (np.abs(p_star - p_hold) > eta * Pmax_agent) & \
                       (t - last_evt > dwell)
            else:
                trig = (t - last_evt) >= poll_period

            total_msgs += trig.sum() / n_agents
            p_hold = np.where(trig, p_star, p_hold)
            last_evt = np.where(trig, t, last_evt)
            Pbess_MW = p_hold.sum()

        cum_msgs_trace[k] = total_msgs
        PL_pu = PL / Sbase
        Pbess_pu = Pbess_MW / Sbase
        dPm = (-Pm - f / R) / Tg
        df = (Pm - PL_pu - D * f + Pbess_pu) / (2 * H)
        Pm += dPm * dt
        f_prev = f
        f += df * dt

        f_trace[k] = f
        pbess_trace[k] = Pbess_MW

    if return_cum_msgs:
        return f_trace, pbess_trace, cum_msgs_trace
    return f_trace, pbess_trace


f_base, _                = run(active=False)
f_evt,  p_evt,  cm_evt   = run(active=True,  event_triggered=True,  return_cum_msgs=True)
f_poll, p_poll, cm_poll  = run(active=True,  event_triggered=False, poll_period=0.100, return_cum_msgs=True)

f0 = 60.0  # Hz nominal (US-style HV distribution substation)

# ---- key metrics -------------------------------------------------
nadir_base = f0 + f_base.min() * f0
nadir_evt  = f0 + f_evt.min()  * f0
nadir_poll = f0 + f_poll.min() * f0

Pss = p_evt[int((t_fault + 0.3) / dt)]  # settled aggregate support level
target90 = 0.9 * Pss
idx_fault = int(t_fault / dt)
post = p_evt[idx_fault:]
idx90 = np.argmax(post >= target90)
t90_ms = idx90 * dt * 1000.0
if t90_ms == 0 and post[0] < target90:
    t90_ms = np.nan

print(f"Frequency nadir  -- no support: {nadir_base:.3f} Hz | periodic (100 ms poll): "
      f"{nadir_poll:.3f} Hz | event-triggered: {nadir_evt:.3f} Hz")
print(f"Steady aggregate BESS support: {Pss:.2f} MW ({Pss/Sbase*100:.2f}% of Sbase)")
print(f"Time to reach 90% of steady support (event-triggered): {t90_ms:.2f} ms "
      f"(one 60-Hz cycle = 16.67 ms)")

# =============================================================
# 3. Battery degradation under added frequency-regulation duty
#    Semi-empirical Ah-throughput capacity-fade model, fade-rate
#    calibrated to ~20% capacity loss at 3000 equivalent full
#    cycles (EFC) for grid-service-duty NMC cells (Naumann et
#    al., J. Energy Storage 2018).
# =============================================================
fade_pct_per_EFC = 20.0 / 3000.0            # %/EFC
rack_capacity_kWh = 250.0                    # kWh per rack-level bank
n_events_per_year = 180                      # contingency-scale events/yr
event_duration_s  = 8.0                      # effective support duration per event
mean_power_MW_per_rack = np.mean(np.abs(p_evt[p_evt != 0])) / n_agents if np.any(p_evt != 0) else Pmax_agent/2

Ah_equiv_kWh = mean_power_MW_per_rack * 1000 * (event_duration_s/3600)  # kWh per rack per event
EFC_per_event = Ah_equiv_kWh / (2 * rack_capacity_kWh)                  # full charge+discharge=1 EFC
EFC_per_year = EFC_per_event * n_events_per_year

n_years = 10
cycles_baseline = np.linspace(0, 300, 400)      # baseline UPS duty (idle/backup testing), EFC
extra_EFC = np.linspace(0, EFC_per_year * n_years, 400)
cap_baseline = 100 - cycles_baseline * fade_pct_per_EFC
cap_with_FR  = 100 - (cycles_baseline + extra_EFC) * fade_pct_per_EFC
delta_fade_final = cap_baseline[-1] - cap_with_FR[-1]

print(f"Added EFC over {n_years} yr of continuous FR duty: {EFC_per_year*n_years:.3f} EFC")
print(f"Resulting extra capacity fade after {n_years} yr: {delta_fade_final:.4f} percentage points")

# =============================================================
# 4. Figure: combined response + cumulative messages panel
# =============================================================
plt.rcParams.update({
    "font.size": 8.5,
    "font.family": "serif",
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
})

fig = plt.figure(figsize=(7.2, 2.75))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.85, 1.05], wspace=0.55)

# --- Panel A: frequency response ---
ax1 = fig.add_subplot(gs[0])
t_ms = (time - t_fault) * 1000
mask = time >= (t_fault - 0.02)
ax1.plot(t_ms[mask], f_base[mask]*f0 + f0, color="0.55", lw=1.1, label="No BESS support")
ax1.plot(t_ms[mask], f_poll[mask]*f0 + f0, color="#4C72B0", lw=1.1, ls="--",
         label="Periodic (100 ms poll)")
ax1.plot(t_ms[mask], f_evt[mask]*f0 + f0, color="#C44E52", lw=1.3,
         label="Event-triggered (proposed)")
ax1.axvline(0, color="k", lw=0.6, ls=":")
ax1.set_xlabel("Time since contingency (ms)")
ax1.set_ylabel("Frequency (Hz)")
ax1.set_xlim(-100, 4000)
ax1.legend(fontsize=6.2, frameon=False, loc="lower right")
ax1.set_title("(a) Frequency nadir response", fontsize=8.5)

# --- Panel B: sub-cycle zoom of aggregate power injection ---
ax2 = fig.add_subplot(gs[1])
mask2 = (t_ms >= -2) & (t_ms <= 40)
ax2.plot(t_ms[mask2], p_evt[mask2], color="#C44E52", lw=1.3, label="Event-triggered")
ax2.plot(t_ms[mask2], p_poll[mask2], color="#4C72B0", lw=1.1, ls="--", label="Periodic")
ax2.axvline(16.67, color="k", lw=0.6, ls=":")
ax2.text(17.3, 0.35*Pss, "1 cycle\n(60 Hz)", fontsize=6, va="bottom")
ax2.axhline(target90, color="0.4", lw=0.5, ls=":")
ax2.annotate(f"t90 = {t90_ms:.1f} ms", xy=(t90_ms, target90),
             xytext=(19, target90*0.5), fontsize=6.3,
             arrowprops=dict(arrowstyle="->", lw=0.6))
ax2.set_xlabel("Time since contingency (ms)")
ax2.set_ylabel("Aggregate BESS\npower injection (MW)")
ax2.set_xlim(-2, 40)
ax2.set_title("(b) Sub-cycle injection", fontsize=8.5)
ax2.legend(fontsize=6.0, frameon=False, loc="lower right")

# --- Panel C: cumulative communication messages + idle inset ---
ax3 = fig.add_subplot(gs[2])
t_s_rel = time - t_fault
ax3.plot(t_s_rel, cm_evt, color="#C44E52", lw=1.3, label="Event-triggered")
ax3.plot(t_s_rel, cm_poll, color="#4C72B0", lw=1.1, ls="--", label="Periodic (10 Hz)")
ax3.axvline(0, color="k", lw=0.6, ls=":")
ax3.set_xlabel("Time since contingency (s)")
ax3.set_ylabel("Cumulative messages/agent")
ax3.set_xlim(-0.1, 5.9)
ax3.set_title("(c) Cumulative messages", fontsize=8.5)
ax3.legend(fontsize=6.0, frameon=False, loc="upper left")

# Idle-period inset showing pre-contingency window [-1.0, 0.0] s
ax3_in = ax3.inset_axes([0.16, 0.49, 0.42, 0.30])
t_idle = np.linspace(-1.0, 0.0, 200)
ax3_in.plot(t_idle, np.zeros_like(t_idle), color="#C44E52", lw=1.1)
ax3_in.plot(t_idle, np.floor((t_idle + 1.0) * 10.0), color="#4C72B0", lw=1.0, ls="--")
ax3_in.set_title("Idle window (s)", fontsize=5.8, pad=2)
ax3_in.tick_params(labelsize=5.5)
ax3_in.set_xlim(-1.0, 0.0)
ax3_in.set_ylim(-0.8, 11.0)

for ax in (ax1, ax2, ax3):
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig("figs/fig1_response.pdf", bbox_inches="tight")
fig.savefig("figs/fig1_response.png", dpi=400, bbox_inches="tight")
print("Figure 1 saved.")

with open("results_summary.txt", "w") as fh:
    fh.write(f"nadir_base={nadir_base:.4f}\n")
    fh.write(f"nadir_poll={nadir_poll:.4f}\n")
    fh.write(f"nadir_evt={nadir_evt:.4f}\n")
    fh.write(f"Pss_MW={Pss:.3f}\n")
    fh.write(f"t90_ms={t90_ms:.3f}\n")
    fh.write(f"delta_fade_pp_{n_years}yr={delta_fade_final:.5f}\n")
    fh.write(f"EFC_per_year={EFC_per_year:.5f}\n")

# =============================================================
# 5. NEW: adaptive-threshold event-triggered controller
#    eta_i(t) = eta0 + (eta_relaxed-eta0)*(1-exp(-(t-t_fault)/tau_relax))
#    Each agent keeps the same tight threshold eta0 used above during
#    the initial sub-cycle transient (preserving the nadir/settling-time
#    performance already reported), then relaxes the threshold over a
#    tau_relax ~ 0.6 s window as the swing decays, directly targeting the
#    post-contingency message-count penalty identified for fixed-eta
#    event triggering. No extra broadcast is needed: each agent times
#    its own threshold schedule from the local frequency-deviation onset
#    it already measures.
# =============================================================
eta0, eta_relaxed, tau_relax = eta, 0.09, 0.6


def run_adaptive(dPL_frac=0.05):
    dPL_local = dPL_frac * Sbase
    f = 0.0
    Pm = 0.0
    f_trace = np.zeros(N)
    pbess_trace = np.zeros(N)
    msg_count = np.zeros(n_agents)
    p_hold = np.zeros(n_agents)
    last_evt = np.full(n_agents, -1.0)

    for k in range(N):
        t = time[k]
        PL = dPL_local if t >= t_fault else 0.0

        idx = np.clip(k - tau_steps, 0, k)
        idxm = np.clip(k - tau_steps - 1, 0, k)
        valid = time[idx] >= t_fault
        f_meas = np.where(valid, f_trace[idx], 0.0)
        df_meas = np.where(valid, (f_trace[idx] - f_trace[idxm]) / dt, 0.0)
        p_star = np.clip(-Kd * df_meas - Kp * f_meas, -Pmax_agent, Pmax_agent)

        elapsed = np.where(valid, t - (t_fault + tau), 0.0)
        eta_i = eta0 + (eta_relaxed - eta0) * (1.0 - np.exp(-np.clip(elapsed, 0, None) / tau_relax))
        trig = (np.abs(p_star - p_hold) > eta_i * Pmax_agent) & (t - last_evt > dwell)

        msg_count += trig.astype(float)
        p_hold = np.where(trig, p_star, p_hold)
        last_evt = np.where(trig, t, last_evt)
        Pbess_MW = p_hold.sum()

        PL_pu = PL / Sbase
        Pbess_pu = Pbess_MW / Sbase
        dPm = (-Pm - f / R) / Tg
        df = (Pm - PL_pu - D * f + Pbess_pu) / (2 * H)
        Pm += dPm * dt
        f += df * dt

        f_trace[k] = f
        pbess_trace[k] = Pbess_MW

    return f_trace, pbess_trace, msg_count.mean()


def run_fixed_with_count(dPL_frac=0.05, event_triggered=True, poll_period=0.100):
    """Same as run(), but also returns mean per-agent message count over [0,Tend]."""
    dPL_local = dPL_frac * Sbase
    f = 0.0
    Pm = 0.0
    f_trace = np.zeros(N)
    pbess_trace = np.zeros(N)
    msg_count = np.zeros(n_agents)
    p_hold = np.zeros(n_agents)
    last_evt = np.full(n_agents, -1.0)

    for k in range(N):
        t = time[k]
        PL = dPL_local if t >= t_fault else 0.0
        idx = np.clip(k - tau_steps, 0, k)
        idxm = np.clip(k - tau_steps - 1, 0, k)
        valid = time[idx] >= t_fault
        f_meas = np.where(valid, f_trace[idx], 0.0)
        df_meas = np.where(valid, (f_trace[idx] - f_trace[idxm]) / dt, 0.0)
        p_star = np.clip(-Kd * df_meas - Kp * f_meas, -Pmax_agent, Pmax_agent)

        if event_triggered:
            trig = (np.abs(p_star - p_hold) > eta * Pmax_agent) & (t - last_evt > dwell)
        else:
            trig = (t - last_evt) >= poll_period

        msg_count += trig.astype(float)
        p_hold = np.where(trig, p_star, p_hold)
        last_evt = np.where(trig, t, last_evt)
        Pbess_MW = p_hold.sum()

        PL_pu = PL / Sbase
        Pbess_pu = Pbess_MW / Sbase
        dPm = (-Pm - f / R) / Tg
        df = (Pm - PL_pu - D * f + Pbess_pu) / (2 * H)
        Pm += dPm * dt
        f += df * dt

        f_trace[k] = f
        pbess_trace[k] = Pbess_MW

    return f_trace, pbess_trace, msg_count.mean()


# ---- 5a. Head-to-head: fixed- vs adaptive-threshold event triggering ----
f_evt2, p_evt2, msgs_fixed_evt = run_fixed_with_count(0.05, event_triggered=True)
f_adap, p_adap, msgs_adaptive = run_adaptive(0.05)
nadir_adap = f0 + f_adap.min() * f0

print(f"\n[Adaptive threshold] nadir: {nadir_adap:.3f} Hz "
      f"(fixed-threshold event-triggered: {nadir_evt:.3f} Hz)")
print(f"[Adaptive threshold] mean messages/agent over {Tend:.0f} s: {msgs_adaptive:.1f} "
      f"(fixed-threshold: {msgs_fixed_evt:.1f})")

# ---- 5b. Robustness sweep over contingency size ----
contingency_fracs = [0.03, 0.05, 0.08]
sweep_rows = []
for frac in contingency_fracs:
    f_b, _, _ = run_fixed_with_count(frac, event_triggered=True, poll_period=1e9)  # dummy poll unused
    # no-support case
    f_ns, _ = run(active=False) if frac == 0.05 else (None, None)
    f_fix, _, m_fix = run_fixed_with_count(frac, event_triggered=True)
    f_pol, _, m_pol = run_fixed_with_count(frac, event_triggered=False, poll_period=0.100)
    f_ada, _, m_ada = run_adaptive(frac)
    nadir_fix = f0 + f_fix.min() * f0
    nadir_pol = f0 + f_pol.min() * f0
    nadir_ada = f0 + f_ada.min() * f0
    sweep_rows.append((frac, nadir_fix, nadir_pol, nadir_ada, m_fix, m_pol, m_ada))
    print(f"[Sweep {frac*100:.0f}% loss] nadir fixed-ET={nadir_fix:.3f} Hz, "
          f"poll={nadir_pol:.3f} Hz, adaptive-ET={nadir_ada:.3f} Hz | "
          f"msgs fixed-ET={m_fix:.1f}, adaptive-ET={m_ada:.1f}")

# ---- 5c. eta trade-off curve (nadir vs message count) at 5% loss ----
eta_grid = [0.005, 0.01, 0.02, 0.04, 0.08, 0.15]
eta_nadirs, eta_msgs = [], []
_eta_save = eta
for eta_val in eta_grid:
    eta = eta_val
    f_e, _, m_e = run_fixed_with_count(0.05, event_triggered=True)
    eta_nadirs.append(f0 + f_e.min() * f0)
    eta_msgs.append(m_e)
eta = _eta_save  # restore

with open("results_summary.txt", "a") as fh:
    fh.write(f"nadir_adaptive={nadir_adap:.4f}\n")
    fh.write(f"msgs_fixed_evt_per_agent_6s={msgs_fixed_evt:.2f}\n")
    fh.write(f"msgs_adaptive_per_agent_6s={msgs_adaptive:.2f}\n")
    for row in sweep_rows:
        fh.write("sweep,%.2f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f\n" % row)
    for ev, en, em in zip(eta_grid, eta_nadirs, eta_msgs):
        fh.write(f"eta_sweep,{ev:.3f},{en:.4f},{em:.2f}\n")

# =============================================================
# 6. Figure 2: novel-experiment panel (sensitivity + adaptive controller)
# =============================================================
fig2 = plt.figure(figsize=(7.4, 2.75))
gs2 = fig2.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.05], wspace=0.85)

# --- Panel A: eta trade-off curve ---
axA = fig2.add_subplot(gs2[0])
axA_twin = axA.twinx()
l1, = axA.plot(eta_grid, eta_nadirs, "-o", color="#C44E52", ms=3.2, lw=1.2, label="Nadir")
l2, = axA_twin.plot(eta_grid, eta_msgs, "-s", color="#4C72B0", ms=3.2, lw=1.2, label="Messages")
axA.set_xscale("log")
axA.ticklabel_format(axis="y", style="plain", useOffset=False)
axA.set_xlabel(r"Threshold $\eta_0$")
axA.set_ylabel("Nadir (Hz)", color="#C44E52")
axA_twin.set_ylabel("Msgs/agent (6 s)", color="#4C72B0")
axA.tick_params(axis="y", labelcolor="#C44E52")
axA_twin.tick_params(axis="y", labelcolor="#4C72B0")
axA.set_title("(a) Threshold trade-off", fontsize=8.5)
axA.legend(handles=[l1, l2], fontsize=5.8, frameon=False, loc="center right")

# --- Panel B: robustness across contingency sizes ---
axB = fig2.add_subplot(gs2[1])
fracs_pct = [r[0] * 100 for r in sweep_rows]
w = 0.65
dev_poll = [f0 - r[2] for r in sweep_rows]
dev_fix = [f0 - r[1] for r in sweep_rows]
dev_ada = [f0 - r[3] for r in sweep_rows]
axB.bar([f - w for f in fracs_pct], dev_poll, width=w, color="0.6", label="Periodic")
axB.bar(fracs_pct, dev_fix, width=w, color="#C44E52", label="FETC (fixed)")
axB.bar([f + w for f in fracs_pct], dev_ada, width=w, color="#55A868", label="PULSE (adaptive)")
axB.set_xticks([3, 5, 8])
axB.set_xticklabels(["3%", "5%", "8%"])
axB.set_ylim(0, 0.72)
axB.set_xlabel("Generation loss (%)")
axB.set_ylabel("Nadir deviation from 60 Hz (Hz)")
axB.set_title("(b) Robustness vs. contingency size", fontsize=8.5)
axB.legend(fontsize=5.4, frameon=False, loc="upper left")

# --- Panel C: message count, fixed vs adaptive, across contingency sizes ---
axC = fig2.add_subplot(gs2[2])
axC.bar([f - 0.45 for f in fracs_pct], [r[4] for r in sweep_rows], width=0.9,
        color="#C44E52", label="FETC (fixed)")
axC.bar([f + 0.45 for f in fracs_pct], [r[6] for r in sweep_rows], width=0.9,
        color="#55A868", label="PULSE (adaptive)")
axC.set_xticks([3, 5, 8])
axC.set_xticklabels(["3%", "5%", "8%"])
axC.set_ylim(0, 2600)
axC.set_xlabel("Generation loss (%)")
axC.set_ylabel("Messages/agent (6 s)")
axC.set_title("(c) PULSE vs. FETC messaging", fontsize=8.5)
axC.legend(fontsize=5.4, frameon=False, loc="upper right")

for ax in (axA, axB, axC):
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axA.spines["right"].set_visible(True)
axA_twin.spines["top"].set_visible(False)

fig2.tight_layout()
fig2.savefig("figs/fig2_sensitivity.pdf", bbox_inches="tight")
fig2.savefig("figs/fig2_sensitivity.png", dpi=400, bbox_inches="tight")
print("Figure 2 saved.")

msg_reduction_pct = 100.0 * (1 - msgs_adaptive / msgs_fixed_evt)
print(f"Adaptive threshold reduces post-contingency messaging by {msg_reduction_pct:.1f}% "
      f"relative to fixed-threshold event triggering, at a nadir of {nadir_adap:.3f} Hz "
      f"vs {nadir_evt:.3f} Hz.")

# =============================================================
# 7. NEW EXPERIMENT A: robustness under communication network
#    imperfections (Bernoulli packet loss + log-normal latency
#    jitter on top of the baseline sensing/relay latency tau_i).
#    A triggered update is dropped with probability p_drop; if
#    delivered, it is applied only after an extra jitter delay
#    sampled from a log-normal distribution (occasional spikes
#    up to jitter_max_ms). Either way the agent's dwell timer is
#    consumed at the attempt instant (fire-and-forget relay), so
#    a lost or delayed packet is not silently retried immediately.
# =============================================================
def run_lossy(dPL_frac=0.05, p_drop=0.0, jitter_scale_ms=3.0, jitter_sigma=0.9,
              jitter_max_ms=50.0, seed=0):
    dPL_local = dPL_frac * Sbase
    local_rng = np.random.default_rng(9000 + seed)
    f = 0.0
    Pm = 0.0
    f_trace = np.zeros(N)
    pbess_trace = np.zeros(N)
    p_hold = np.zeros(n_agents)
    last_evt = np.full(n_agents, -1.0)
    pending_time = np.full(n_agents, np.inf)
    pending_val = np.zeros(n_agents)

    for k in range(N):
        t = time[k]
        PL = dPL_local if t >= t_fault else 0.0
        idx = np.clip(k - tau_steps, 0, k)
        idxm = np.clip(k - tau_steps - 1, 0, k)
        valid = time[idx] >= t_fault
        f_meas = np.where(valid, f_trace[idx], 0.0)
        df_meas = np.where(valid, (f_trace[idx] - f_trace[idxm]) / dt, 0.0)
        p_star = np.clip(-Kd * df_meas - Kp * f_meas, -Pmax_agent, Pmax_agent)

        trig = (np.abs(p_star - p_hold) > eta * Pmax_agent) & (t - last_evt > dwell)
        if np.any(trig):
            trig_idx = np.where(trig)[0]
            dropped = local_rng.random(trig_idx.size) < p_drop
            delivered_idx = trig_idx[~dropped]
            if delivered_idx.size:
                jitter_s = local_rng.lognormal(mean=np.log(jitter_scale_ms * 1e-3),
                                                sigma=jitter_sigma, size=delivered_idx.size)
                jitter_s = np.minimum(jitter_s, jitter_max_ms * 1e-3)
                pending_time[delivered_idx] = t + jitter_s
                pending_val[delivered_idx] = p_star[delivered_idx]
            last_evt[trig_idx] = t  # dwell timer consumed on attempt, delivered or not

        due = pending_time <= t
        if np.any(due):
            p_hold[due] = pending_val[due]
            pending_time[due] = np.inf

        Pbess_MW = p_hold.sum()
        PL_pu = PL / Sbase
        Pbess_pu = Pbess_MW / Sbase
        dPm = (-Pm - f / R) / Tg
        df = (Pm - PL_pu - D * f + Pbess_pu) / (2 * H)
        Pm += dPm * dt
        f += df * dt

        f_trace[k] = f
        pbess_trace[k] = Pbess_MW

    return f_trace, pbess_trace


def windowed_rocof_hz_s(f_trace, window_s=0.5, smooth_ms=5.0):
    """Peak RoCoF magnitude (Hz/s) over the first `window_s` after the
    contingency, computed from a lightly smoothed frequency trace to
    suppress discretization noise from individual trigger events."""
    i0 = int(t_fault / dt)
    i1 = i0 + int(window_s / dt)
    seg = f_trace[i0:i1]
    smooth_n = max(1, int(round(smooth_ms * 1e-3 / dt)))
    if smooth_n > 1:
        kernel = np.ones(smooth_n) / smooth_n
        seg = np.convolve(seg, kernel, mode="same")
    slope_pu = np.diff(seg) / dt
    return float(np.min(slope_pu) * f0)  # most negative slope -> Hz/s (negative)


packet_loss_grid = [0.0, 0.05, 0.10, 0.20, 0.30]
loss_nadirs, loss_rocofs = [], []
f_loss_example = {}
for p_drop in packet_loss_grid:
    f_l, p_l = run_lossy(0.05, p_drop=p_drop, seed=int(p_drop * 100))
    loss_nadirs.append(f0 + f_l.min() * f0)
    loss_rocofs.append(abs(windowed_rocof_hz_s(f_l)))
    if p_drop in (0.0, 0.20):
        f_loss_example[p_drop] = f_l

print("\n[Comm. robustness] packet loss sweep (5% contingency):")
for pd_, nd_, rc_ in zip(packet_loss_grid, loss_nadirs, loss_rocofs):
    print(f"  p_drop={pd_*100:4.0f}%  nadir={nd_:.3f} Hz  peak RoCoF={rc_:.3f} Hz/s")

with open("results_summary.txt", "a") as fh:
    for pd_, nd_, rc_ in zip(packet_loss_grid, loss_nadirs, loss_rocofs):
        fh.write(f"packet_loss_sweep,{pd_:.2f},{nd_:.4f},{rc_:.4f}\n")

# ---- Figure 3: communication-robustness panel ----
fig3 = plt.figure(figsize=(7.2, 2.75))
gs3 = fig3.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.55)

ax3a = fig3.add_subplot(gs3[0])
t_ms_full = (time - t_fault) * 1000
mask3 = (t_ms_full >= -5) & (t_ms_full <= 60)
ax3a.plot(t_ms_full[mask3], f_loss_example[0.0][mask3] * f0 + f0, color="#C44E52",
          lw=1.3, label="0% packet loss")
ax3a.plot(t_ms_full[mask3], f_loss_example[0.20][mask3] * f0 + f0, color="#4C72B0",
          lw=1.1, ls="--", label="20% packet loss")
ax3a.axvline(0, color="k", lw=0.6, ls=":")
ax3a.axvline(16.67, color="0.5", lw=0.6, ls=":")
ax3a.ticklabel_format(axis="y", style="plain", useOffset=False)
ax3a.set_xlabel("Time since contingency (ms)")
ax3a.set_ylabel("Frequency (Hz)")
ax3a.set_title("(a) Sub-cycle response under loss", fontsize=8.5)
ax3a.legend(fontsize=6.2, frameon=False, loc="lower left")

ax3b = fig3.add_subplot(gs3[1])
ax3b_twin = ax3b.twinx()
pl_pct = [p * 100 for p in packet_loss_grid]
l1, = ax3b.plot(pl_pct, loss_nadirs, "-o", color="#C44E52", ms=3.2, lw=1.2, label="Nadir")
l2, = ax3b_twin.plot(pl_pct, loss_rocofs, "-s", color="#4C72B0", ms=3.2, lw=1.2, label="Peak RoCoF")
ax3b.set_ylim(59.738, 59.776)
ax3b_twin.set_ylim(0.448, 0.464)
ax3b.ticklabel_format(axis="y", style="plain", useOffset=False)
ax3b.set_xlabel("Packet loss (%)")
ax3b.set_ylabel("Nadir (Hz)", color="#C44E52")
ax3b_twin.set_ylabel("Peak RoCoF (Hz/s)", color="#4C72B0")
ax3b.tick_params(axis="y", labelcolor="#C44E52")
ax3b_twin.tick_params(axis="y", labelcolor="#4C72B0")
ax3b.set_title("(b) Nadir / RoCoF vs. packet loss", fontsize=8.5)
ax3b.legend(handles=[l1, l2], fontsize=5.8, frameon=False, loc="upper right")

for ax_ in (ax3a, ax3b):
    ax_.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax_.spines[s].set_visible(False)
ax3b.spines["right"].set_visible(True)

fig3.tight_layout()
fig3.savefig("figs/fig3_comm_robustness.pdf", bbox_inches="tight")
fig3.savefig("figs/fig3_comm_robustness.png", dpi=400, bbox_inches="tight")
print("Figure 3 saved.")

# =============================================================
# 8. NEW EXPERIMENT B: heterogeneous state-of-charge (SoC) and
#    IT-backup reserve preservation. Each agent's initial SoC is
#    drawn once from SoC_i ~ U(70%, 95%); a reserve floor
#    SoC_min = 80% is protected exclusively for N+1 backup duty.
#    An agent's available fraction of its rated power tapers
#    linearly to zero over a 10-percentage-point margin above the
#    floor and is fully available at SoC_i >= 90%; agents at or
#    below the floor do not participate in the grid-support loop
#    at all. This is a static per-event allocation (a single
#    contingency changes SoC by a negligible amount, per the
#    throughput estimate in Sec. 2.3/3), not a dynamic depletion
#    during the transient.
# =============================================================
soc_rng = np.random.default_rng(42)
soc_i = soc_rng.uniform(70.0, 95.0, n_agents)
soc_min_reserve = 80.0
soc_margin = 10.0
avail_i = np.clip((soc_i - soc_min_reserve) / soc_margin, 0.0, 1.0)
Pmax_i_soc = Pmax_agent * avail_i
mean_avail_pct = 100.0 * avail_i.mean()
frac_fully_reserved = 100.0 * np.mean(avail_i == 0.0)


def run_soc_constrained(dPL_frac=0.05, n_bins=5):
    """As run(active=True, event_triggered=True) but with a per-agent
    saturation limit Pmax_i_soc reflecting SoC-aware backup-reserve
    preservation. Returns the frequency trace, aggregate injection,
    and the injection binned into n_bins groups by SoC for the
    power-sharing timeline plot."""
    dPL_local = dPL_frac * Sbase
    f = 0.0
    Pm = 0.0
    f_trace = np.zeros(N)
    pbess_trace = np.zeros(N)
    p_hold = np.zeros(n_agents)
    last_evt = np.full(n_agents, -1.0)

    bin_edges = np.linspace(soc_i.min(), soc_i.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(soc_i, bin_edges[1:-1]), 0, n_bins - 1)
    group_trace = np.zeros((N, n_bins))

    for k in range(N):
        t = time[k]
        PL = dPL_local if t >= t_fault else 0.0
        idx = np.clip(k - tau_steps, 0, k)
        idxm = np.clip(k - tau_steps - 1, 0, k)
        valid = time[idx] >= t_fault
        f_meas = np.where(valid, f_trace[idx], 0.0)
        df_meas = np.where(valid, (f_trace[idx] - f_trace[idxm]) / dt, 0.0)
        p_star = np.clip(-Kd * df_meas - Kp * f_meas, -Pmax_i_soc, Pmax_i_soc)

        trig = (np.abs(p_star - p_hold) > eta * Pmax_agent) & (t - last_evt > dwell)
        p_hold = np.where(trig, p_star, p_hold)
        last_evt = np.where(trig, t, last_evt)
        Pbess_MW = p_hold.sum()

        PL_pu = PL / Sbase
        Pbess_pu = Pbess_MW / Sbase
        dPm = (-Pm - f / R) / Tg
        df = (Pm - PL_pu - D * f + Pbess_pu) / (2 * H)
        Pm += dPm * dt
        f += df * dt

        f_trace[k] = f
        pbess_trace[k] = Pbess_MW
        for b in range(n_bins):
            group_trace[k, b] = p_hold[bin_idx == b].sum()

    return f_trace, pbess_trace, group_trace, bin_edges


f_soc, p_soc, group_trace, soc_bin_edges = run_soc_constrained(0.05, n_bins=5)
nadir_soc = f0 + f_soc.min() * f0

print(f"\n[SoC-aware reserve] mean available capacity fraction: {mean_avail_pct:.1f}% "
      f"({frac_fully_reserved:.1f}% of agents fully reserved, SoC below {soc_min_reserve:.0f}%)")
print(f"[SoC-aware reserve] nadir with heterogeneous SoC + reserve floor: {nadir_soc:.3f} Hz "
      f"(full-fleet fixed-threshold nadir: {nadir_evt:.3f} Hz; no-support nadir: {nadir_base:.3f} Hz)")

with open("results_summary.txt", "a") as fh:
    fh.write(f"soc_mean_avail_pct={mean_avail_pct:.2f}\n")
    fh.write(f"soc_frac_fully_reserved_pct={frac_fully_reserved:.2f}\n")
    fh.write(f"nadir_soc_constrained={nadir_soc:.4f}\n")

# ---- Figure 4: SoC-heterogeneity panel ----
fig4 = plt.figure(figsize=(7.2, 2.75))
gs4 = fig4.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.55)

ax4a = fig4.add_subplot(gs4[0])
mask4 = (t_ms_full >= -100) & (t_ms_full <= 4000)
colors5 = ["#9AA5B1", "#7FA6C9", "#C4A5D6", "#E8A46F", "#C44E52"]
stack = np.zeros(mask4.sum())
t_plot = t_ms_full[mask4]
labels5 = [f"{soc_bin_edges[b]:.0f}-{soc_bin_edges[b+1]:.0f}%" for b in range(5)]
for b in range(5):
    vals = group_trace[mask4, b]
    ax4a.fill_between(t_plot, stack, stack + vals, color=colors5[b], lw=0, label=labels5[b])
    stack = stack + vals
ax4a.set_ylim(-0.6, 5.85)
ax4a.set_xlabel("Time since contingency (ms)")
ax4a.set_ylabel("Power by SoC band (MW)")
ax4a.set_title("(a) Power sharing by SoC group", fontsize=8.5)
ax4a.legend(fontsize=5.2, frameon=False, ncol=2, loc="upper right", title="SoC band",
            title_fontsize=5.4)

ax4b = fig4.add_subplot(gs4[1])
ax4b.plot(t_ms_full[mask4], f_base[mask4] * f0 + f0, color="0.55", lw=1.1, label="No support")
ax4b.plot(t_ms_full[mask4], f_evt[mask4] * f0 + f0, color="#C44E52", lw=1.3,
          label="Full fleet")
ax4b.plot(t_ms_full[mask4], f_soc[mask4] * f0 + f0, color="#55A868", lw=1.2, ls="--",
          label="SoC-constrained")
ax4b.axhline(59.70, color="0.4", lw=0.6, ls=":")
ax4b.axvline(0, color="k", lw=0.6, ls=":")
ax4b.set_xlabel("Time since contingency (ms)")
ax4b.set_ylabel("Frequency (Hz)")
ax4b.set_title("(b) Nadir with reserve preserved", fontsize=8.5)
ax4b.legend(fontsize=6.0, frameon=False, loc="lower right")

for ax_ in (ax4a, ax4b):
    ax_.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax_.spines[s].set_visible(False)

fig4.tight_layout()
fig4.savefig("figs/fig4_soc_reserve.pdf", bbox_inches="tight")
fig4.savefig("figs/fig4_soc_reserve.png", dpi=400, bbox_inches="tight")
print("Figure 4 saved.")

# =============================================================
# 9. NEW EXPERIMENT C: Realistic PMU/PLL measurement filtering,
#    Gaussian measurement noise, frequency deadband, and
#    derivative gain (Kd) vs. saturation regime analysis.
# =============================================================
def run_pmu_gain(Kd_val=8.0, Kp_val=9.0, tau_f=0.0167, tau_r=0.0333,
                 noise_std_hz=0.0005, deadband_hz=0.015, seed=101):
    """Simulate fleet response with two-stage PLL/PMU filtering (tau_f for
    frequency estimation, tau_r for RoCoF smoothing), additive white
    Gaussian frequency noise (noise_std_hz), and a deadband."""
    local_rng = np.random.default_rng(seed)
    f = 0.0
    Pm = 0.0
    f_trace = np.zeros(N)
    pbess_trace = np.zeros(N)
    p_hold = np.zeros(n_agents)
    last_evt = np.full(n_agents, -1.0)
    f_est = np.zeros(n_agents)
    f_est_prev = np.zeros(n_agents)
    df_est = np.zeros(n_agents)
    db_pu = deadband_hz / f0
    noise_pu = noise_std_hz / f0

    for k in range(N):
        t = time[k]
        PL = dPL if t >= t_fault else 0.0
        idx = np.clip(k - tau_steps, 0, k)
        idxm = np.clip(k - tau_steps - 1, 0, k)
        valid = time[idx] >= t_fault
        f_raw = np.where(valid, f_trace[idx], 0.0)

        if tau_f > 0.0:
            w_noise = local_rng.normal(0.0, noise_pu, n_agents) if t >= t_fault else 0.0
            f_noisy = f_raw + w_noise
            f_est_new = f_est + (dt / tau_f) * (f_noisy - f_est)
            raw_df = (f_est_new - f_est_prev) / dt
            df_est = df_est + (dt / tau_r) * (raw_df - df_est)
            f_est_prev = f_est.copy()
            f_est = f_est_new
            f_eff = np.where(np.abs(f_est) > db_pu, f_est, 0.0)
            df_eff = np.where(np.abs(f_est) > db_pu, df_est, 0.0)
        else:
            f_eff = f_raw
            df_eff = np.where(valid, (f_trace[idx] - f_trace[idxm]) / dt, 0.0)

        p_star = np.clip(-Kd_val * df_eff - Kp_val * f_eff, -Pmax_agent, Pmax_agent)
        trig = (np.abs(p_star - p_hold) > eta * Pmax_agent) & (t - last_evt > dwell)
        p_hold = np.where(trig, p_star, p_hold)
        last_evt = np.where(trig, t, last_evt)
        Pbess_MW = p_hold.sum()

        PL_pu = PL / Sbase
        Pbess_pu = Pbess_MW / Sbase
        dPm = (-Pm - f / R) / Tg
        df = (Pm - PL_pu - D * f + Pbess_pu) / (2 * H)
        Pm += dPm * dt
        f += df * dt

        f_trace[k] = f
        pbess_trace[k] = Pbess_MW

    nadir_hz = f0 + f_trace.min() * f0
    peak_mw = pbess_trace.max()
    idx90 = np.where(pbess_trace >= 0.9 * peak_mw)[0]
    t90_ms = (time[idx90[0]] - t_fault) * 1000.0 if idx90.size else np.nan
    return f_trace, pbess_trace, nadir_hz, peak_mw, t90_ms


# 1) Compare Idealized vs 1-cycle PLL (16.7ms/33.3ms) vs 2-cycle PMU (33.3ms/100ms)
#    Use Kp=9.0 MW/pu (same as baseline) and Kd=4.0 MW*s (non-saturating linear regime)
f_pll, p_pll, nad_pll, pk_pll, t90_pll = run_pmu_gain(
    Kd_val=4.0, Kp_val=9.0, tau_f=0.0167, tau_r=0.0333, noise_std_hz=0.0005)
f_pmu2, p_pmu2, nad_pmu2, pk_pmu2, t90_pmu2 = run_pmu_gain(
    Kd_val=4.0, Kp_val=9.0, tau_f=0.0333, tau_r=0.1000, noise_std_hz=0.0005)

print(f"\n[PMU/PLL & Gain Analysis]")
print(f"  Baseline unfiltered (Kd=55, Kp=9): nadir={nadir_evt:.3f} Hz, peak={p_evt.max():.2f} MW (100% cap)")
print(f"  1-cycle PLL + 33 ms RoCoF (Kd=4, Kp=9, noisy): nadir={nad_pll:.3f} Hz, peak={pk_pll:.2f} MW ({100*pk_pll/(n_agents * Pmax_agent):.1f}% cap), t90={t90_pll:.1f} ms")
print(f"  2-cycle PMU + 100 ms RoCoF (Kd=4, Kp=9, noisy): nadir={nad_pmu2:.3f} Hz, peak={pk_pmu2:.2f} MW ({100*pk_pmu2/(n_agents * Pmax_agent):.1f}% cap), t90={t90_pmu2:.1f} ms")

# 2) Sweep Kd across linear/gradual regime and saturated regime under 1-cycle PLL (with Kp=9.0)
kd_grid = [2.0, 4.0, 6.0, 8.0, 12.0, 25.0, 55.0]
kd_nadirs, kd_peaks_pct = [], []
for kd_v in kd_grid:
    _, _, nad_k, pk_k, _ = run_pmu_gain(
        Kd_val=kd_v, Kp_val=9.0, tau_f=0.0167, tau_r=0.0333, noise_std_hz=0.0005)
    kd_nadirs.append(nad_k)
    kd_peaks_pct.append(100.0 * pk_k / (n_agents * Pmax_agent))
    print(f"    Kd={kd_v:4.1f} -> nadir={nad_k:.3f} Hz, peak={pk_k:.2f} MW ({100*pk_k/(n_agents * Pmax_agent):.1f}% cap)")

with open("results_summary.txt", "a") as fh:
    fh.write(f"pmu_1cyc_nadir={nad_pll:.4f},peak_mw={pk_pll:.3f},t90_ms={t90_pll:.1f}\n")
    fh.write(f"pmu_2cyc_nadir={nad_pmu2:.4f},peak_mw={pk_pmu2:.3f},t90_ms={t90_pmu2:.1f}\n")
    for kv, kn, kp in zip(kd_grid, kd_nadirs, kd_peaks_pct):
        fh.write(f"kd_sweep,{kv:.1f},{kn:.4f},{kp:.2f}\n")

# ---- Figure 5: PMU filtering & gain-saturation regime ----
fig5 = plt.figure(figsize=(7.2, 2.75))
gs5 = fig5.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.58)

ax5a = fig5.add_subplot(gs5[0])
mask5 = (t_ms_full >= -100) & (t_ms_full <= 5800)
ax5a.plot(t_ms_full[mask5], f_evt[mask5] * f0 + f0, color="#C44E52", lw=1.25,
          label=r"Unfiltered ($K_d=55$)")
ax5a.plot(t_ms_full[mask5], f_pll[mask5] * f0 + f0, color="#55A868", lw=1.3, ls="--",
          label=r"1-cyc PLL + 33 ms RoCoF ($K_d=4$)")
ax5a.plot(t_ms_full[mask5], f_pmu2[mask5] * f0 + f0, color="#4C72B0", lw=1.25, ls="-.",
          label=r"2-cyc PMU + 100 ms RoCoF ($K_d=4$)")
ax5a.axvline(0, color="k", lw=0.6, ls=":")
ax5a.set_ylim(59.72, 60.01)
ax5a.set_xlabel("Time since contingency (ms)")
ax5a.set_ylabel("Frequency (Hz)")
ax5a.set_title("(a) PMU/PLL filtering & noise impact", fontsize=8.5)
ax5a.legend(fontsize=5.4, frameon=False, loc="upper right")

ax5b = fig5.add_subplot(gs5[1])
ax5b_twin = ax5b.twinx()
l5_1, = ax5b.plot(kd_grid, kd_nadirs, "-o", color="#C44E52", ms=3.5, lw=1.2, label="Nadir (Hz)")
l5_2, = ax5b_twin.plot(kd_grid, kd_peaks_pct, "-s", color="#4C72B0", ms=3.5, lw=1.2,
                       label="Peak fleet power (%)")
ax5b.set_xscale("log")
ax5b.set_xticks([2, 4, 8, 12, 25, 55])
ax5b.get_xaxis().set_major_formatter( matplotlib.ticker.ScalarFormatter() )
ax5b.get_xaxis().set_minor_formatter( matplotlib.ticker.NullFormatter() )
ax5b.ticklabel_format(axis="y", style="plain", useOffset=False)
ax5b.set_xlabel(r"Derivative gain $K_d$ ($\mathrm{MW\cdot s}$)")
ax5b.set_ylabel("Nadir (Hz)", color="#C44E52")
ax5b_twin.set_ylabel("Peak fleet power (% of 12 MW)", color="#4C72B0")
ax5b.tick_params(axis="y", labelcolor="#C44E52")
ax5b_twin.tick_params(axis="y", labelcolor="#4C72B0")
ax5b.set_title("(b) Gain vs. saturation regime", fontsize=8.5)
ax5b.legend(handles=[l5_1, l5_2], fontsize=5.6, frameon=False, loc="lower right")

for ax_ in (ax5a, ax5b):
    ax_.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax_.spines[s].set_visible(False)
ax5b.spines["right"].set_visible(True)
ax5b_twin.spines["top"].set_visible(False)

fig5.tight_layout()
fig5.savefig("figs/fig5_pmu_gain.pdf", bbox_inches="tight")
fig5.savefig("figs/fig5_pmu_gain.png", dpi=400, bbox_inches="tight")
print("Figure 5 saved.")

