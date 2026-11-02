#!/usr/bin/env python3
"""
pulse_review_toolkit.py
Independent re-check + extended evaluation of the PULSE paper's claims.

Requires: numpy, matplotlib.      Run:  python pulse_review_toolkit.py [--fast]

Modelling choices that differ from the paper (change them if you disagree):
  * Euler integration, dt = 1 ms (paper: 50 us). Dwell time resolves to 2 steps.
  * Soft deadband (subtract the band) instead of a hard cut, to avoid a jump in p*.
  * Local contingency-onset detection (|df_hat| > 20 mHz), not oracle onset.
  * 5 ms first-order inverter lag on the ZOH setpoint.
  * A dropped message leaves the setpoint unchanged and is retried after the dwell time.
  * Hard local clamp p_i <= a_i * Pmax so the SoC floor is respected between events.
"""
import argparse, os, time
from dataclasses import dataclass, replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7")   # Okabe-Ito (colour-blind safe)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "savefig.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False})
OUT, SEED, UFLS_HZ = "figures", 0, 59.5   # UFLS_HZ is illustrative: set to your region's stage


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png"); fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig); print("  saved", name)


@dataclass
class Params:
    Sbase: float = 300.0; f0: float = 60.0; H: float = 3.2; D: float = 1.0
    R: float = 0.05; Tg: float = 6.0; dPL: float = 0.05
    N: int = 200; Pmax: float = 0.060; Enom: float = 30.0; P_IT: float = 30.0
    eta_dis: float = 0.95; soc_floor: float = 80.0; soc_margin: float = 10.0
    Kd: float = 4.0; Kp: float = 9.0
    eta0: float = 0.02; eta_r: float = 0.09; tau_thr: float = 0.6; dwell: float = 0.0015
    lat_lo: float = 0.001; lat_hi: float = 0.008
    db_hz: float = 0.015; rdb_hzs: float = 0.02; detect_hz: float = 0.020
    sigma_hz: float = 0.0005
    tau_f: float = 1/60; tau_r: float = 2/60; tau_inv: float = 0.005
    t_poll: float = 0.1; t_reset: float = 5.0


def sdb(x, d):                         # soft deadband
    return np.sign(x) * np.maximum(np.abs(x) - d, 0.0)

# ----------------------------------------------------------------------------
# 1. Frequency-domain analysis (independent re-computation of Eq. 18-24)
# ----------------------------------------------------------------------------
def tf_parts(p, Kd, tau_f):
    nG = [p.R * p.Tg, p.R]
    dG = [2*p.H*p.R*p.Tg, 2*p.H*p.R + p.D*p.R*p.Tg, p.D*p.R + 1]
    nC = [p.N*Kd/p.Sbase, p.N*p.Kp/p.Sbase]
    dC = [tau_f, 1.0]
    return nG, dG, nC, dC

def Lfun(p, Kd, tau_f, w):
    nG, dG, nC, dC = tf_parts(p, Kd, tau_f); s = 1j*np.asarray(w)
    return (np.polyval(nG, s)*np.polyval(nC, s)) / (np.polyval(dG, s)*np.polyval(dC, s))

def crossings(p, Kd, tau_f, w=None):
    """All 0 dB crossings: (wc [rad/s], PM [deg], tau_k = PM/wc [s])."""
    w = np.logspace(-3, 3, 8000) if w is None else w
    m = np.abs(Lfun(p, Kd, tau_f, w)); sg = np.sign(m - 1.0)
    out = []
    for i in np.where(sg[:-1]*sg[1:] < 0)[0]:
        a = (1 - m[i])/(m[i+1] - m[i])
        wc = 10**(np.log10(w[i]) + a*(np.log10(w[i+1]) - np.log10(w[i])))
        pm = (np.pi + np.angle(Lfun(p, Kd, tau_f, [wc])[0])) % (2*np.pi)
        out.append((wc, np.degrees(pm), pm/wc))
    return out

def delay_margin(p, Kd, tau_f, w=None):
    c = crossings(p, Kd, tau_f, w)
    return (min(x[2] for x in c) if c else np.inf), c

def cl_poles(p, Kd, tau_f):
    nG, dG, nC, dC = tf_parts(p, Kd, tau_f)
    return np.roots(np.polyadd(np.polymul(dG, dC), np.polymul(nG, nC)))

def analytic_report(p):
    Hv = p.N*p.Kd/(2*p.Sbase); Dv = p.N*p.Kp/p.Sbase
    r0 = lambda H: -p.dPL/(2*H)*p.f0
    dss_hz = p.f0*(1 - p.dPL/(p.D + 1/p.R + Dv))
    print("\n=== ANALYTIC CHECK (script vs paper) ===")
    print(f"Hv={Hv:.4f} s (paper 1.3333)  Dv={Dv:.4f} pu (paper 6.0)")
    print(f"RoCoF0 base={r0(p.H):.4f} Hz/s  PULSE={r0(p.H+Hv):.4f} Hz/s")
    print(f"Quasi-steady f, no BESS   = {p.f0*(1 - p.dPL/(p.D+1/p.R)):.3f} Hz")
    print(f"Quasi-steady f, with BESS = {dss_hz:.3f} Hz   (paper text says 59.750 Hz)")
    for lab, tf in (("nominal tau_f=16.7 ms", p.tau_f), ("idealised tau_f->0", 1e-5)):
        tm, cr = delay_margin(p, p.Kd, tf, np.logspace(-3, 3, 40000))
        print(f"{lab}: closed-loop poles =", np.round(cl_poles(p, p.Kd, tf), 3))
        for wc, pm, tk in cr:
            print(f"   crossover {wc:.3f} rad/s  PM={pm:.1f} deg  delay={tk:.3f} s")
        print(f"   -> delay margin = {tm:.3f} s")
    nG, dG, nC, dC = tf_parts(p, p.Kd, p.tau_f); s = 1j*1.25
    G = np.polyval(nG, s)/np.polyval(dG, s); C = np.polyval(nC, s)/np.polyval(dC, s)
    print(f"At w=1.25 rad/s: |L|={abs(G*C):.3f} (paper: 1), angle(G)={np.degrees(np.angle(G)):.1f} "
          f"(paper -105.8), angle(C)={np.degrees(np.angle(C)):.1f} (paper +29.0)")

def fig_margins(p):
    w = np.logspace(-2, 2.5, 4000); L = Lfun(p, p.Kd, p.tau_f, w)
    tm, cr = delay_margin(p, p.Kd, p.tau_f, np.logspace(-3, 3, 40000))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].semilogx(w, 20*np.log10(np.abs(L)), color=OI["blue"]); ax[0].axhline(0, color="k", lw=.8)
    for wc, pm, tk in cr:
        for a in ax[:2]: a.axvline(wc, ls=":", color=OI["verm"])
        ax[0].text(wc, 3, f" {wc:.2f}", color=OI["verm"], fontsize=8)
    ax[0].set(xlabel="rad/s", ylabel="|L| (dB)", title="Open loop L(jw): all 0 dB crossings")
    ax[1].semilogx(w, np.degrees(np.unwrap(np.angle(L))), color=OI["blue"])
    ax[1].axhline(-180, color="k", lw=.8)
    ax[1].set(xlabel="rad/s", ylabel="phase (deg)", title=f"Delay margin = {tm:.2f} s")
    Kds, tfs = np.linspace(0.5, 10, 24), np.geomspace(0.005, 0.1, 24)
    ww = np.logspace(-3, 3, 3000)
    Z = np.minimum([[delay_margin(p, k, t, ww)[0] for k in Kds] for t in tfs], 5.0)
    pc = ax[2].pcolormesh(Kds, tfs*1000, Z, shading="auto", cmap="viridis")
    cs = ax[2].contour(Kds, tfs*1000, Z, levels=[0.25, 0.5, 1.0, 2.0], colors="w", linewidths=.8)
    ax[2].clabel(cs, fmt="%.2f s", fontsize=7)
    ax[2].plot(p.Kd, p.tau_f*1000, "*", color=OI["verm"], ms=12)
    ax[2].set(xlabel="K_d (MW s)", ylabel="tau_f (ms)", yscale="log", title="Delay margin map")
    fig.colorbar(pc, ax=ax[2], label="s (capped at 5)")
    save(fig, "fig1_margins"); return tm

def lin_sim(p, delay, Kd, tau_f, T=150.0, dt=1e-3):
    n = int(T/dt); d = int(round(delay/dt)); h = np.zeros(n+1); out = np.empty(n)
    df = dPm = y = 0.0; kk = p.N/p.Sbase
    for k in range(n):
        dpl = p.dPL if k*dt >= 0.1 else 0.0
        fd = h[k-d] if k >= d else 0.0
        ydot = (-y + fd)/tau_f
        u = -kk*(Kd*ydot + p.Kp*y)
        ddf = (dPm - dpl - p.D*df + u)/(2*p.H); ddPm = (-dPm - df/p.R)/p.Tg
        y += dt*ydot; df += dt*ddf; dPm += dt*ddPm; h[k+1] = df; out[k] = df
        if abs(df) > 5: out[k:] = np.nan; break
    return np.arange(n)*dt, out

def fig_delay_validation(p, tm):
    if not np.isfinite(tm): return
    dss = -p.dPL/(p.D + 1/p.R + p.N*p.Kp/p.Sbase); fac = [0.5, 0.9, 1.1, 1.5]
    fig, ax = plt.subplots(len(fac), 1, figsize=(8, 8), sharex=True)
    for a, f in zip(ax, fac):
        t, x = lin_sim(p, f*tm, p.Kd, p.tau_f); dev = x - dss
        mid = np.nanmax(np.abs(dev[60000:90000])); late = np.nanmax(np.abs(dev[120000:]))
        grows = (not np.isfinite(late)) or late > 1.02*mid
        a.plot(t, dev*p.f0*1000, color=OI["verm"] if grows else OI["blue"])
        a.set_ylabel("mHz"); a.set_title(f"delay = {f}x margin = {f*tm:.2f} s -> "
                                         f"{'GROWING' if grows else 'decaying'}", fontsize=9)
    ax[-1].set_xlabel("time (s)"); save(fig, "fig2_delay_validation")

# ----------------------------------------------------------------------------
# 2. Time-domain simulation
# ----------------------------------------------------------------------------
def simulate(p, scheme, T=30.0, dt=1e-3, dP=None, pdrop=0.0, extra_delay=0.0, jitter=False,
             gate=True, hetero=True, ambient=False, seed=0, t_fault=0.1, record=False):
    """scheme in {'none','droop','periodic','pdb','fetc','pulse'}"""
    rng = np.random.default_rng(seed); N = p.N; n = int(round(T/dt))
    dP = p.dPL if dP is None else dP
    soc0 = rng.uniform(75, 95, N) if hetero else np.full(N, 95.0)
    Enom = rng.uniform(25, 30, N) if hetero else np.full(N, p.Enom)
    E = soc0/100*Enom
    d0 = np.rint((rng.uniform(p.lat_lo, p.lat_hi, N) + extra_delay)/dt).astype(int); d = d0.copy()
    u = 1.0/p.f0; db, rdb, det, sig = p.db_hz*u, p.rdb_hzs*u, p.detect_hz*u, p.sigma_hz*u
    df = dPm = amb = 0.0
    dfh = np.zeros(N); rh = np.zeros(N); pw = np.zeros(N); pset = np.zeros(N)
    tlast = np.full(N, -1.0); armed = np.zeros(N, bool); t_on = np.zeros(N); quiet = np.zeros(N)
    msgs = np.zeros(N); hist = np.zeros(n+1)
    f_hz = np.empty(n); pb = np.empty(n); mrec = np.empty(n)
    sent_m = np.zeros((n, N), bool) if record else None
    tr = np.zeros((n, 4)) if record else None
    ia = int(np.argmax(soc0)); per = max(1, int(round(p.t_poll/dt)))
    for k in range(n):
        t = k*dt; dpl = dP if t >= t_fault else 0.0
        if ambient:
            amb += dt*(-amb/5.0) + 0.0015*np.sqrt(2*dt/5.0)*rng.standard_normal()
        ddf = (dPm - dpl - amb - p.D*df + pw.sum()/p.Sbase)/(2*p.H)
        ddPm = (-dPm - df/p.R)/p.Tg
        if jitter and k % 50 == 0:
            d = d0 + np.rint(np.minimum(rng.lognormal(np.log(0.005), 0.8, N), 0.05)/dt).astype(int)
        fd = hist[np.maximum(k - d, 0)]
        dfd = (-dfh + fd + sig*rng.standard_normal(N))/p.tau_f
        rh += dt*(-rh + dfd)/p.tau_r; dfh += dt*dfd
        if scheme != "none":
            soc = E/Enom*100
            a = np.clip((soc - p.soc_floor)/p.soc_margin, 0, 1) if gate else np.ones(N)
            pm = a*p.Pmax
            pstar = np.clip(-p.Kd*sdb(rh, rdb) - p.Kp*sdb(dfh, db), 0, pm)
            trig = None; eta_i = p.eta0
            if scheme == "droop":
                pset = pstar
            else:
                if scheme in ("fetc", "pulse"):
                    eta = p.eta0
                    if scheme == "pulse":
                        new = (np.abs(dfh) > det) & ~armed
                        t_on[new] = t; armed |= new
                        quiet = np.where(np.abs(dfh) < db, quiet + dt, 0.0)
                        armed &= ~(quiet > p.t_reset)
                        eta = np.where(armed, p.eta0 + (p.eta_r - p.eta0)*(1 - np.exp(-(t - t_on)/p.tau_thr)), p.eta0)
                        eta_i = eta[ia]
                    trig = (np.abs(pstar - pset) > eta*p.Pmax) & (t - tlast >= p.dwell)
                elif scheme == "periodic":
                    trig = np.full(N, k % per == 0)
                else:  # 'pdb' periodic + threshold
                    trig = np.logical_and(k % per == 0, np.abs(pstar - pset) > 0.005*p.Pmax)
                ok = (rng.random(N) >= pdrop) if pdrop > 0 else True
                pset = np.where(trig & ok, pstar, pset)
                tlast = np.where(trig, t, tlast); msgs += trig
            pw += dt*(pset - pw)/p.tau_inv
            pw = np.minimum(pw, pm)
            E = E - pw*1000/p.eta_dis*dt/3600          # MW -> kW, s -> h
            if record and trig is not None: sent_m[k] = trig
            if record and scheme in ("fetc", "pulse"): tr[k] = (eta_i, pstar[ia], pset[ia], pw[ia])
        df += dt*ddf; dPm += dt*ddPm; hist[k+1] = df
        f_hz[k] = p.f0*(1 + df); pb[k] = pw.sum(); mrec[k] = msgs.mean()
    t = np.arange(n)*dt
    return dict(t=t, f=f_hz, pb=pb, msg=mrec, msgs=msgs, soc0=soc0, E=E, sent=sent_m, trace=tr,
                runtime=E*p.eta_dis/p.P_IT*60, nadir=f_hz[t >= t_fault].min(), agent=ia)

def idx(r, t): return int(round(t/(r["t"][1] - r["t"][0]))) - 1
def msgs_between(r, t0, t1): return r["msg"][idx(r, t1)] - r["msg"][idx(r, t0)]
def rocof_min(r, win, t0=0.1):
    dt = r["t"][1] - r["t"][0]; k = int(round(win/dt))
    x = (r["f"][k:] - r["f"][:-k])/win
    return x[r["t"][k:] >= t0].min()

SCHEMES = [("none", "No support", OI["black"], "-"),
           ("droop", "Local droop/PD (no messages)", OI["sky"], "--"),
           ("periodic", "Periodic 100 ms", OI["purple"], ":"),
           ("pdb", "Periodic + threshold", OI["green"], "-."),
           ("fetc", "FETC (fixed eta)", OI["orange"], "-"),
           ("pulse", "PULSE (adaptive eta)", OI["verm"], "-")]

def fig_schemes(p):
    R = {k: simulate(p, k, T=30, record=k in ("fetc", "pulse"), seed=SEED) for k, *_ in SCHEMES}
    print("\n=== CANONICAL CONFIG (heterogeneous, SoC-gated): all schemes, same fleet ===")
    print(f"{'scheme':10s} {'nadir Hz':>9s} {'RoCoF100':>9s} {'RoCoF500':>9s} {'msg 6s':>7s} {'msg10s':>7s} {'msg30s':>7s} {'minRun':>7s}")
    for k, *_ in SCHEMES:
        r = R[k]
        print(f"{k:10s} {r['nadir']:9.4f} {rocof_min(r,.1):9.3f} {rocof_min(r,.5):9.3f} "
              f"{msgs_between(r,.1,6.1):7.1f} {msgs_between(r,.1,10.1):7.1f} {msgs_between(r,.1,29.9):7.1f} {r['runtime'].min():7.1f}")
    for h in (6.1, 10.1, 29.9):
        print(f"PULSE vs FETC message reduction over {h-.1:.0f} s: "
              f"{100*(1 - msgs_between(R['pulse'],.1,h)/msgs_between(R['fetc'],.1,h)):.1f}%")
    print("Standby (ambient drift, no contingency, 20 s) messages/agent:")
    for k in ("periodic", "pdb", "fetc", "pulse"):
        print(f"   {k:9s} {simulate(p, k, T=20, dP=0.0, ambient=True, seed=1)['msgs'].mean():.2f}")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for k, lab, c, ls in SCHEMES:
        for a in ax[0]: a.plot(R[k]["t"], R[k]["f"], color=c, ls=ls, label=lab)
    ax[0, 0].set(title="30 s recovery", ylabel="Hz"); ax[0, 1].set(xlim=(0, 6), title="Zoom to nadir")
    for a in ax[0]: a.axhline(UFLS_HZ, color="grey", ls=":", lw=.8)
    ax[0, 1].text(5.9, UFLS_HZ + .01, "illustrative UFLS stage", ha="right", fontsize=7, color="grey")
    ax[0, 0].legend(fontsize=7)
    for k, lab, c, ls in SCHEMES[2:]:
        r = R[k]; ax[1, 0].plot(r["t"], r["msg"] - r["msg"][idx(r, .1)], color=c, ls=ls, label=lab)
    ax[1, 0].set(xlim=(0, 30), ylim=(0, 60), xlabel="time (s)", ylabel="cumulative msgs / agent",
                 title="Communication (periodic clipped at 60)")
    for k in ("droop", "fetc", "pulse"):
        ax[1, 1].plot(R[k]["t"], R[k]["pb"], color=dict((s[0], s[2]) for s in SCHEMES)[k], label=k)
    ax[1, 1].set(xlim=(0, 10), xlabel="time (s)", ylabel="fleet power (MW)", title="Fleet injection"); ax[1, 1].legend()
    save(fig, "fig3_schemes"); return R

def fig_raster_trace(p, R):
    soc0 = R["pulse"]["soc0"]; rank = np.empty(p.N, int); rank[np.argsort(soc0)] = np.arange(p.N)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a, k in zip(ax, ("fetc", "pulse")):
        r = R[k]; nz = np.nonzero(r["sent"][:10000])
        a.plot(r["t"][nz[0]], rank[nz[1]], "|", ms=3, color=OI["orange"] if k == "fetc" else OI["verm"])
        a.set(title=f"{k.upper()}: {msgs_between(r,.1,10.1):.1f} msgs/agent in 10 s", xlabel="time (s)")
    ax[0].set_ylabel("agent (sorted by initial SoC)"); save(fig, "fig4_raster")
    fig, ax = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for j, k in enumerate(("fetc", "pulse")):
        r = R[k]; tr = r["trace"]; t = r["t"]; m = t <= 10
        ax[0, j].plot(t[m], tr[m, 1]*1000, label="p* candidate")
        ax[0, j].step(t[m], tr[m, 2]*1000, where="post", label="ZOH setpoint")
        ax[0, j].plot(t[m], tr[m, 3]*1000, label="actual power", ls=":")
        ax[0, j].set(title=f"{k.upper()} - one fully available agent", ylabel="kW")
        ax[1, j].plot(t[m], np.abs(tr[m, 1] - tr[m, 2])/p.Pmax, label="|p* - p| / Pmax")
        ax[1, j].plot(t[m], tr[m, 0], label="threshold eta(t)"); ax[1, j].set_xlabel("time (s)")
        ax[0, j].legend(fontsize=7); ax[1, j].legend(fontsize=7)
    save(fig, "fig5_agent_trace")

def fig_pareto(p, fast):
    etas = np.geomspace(0.005, 0.15, 5 if fast else 8); pts = {"fetc": [], "pulse": []}
    for e in etas:
        for k in pts:
            r = simulate(replace(p, eta0=e, eta_r=min(0.9, 4.5*e)), k, T=10, seed=SEED)
            pts[k].append((msgs_between(r, .1, 10.1), r["nadir"]))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for k, c, mk in (("fetc", OI["orange"], "o"), ("pulse", OI["verm"], "s")):
        a = np.array(pts[k]); ax.plot(a[:, 0], a[:, 1], marker=mk, color=c, label=k.upper()+" (eta0 sweep)")
    for k, mk in (("periodic", "^"), ("pdb", "v")):
        r = simulate(p, k, T=10, seed=SEED)
        ax.plot(msgs_between(r, .1, 10.1), r["nadir"], mk, color="k", label=k)
    ax.axhline(simulate(p, "droop", T=10, seed=SEED)["nadir"], color=OI["sky"], ls="--", label="droop (0 msgs)")
    ax.set(xscale="log", xlabel="messages / agent (10 s)", ylabel="nadir (Hz)", title="Pareto: nadir vs messages")
    ax.legend(fontsize=7); save(fig, "fig6_pareto")

def fig_fleet(p, R, fast):
    r = R["pulse"]; fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].hist(r["soc0"], bins=20, color=OI["sky"]); ax[0].set(xlabel="initial SoC (%)", ylabel="racks")
    x = np.linspace(70, 100, 200); a2 = ax[0].twinx()
    a2.plot(x, np.clip((x - p.soc_floor)/p.soc_margin, 0, 1), color=OI["verm"]); a2.set_ylabel("availability a_i")
    rt = np.sort(r["runtime"]); ax[1].plot(rt, np.arange(1, p.N + 1)/p.N, color=OI["blue"])
    for v in (10, 15): ax[1].axvline(v, color="k", ls=":")
    ax[1].set(xlabel="IT runtime after event (min)", ylabel="CDF",
              title="Runtime vs assumed 10/15 min (verify the tier source)")
    floors = [60, 70, 80, 90] if fast else [60, 65, 70, 75, 80, 85, 90]; nad, mn, av = [], [], []
    for fl in floors:
        q = replace(p, soc_floor=fl); s = simulate(q, "pulse", T=10, seed=SEED)
        nad.append(s["nadir"]); mn.append(s["runtime"].min())
        av.append(np.clip((s["soc0"] - fl)/p.soc_margin, 0, 1).mean())
    ax[2].plot(floors, nad, "o-", color=OI["verm"]); ax[2].set(xlabel="SoC floor (%)", ylabel="nadir (Hz)")
    b = ax[2].twinx(); b.plot(floors, mn, "s--", color=OI["blue"]); b.set_ylabel("min IT runtime (min)")
    print("\nFloor sweep (floor, nadir, min runtime, mean availability):")
    for row in zip(floors, nad, mn, av): print("  %d  %.4f  %.1f  %.2f" % row)
    save(fig, "fig7_fleet")

def heat(ax, Z, xl, yl, xlab, ylab, title, fmt="{:.1f}", cmap="viridis"):
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap)
    ax.set(xticks=range(len(xl)), yticks=range(len(yl)), xticklabels=xl, yticklabels=yl,
           xlabel=xlab, ylabel=ylab, title=title); ax.grid(False)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, fmt.format(Z[i, j]), ha="center", va="center", color="w", fontsize=7)
    return im

def fig_robustness(p, fast):
    dl = [0, 0.1, 0.4, 0.8] if fast else [0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2]
    pdp = [0, 0.3, 0.6, 0.9] if fast else [0, 0.1, 0.3, 0.5, 0.7, 0.9]
    nad = np.zeros((len(dl), len(pdp))); osc = np.zeros_like(nad)
    for i, d_ in enumerate(dl):
        for j, pd in enumerate(pdp):
            r = simulate(p, "pulse", T=15, pdrop=pd, extra_delay=d_, seed=SEED)
            m = r["t"] >= 10; nad[i, j] = r["nadir"]; osc[i, j] = (r["f"][m].max() - r["f"][m].min())*1000
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); xl = [f"{x:.1f}" for x in pdp]; yl = [f"{x:g}" for x in dl]
    heat(ax[0], nad, xl, yl, "packet drop prob.", "extra sensing delay (s)", "nadir (Hz)", "{:.3f}")
    heat(ax[1], osc, xl, yl, "packet drop prob.", "extra sensing delay (s)", "f peak-to-peak, 10-15 s (mHz)", "{:.0f}", "magma")
    save(fig, "fig8_robustness")

def fig_contingency(p, fast):
    Hs, dPs = [2.0, 3.2, 5.0], [0.03, 0.05, 0.08]
    lift, rel, ms = (np.zeros((3, 3)) for _ in range(3))
    for i, H in enumerate(Hs):
        for j, dP in enumerate(dPs):
            q = replace(p, H=H); b = simulate(q, "none", T=6, dP=dP, seed=SEED)
            s = simulate(q, "pulse", T=6, dP=dP, seed=SEED)
            lift[i, j] = (s["nadir"] - b["nadir"])*1000
            rel[i, j] = 100*(s["nadir"] - b["nadir"])/(p.f0 - b["nadir"]); ms[i, j] = msgs_between(s, .1, 6.0)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6)); xl = [f"{int(d*100)}%" for d in dPs]; yl = [f"H={h}" for h in Hs]
    heat(ax[0], lift, xl, yl, "trip size", "inertia", "nadir lift (mHz)")
    heat(ax[1], rel, xl, yl, "trip size", "inertia", "relative lift (%)")
    heat(ax[2], ms, xl, yl, "trip size", "inertia", "PULSE msgs / agent (6 s)")
    save(fig, "fig9_contingency")

def fig_montecarlo(p, fast):
    ns = 8 if fast else 20; nads = []; fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); F = []
    for s in range(ns):
        r = simulate(p, "pulse", T=15, pdrop=0.05, jitter=True, seed=100 + s)
        nads.append(r["nadir"]); F.append(r["f"][::10]); t = r["t"][::10]
        ax[0].plot(t, F[-1], color=OI["verm"], alpha=.25, lw=.8)
    F = np.array(F); mu, sd = F.mean(0), F.std(0)
    ax[0].plot(t, mu, color="k"); ax[0].fill_between(t, mu - 2*sd, mu + 2*sd, color="grey", alpha=.3)
    ax[0].set(xlabel="time (s)", ylabel="Hz", title=f"{ns} seeds: channel AND fleet randomised")
    ax[1].hist(nads, bins=8, color=OI["sky"]); ax[1].set(xlabel="nadir (Hz)", ylabel="runs",
        title=f"nadir {np.mean(nads):.4f} +/- {np.std(nads, ddof=1):.4f} Hz")
    print(f"\nMonte Carlo nadir: {np.mean(nads):.4f} +/- {np.std(nads, ddof=1):.4f} Hz "
          f"(min {min(nads):.4f}, max {max(nads):.4f})")
    save(fig, "fig10_montecarlo")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true"); a = ap.parse_args()
    p = Params(); t0 = time.time()
    analytic_report(p); tm = fig_margins(p); fig_delay_validation(p, tm)
    R = fig_schemes(p); fig_raster_trace(p, R); fig_pareto(p, a.fast); fig_fleet(p, R, a.fast)
    fig_robustness(p, a.fast); fig_contingency(p, a.fast); fig_montecarlo(p, a.fast)
    print(f"\nDone in {time.time()-t0:.0f} s. Figures in ./{OUT}/")

if __name__ == "__main__":
    main()
