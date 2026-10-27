# PULSE E2E Testing Infrastructure (`TEST_INFRA.md`)

## 1. Overview & Architecture

This document defines the End-to-End (E2E) Testing Architecture for the PULSE framework and its supporting simulation and manuscript assets, targeted for Elsevier's *Sustainable Energy, Grids and Networks* (SEGAN).

The test suite provides an independent, opaque-box, requirement-driven verification harness implemented in `test_pulse_e2e.py`. It rigorously evaluates all 20 features from the `PROJECT.md` Feature Inventory across a 4-tier verification hierarchy.

```
+-----------------------------------------------------------------------------------+
|                        PULSE E2E Verification Hierarchy                            |
+-----------------------------------------------------------------------------------+
| Tier 4: Real-World Application Scenarios (Substation 8% trip, 30s, lossy network) |
+-----------------------------------------------------------------------------------+
| Tier 3: Cross-Feature Pairwise Interactions (Trigger x Loss, Reserve x Gain, etc.)|
+-----------------------------------------------------------------------------------+
| Tier 2: Boundary & Corner Cases (Limits: 0-15% trips, 0-30% drop, SoC 70-100%)    |
+-----------------------------------------------------------------------------------+
| Tier 1: Feature Coverage (>=5 test cases per feature across Features 1 to 20)     |
+-----------------------------------------------------------------------------------+
```

---

## 2. 4-Tier Verification Methodology

### Tier 1: Feature Coverage
Validates the functional and nominal specifications of every feature in the `PROJECT.md` Feature Inventory (Features 1 to 20), with at least 5 distinct test cases per feature:
- **Feature 1: Simulation Horizon 30s** — Verifies governor restoration settling time, numerical integration stability, trajectory array dimensions, and energy accumulation over 30s.
- **Feature 2: Realistic Measurement Filtering** — Verifies 1-cycle PLL (16.7 ms filter, 33.3 ms RoCoF, $t_{90} \approx 55.7\,\mathrm{ms}$), 2-cycle PMU (33.3 ms filter, 100 ms RoCoF, $t_{90} \approx 197.6\,\mathrm{ms}$), and idealized benchmark (7.4 ms).
- **Feature 3: Virtual Inertia & Damping Derivation** — Validates analytic formulas $H_v = \frac{\sum K_d}{2 S_{\mathrm{base}}} = 1.333\,\mathrm{s}$ (+41.7%), $D_v = \frac{\sum K_p}{S_{\mathrm{base}}} = 6.0\,\mathrm{pu}$ (+600%), and initial RoCoF reduction from $-0.469\,\mathrm{Hz/s}$ to $-0.331\,\mathrm{Hz/s}$.
- **Feature 4: Dual-Window RoCoF** — Evaluates RoCoF sliding window formulations over 100 ms (inertial/FFR window) and 500 ms (equipment withstand window).
- **Feature 5: Non-Saturating Gain Regime** — Verifies linear proportional-derivative regime ($K_d = 4.0\,\mathrm{MW\cdot s}$, utilizing 72.8–74.4% of inverter capacity) versus saturated relay mode ($K_d = 55.0\,\mathrm{MW\cdot s}$).
- **Feature 6: Dynamic Battery Energy/Runtime Model** — Verifies 30 kWh pack, 2.0C discharge rate (60 kW peak), float state maintenance, reserve minutes tracking ($t_{\mathrm{runtime}} \ge 25\,\mathrm{min}$ at 80% reserve floor), and 30s energy consumption ($\le 0.50\,\mathrm{kWh}$ or 1.67% capacity).
- **Feature 7: Industry Local Droop/PD FFR Baseline** — Verifies continuous local droop/PD controller with deadband ($\pm 15\,\mathrm{mHz}$) without communication dependency.
- **Feature 8: Fast Periodic Polling Baseline** — Verifies multi-rate periodic polling intervals (10 ms, 20 ms, 50 ms, 100 ms).
- **Feature 9: Explicit No-Support Contingency Sweeps** — Verifies explicit no-support baseline simulation across 3%, 5%, and 8% generation trip sweeps.
- **Feature 10: Ambient Drift & Message Statistics** — Verifies Ornstein-Uhlenbeck ambient frequency drift ($\sigma = 0.012\,\mathrm{Hz/\sqrt{s}}, \alpha = 0.1\,\mathrm{s}^{-1}$), zero spurious triggers with deadband, and peak msg/s metrics.
- **Feature 11: Nadir vs Messages Pareto Front** — Validates Pareto trade-off between frequency nadir and message counts comparing adaptive PULSE vs fixed-threshold event-triggered control (FETC).
- **Feature 12: Factorial Experiment & Delay Margins** — Verifies $3 \times 3 \times 3$ factorial design matrix and analytical delay margin $\tau_{\max} = \mathrm{PM} / \omega_c \approx 952\,\mathrm{ms}$ ($K_d=4.0$) vs $34\,\mathrm{ms}$ ($K_d=55.0$).
- **Feature 13: Monte Carlo Robustness Campaign** — Validates multi-seed stochastic simulation consistency, normal/log-normal channel statistics, and 95% confidence intervals.
- **Feature 14: Publication Display Figures Polish** — Validates existence, 300 DPI resolution, non-zero file sizes, and metadata of Figures 0–5 and graphical abstract.
- **Feature 15: Abstract Condensation & Highlights** — Verifies abstract word count ($< 250$ words) and Elsevier SEGAN highlight constraints (3–5 bullets, each $\le 85$ characters).
- **Feature 16: Overclaims Pruning** — Systematically checks that forbidden overclaim phrases ("gigawatt-scale", "multi-tenant", "calibrated", "provably excludes Zeno", "centrally dispatched BESS", "peer-to-peer") are excised from the manuscript.
- **Feature 17: Theoretical Storytelling & Framing** — Validates proper contextual framing of 7.4 ms idealized benchmark, lead filtering (56–198 ms), and dwell-time refractory period.
- **Feature 18: Bibliography Expansion & Verification** — Verifies BibTeX database integrity, inclusion of IEEE C37.118, FFR standards, and SEGAN citations.
- **Feature 19: Dual-Track E2E Test Suite** — Self-testing and verification of test infrastructure, coverage, and reproducibility.
- **Feature 20: Clean LaTeX Build & Publication Polish** — Verifies LaTeX compilation prerequisites, document class `elsarticle.cls`, and absence of undefined citations.

### Tier 2: Boundary & Corner Cases
Tests mathematical and physical limits:
- Extreme generation-loss contingencies: $0\%$ (null event), $3\%$ (mild), $5\%$ (design basis), $8\%$ (severe), $15\%$ (extreme stress).
- Latency extremes: $0\,\mathrm{ms}$ (instantaneous), $1\text{--}8\,\mathrm{ms}$ (nominal star-tree Ethernet), $50\,\mathrm{ms}$ (bufferbloat spike), $100\,\mathrm{ms}$ (severe WAN delay).
- Packet drop extremes: $0\%$ (ideal), $5\%$, $20\%$, $30\%$ (lossy wireless/congested), $100\%$ (complete communication blackout / failsafe).
- Battery SoC boundaries: $100\%$ (full charge), $95\%$, $80\%$ (reserve floor), $70\%$ (depleted reserve), $0\%$ (dead battery protection).
- Deadband boundaries: $0\,\mathrm{mHz}$ (no deadband / noisy triggers), $\pm 15\,\mathrm{mHz}$ (PULSE nominal), $\pm 36\,\mathrm{mHz}$ (NERC standard).
- Dwell time refractory limits: $\delta = 0$ (unguarded), $\delta = 1.5\,\mathrm{ms}$ (nominal anti-Zeno), $\delta = 10\,\mathrm{ms}$ (chatter suppression).
- Gain parameter extremes: $K_d = 0$ (pure droop), $K_d = 4.0$ (linear regime), $K_d = 55.0$ (saturated relay), $K_d \to \infty$ (infinite gain clamping).

### Tier 3: Cross-Feature Pairwise Interactions
Tests coupled subsystems and joint parameter sweeps:
- Adaptive Triggering $\times$ Packet Loss ($0\%$, $10\%$, $20\%$, $30\%$).
- Battery Reserve Floor ($80\%$) $\times$ High Derivative Gain ($K_d=55.0$).
- PMU/PLL Measurement Filtering (1-cycle, 2-cycle) $\times$ Contingency Size ($3\%$, $5\%$, $8\%$).
- Ambient Grid Drift (Ornstein-Uhlenbeck) $\times$ Frequency Deadband ($\pm 15\,\mathrm{mHz}$).
- Fast Periodic Polling ($10\,\mathrm{ms}$, $20\,\mathrm{ms}$) $\times$ Packet Loss ($10\%$, $20\%$).
- Non-Saturating Gain ($K_d=4.0$) $\times$ Log-Normal Latency Jitter.
- Dual-Window RoCoF ($100\,\mathrm{ms}$ vs $500\,\mathrm{ms}$) $\times$ Measurement Noise Bandwidth.
- Fleet Aggregation Scale ($N=50, 100, 200$) $\times$ Virtual Inertia $H_v$.

### Tier 4: Real-World Application & Mission Scenarios
Tests end-to-end mission-level operational scenarios:
- **Scenario 1**: Distribution substation under severe $8\%$ generation trip with $20\%$ packet loss and $80\%$ battery reserve floor over full $30\,\mathrm{s}$ horizon.
- **Scenario 2**: Peak communication burst and channel utilization under combined ambient frequency drift and sudden $5\%$ contingency.
- **Scenario 3**: Full 4-way comparative performance across $30\,\mathrm{s}$ horizon: PULSE (adaptive event-triggered) vs FETC (fixed event-triggered) vs Fast Periodic Polling ($10\,\mathrm{ms}$) vs No Support.
- **Scenario 4**: Cascading contingency resilience: initial $5\%$ generation trip followed by secondary $3\%$ trip under lossy channel.
- **Scenario 5**: Battery runtime safety verification: confirming that post-contingency IT backup runtime remains above 25 minutes after full 30s grid support.

---

## 3. Test Traceability Matrix

| Feature # | Feature Name | Test Class | Test Methods |
|---|---|---|---|
| F1 | Simulation Horizon 30s | `TestTier1FeatureCoverage` | `test_f01_horizon_duration`, `test_f01_numerical_stability`, `test_f01_governor_recovery_settling`, `test_f01_energy_accumulation_30s`, `test_f01_time_array_consistency` |
| F2 | Realistic Measurement Filtering | `TestTier1FeatureCoverage` | `test_f02_pll_1cycle_delay`, `test_f02_pmu_2cycle_delay`, `test_f02_idealized_benchmark_7_4ms`, `test_f02_noise_filtering_variance`, `test_f02_filter_time_constants` |
| F3 | Virtual Inertia & Damping | `TestTier1FeatureCoverage` | `test_f03_hv_analytic_formula`, `test_f03_dv_analytic_formula`, `test_f03_rocof_reduction_percentage`, `test_f03_heff_total_inertia`, `test_f03_linear_regime_validity` |
| F4 | Dual-Window RoCoF | `TestTier1FeatureCoverage` | `test_f04_rocof_100ms_window`, `test_f04_rocof_500ms_window`, `test_f04_smoothing_filter_kernel`, `test_f04_rocof_ordering_property`, `test_f04_discretization_robustness` |
| F5 | Non-Saturating Gain Regime | `TestTier1FeatureCoverage` | `test_f05_linear_kd4_capacity_utilization`, `test_f05_saturated_kd55_relay_mode`, `test_f05_saturation_threshold_derivation`, `test_f05_chattering_prevention`, `test_f05_monotonic_gain_response` |
| F6 | Dynamic Battery Model | `TestTier1FeatureCoverage` | `test_f06_battery_pack_capacity_kwh`, `test_f06_crate_limit_enforcement`, `test_f06_reserve_minutes_calculation`, `test_f06_30s_energy_throughput`, `test_f06_soc_depletion_integration` |
| F7 | Industry Droop/PD Baseline | `TestTier1FeatureCoverage` | `test_f07_continuous_droop_control`, `test_f07_deadband_suppression`, `test_f07_ffr_response_profile`, `test_f07_uncoordinated_injection`, `test_f07_governor_coordination` |
| F8 | Fast Periodic Polling | `TestTier1FeatureCoverage` | `test_f08_polling_10ms_response`, `test_f08_polling_20ms_response`, `test_f08_polling_50ms_response`, `test_f08_polling_100ms_response`, `test_f08_message_rate_scaling` |
| F9 | Explicit No-Support Sweeps | `TestTier1FeatureCoverage` | `test_f09_nosupport_3pct_nadir`, `test_f09_nosupport_5pct_nadir`, `test_f09_nosupport_8pct_nadir`, `test_f09_nadir_monotonicity_loss`, `test_f09_governor_recovery_nadir` |
| F10 | Ambient Drift & Statistics | `TestTier1FeatureCoverage` | `test_f10_ornstein_uhlenbeck_parameters`, `test_f10_deadband_zero_false_triggers`, `test_f10_ambient_variance_stationarity`, `test_f10_peak_msg_rate_calculation`, `test_f10_idle_messaging_elimination` |
| F11 | Pareto Front (Nadir vs Msgs)| `TestTier1FeatureCoverage` | `test_f11_adaptive_vs_fetc_tradeoff`, `test_f11_eta_sweep_monotonicity`, `test_f11_message_reduction_percentage`, `test_f11_nadir_preservation`, `test_f11_pareto_dominance_check` |
| F12 | Factorial Grid & Delays | `TestTier1FeatureCoverage` | `test_f12_factorial_design_dimension`, `test_f12_delay_margin_kd4_analytic`, `test_f12_delay_margin_kd55_analytic`, `test_f12_noise_bandwidth_stability`, `test_f12_phase_margin_crossover` |
| F13 | Monte Carlo Robustness | `TestTier1FeatureCoverage` | `test_f13_monte_carlo_seed_reproducibility`, `test_f13_confidence_interval_95pct`, `test_f13_lognormal_jitter_distribution`, `test_f13_bernoulli_packet_drop`, `test_f13_statistical_significance` |
| F14 | Publication Display Figures | `TestTier1FeatureCoverage` | `test_f14_figure_files_exist`, `test_f14_figure_dpi_and_vector_format`, `test_f14_graphical_abstract_exists`, `test_f14_figure_file_sizes_nonempty`, `test_f14_aspect_ratios_standard` |
| F15 | Abstract & Highlights | `TestTier1FeatureCoverage` | `test_f15_abstract_word_count_limit`, `test_f15_highlights_bullet_count`, `test_f15_highlights_char_limit_85`, `test_f15_highlights_file_exists`, `test_f15_abstract_keywords_presence` |
| F16 | Overclaims Pruning | `TestTier1FeatureCoverage` | `test_f16_excise_gw_scale_claim`, `test_f16_excise_multi_tenant_claim`, `test_f16_excise_calibrated_claim`, `test_f16_excise_zeno_proof_claim`, `test_f16_excise_central_dispatch_claim` |
| F17 | Storytelling & Framing | `TestTier1FeatureCoverage` | `test_f17_framing_7_4ms_idealized`, `test_f17_lead_realistic_filtering`, `test_f17_dwell_time_anti_zeno_design`, `test_f17_tier_iii_iv_dc_framing`, `test_f17_sfr_low_order_validity` |
| F18 | Bibliography Integrity | `TestTier1FeatureCoverage` | `test_f18_bibtex_syntax_validity`, `test_f18_ieee_c37_118_citation`, `test_f18_ffr_standards_citations`, `test_f18_segan_relevant_citations`, `test_f18_no_broken_cite_keys` |
| F19 | E2E Testing Suite Track | `TestTier1FeatureCoverage` | `test_f19_test_harness_standalone_executable`, `test_f19_test_infra_documentation_exists`, `test_f19_results_summary_format_validity`, `test_f19_tier_structure_completeness`, `test_f19_progressive_testability_clean` |
| F20 | Clean LaTeX Build | `TestTier1FeatureCoverage` | `test_f20_latex_source_exists`, `test_f20_elsarticle_cls_present`, `test_f20_bibtex_bst_files_present`, `test_f20_zero_unresolved_questionmarks`, `test_f20_manuscript_clean_syntax` |
| T2 | Boundary & Corner Cases | `TestTier2BoundaryCases` | 7 sub-domains $\times$ 5 tests = 35 test cases |
| T3 | Cross-Feature Interactions | `TestTier3CrossFeature` | 10 pairwise cross-feature tests |
| T4 | Real-World Application | `TestTier4ApplicationScenarios`| 5 mission-scale operational scenarios |

---

## 4. Authoritative Verification Oracles

Expected outputs and physical tolerances are strictly derived from:
1. **Anderson-Fouad Low-Order System Frequency Response (SFR)**:
   - Synchronous machine equation: $2H \frac{d\Delta f}{dt} = \Delta P_m - \Delta P_L - D \Delta f + P_{\mathrm{bess}} / S_{\mathrm{base}}$
   - Governor lag: $T_g \frac{d\Delta P_m}{dt} = -\Delta P_m - \Delta f / R$
   - Parameter values: $S_{\mathrm{base}} = 300\,\mathrm{MVA}, H = 3.2\,\mathrm{s}, D = 1.0\,\mathrm{pu}, R = 0.05\,\mathrm{pu}, T_g = 6.0\,\mathrm{s}, f_0 = 60.0\,\mathrm{Hz}$.
2. **Virtual Inertia Derivation**:
   - $H_v = \frac{\sum_i K_{d,i}}{2 S_{\mathrm{base}}} = \frac{200 \times 4.0}{2 \times 300} = 1.333\,\mathrm{s} \implies H_{\mathrm{eff}} = 4.533\,\mathrm{s}$ (+41.7%).
   - $D_v = \frac{\sum_i K_{p,i}}{S_{\mathrm{base}}} = \frac{200 \times 9.0}{300} = 6.0\,\mathrm{pu} \implies D_{\mathrm{eff}} = 7.0\,\mathrm{pu}$ (+600%).
3. **Analytical Frequency Nadir & RoCoF Bounds**:
   - Baseline RoCoF at $t=0^+$ for 5% trip (15 MW): $\operatorname{RoCoF}(0^+) = -\frac{\Delta P_L}{2 H} f_0 = -\frac{0.05}{6.4} \times 60 = -0.46875\,\mathrm{Hz/s}$.
   - Virtual Inertia RoCoF at $t=0^+$: $\operatorname{RoCoF}_{H_v}(0^+) = -\frac{0.05}{2 \times 4.5333} \times 60 = -0.33088\,\mathrm{Hz/s}$ (29.4% reduction).
   - Unassisted nadir for 5% trip: $59.423 \pm 0.005\,\mathrm{Hz}$.
   - Supported nadir: $59.734 \text{--} 59.750\,\mathrm{Hz}$ (arresting $> 0.31\,\mathrm{Hz}$ of decline).
4. **Frequency Measurement Filtering**:
   - 1-cycle PLL ($f_0 = 60\,\mathrm{Hz} \implies \tau_f = 16.7\,\mathrm{ms}, \tau_r = 33.3\,\mathrm{ms} \implies t_{90} \approx 55.7\,\mathrm{ms}$).
   - 2-cycle PMU ($\tau_f = 33.3\,\mathrm{ms}, \tau_r = 100\,\mathrm{ms} \implies t_{90} \approx 197.6\,\mathrm{ms}$).
5. **Battery IT Reserve Model**:
   - Pack capacity: $E_{\mathrm{nom}} = 30\,\mathrm{kWh}$, continuous $P_{\max} = 60\,\mathrm{kW}$ (2.0C rate).
   - Reserve floor: $\mathrm{SoC}_{\min} = 80\%$ ($24\,\mathrm{kWh}$ reserved for IT facility backup).
   - Reserve minutes: $t_{\mathrm{runtime}} = \frac{E(t)}{P_{\mathrm{IT}}} \times 60 \approx 27.4\,\mathrm{min}$.
   - 30s contingency energy: $\Delta E = 60\,\mathrm{kW} \times \frac{30}{3600}\,\mathrm{h} = 0.50\,\mathrm{kWh}$ (1.67% of rated capacity).
6. **Communication Network Constraints**:
   - Application payload: 30 bytes (frequency, RoCoF, status, sequence number).
   - Wire frame size: 72 bytes (UDP/IP/Ethernet header).
   - Ornstein-Uhlenbeck drift: $\sigma = 0.012\,\mathrm{Hz/\sqrt{s}}, \alpha = 0.1\,\mathrm{s}^{-1}$. Deadband $\pm 15\,\mathrm{mHz}$ guarantees zero spurious triggers during nominal idle periods.

---

## 5. Execution & Verification Instructions

### Direct Execution
Execute via standard Python CLI:
```powershell
python test_pulse_e2e.py
```

### Pytest Execution
Execute with full verbose logging:
```powershell
pytest test_pulse_e2e.py -v
```

### Filter by Verification Tier
Run specific tiers as needed:
```powershell
pytest test_pulse_e2e.py -k "TestTier1" -v
pytest test_pulse_e2e.py -k "TestTier2" -v
pytest test_pulse_e2e.py -k "TestTier3" -v
pytest test_pulse_e2e.py -k "TestTier4" -v
```

---

## 6. Acceptance Criteria

The E2E test suite confirms compliance with Elsevier SEGAN publication standards when:
1. **Pass Rate**: 100% of all test cases across Tiers 1–4 execute and pass.
2. **Speed**: Entire suite executes in under 60 seconds without hanging or memory leakage.
3. **Integrity**: Zero facade tests; all physical, control, and manuscript assertions validate real mathematical invariants and filesystem artifacts.
4. **Artifact Cleanliness**: Zero temporary files or side-effect mutations left in the repository.
