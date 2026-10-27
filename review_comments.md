# Decision and Review Comments

## Manuscript Details
- **Title**: Sub-Cycle Power System Frequency and Inertia Support from Data Center UPS Arrays via Distributed Adaptive Event-Triggered Control
- **Reference**: Ms. No. EPSR-D-26-09098
- **Journal**: Electric Power Systems Research (EPSR)
- **Editor**: Rui Fan
- **Decision**: Reject / Not recommended for publication

---

## Editor & Reviewer Decision Letter

```text
[EXTERNAL EMAIL]
Ref.: Ms. No. EPSR-D-26-09098
Sub-Cycle Power System Frequency and Inertia Support from Data Center UPS Arrays via Distributed Adaptive Event-Triggered Control
Electric Power Systems Research

Dear Mr. Sajjadi Mohammadabadi,

Thank you for submitting your manuscript to Electric Power Systems Research. I regret to inform you that your manuscript is not recommended for publication.

For your information and guidance, any specific comments explaining why we have reached this decision and those received from reviewers, if available, are listed at the end of this letter.

If you want to find an alternative, you can use our Journal Finder to search for other journals that could be better suited for publishing your manuscript.

We appreciate your submitting your manuscript to this journal and for giving us the opportunity to consider your work.

Kind regards,
Rui Fan
Editor
Electric Power Systems Research
```

---

## Technical Comments from Reviewers / Editors

1. **Measurement & Estimation Realism:**
   - *Comment*: Eq. (3) supplies each agent with the exact simulated frequency and its derivative, subject only to a pure time delay of 1–8 ms; no estimation algorithm, filtering, or measurement noise is described.
   - *Practical Context*: In practice, P-class PMUs use roughly two-cycle windows for frequency, and RoCoF is typically evaluated over windows of hundreds of milliseconds.

2. **Gain vs. Actuator Saturation:**
   - *Comment*: With the stated gains ($K_d = 55\text{ MW}\cdot\text{s}$ against $P_{\max} = 0.06\text{ MW}$ per agent), each agent saturates as soon as its delayed measurement registers the disturbance.

3. **Figure 2 & Text Inconsistencies:**
   - *Comment*: The Fig. 2 caption, and the text at line 177, describe panel (c) as cumulative messages with an idle-period inset, but the rendered panel (c) shows battery capacity fade. The message-count comparison between event-triggered control and periodic polling, and the idle-period inset, are therefore not shown.

4. **Quantitative Inconsistencies (Nadir):**
   - *Comment*: The text states that the nadir stays within 1 mHz across the $\eta_0$ sweep, whereas Fig. 3(a) shows a range of about 3.5 mHz.

5. **Undefined Citations / Remnants:**
   - *Comment*: The manuscript refers to “Section 2.3 of the original numerical study”, but that is not identified.
   - *Comment*: A “0.02% target” and a “target 20 ms window” are referred to without being defined; these could be from another work.
