# Target Journal Recommendations for Resubmission

**Paper Title:** *Sub-Cycle Power System Frequency and Inertia Support from Data Center UPS Arrays via Distributed Adaptive Event-Triggered Control*

---

## 1. Selection Criteria
1. **Cost:** 100% Free / No APC (via traditional **Subscription model**).
2. **Acceptance Probability & Scope Alignment:** A control-theory, sustainable energy/computing, or applied electrical power journal where the event-triggered control scheme and data-center microgrid integration fit natively.
3. **Speed to First Decision:** Rapid feedback to maintain publication momentum.

---

## 2. Top Recommended Free (Subscription) Journals

| Rank | Journal | Publisher | Impact Factor / CiteScore | Cost (Subscription) | 1st Decision | Scope & Fit Analysis | Acceptance Probability |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1** | **Sustainable Energy, Grids and Networks (SEGAN)** | Elsevier | IF: **5.7** / CS: **9.1** | **Free ($0)** | **6 days** | **Perfect Fit:** Bridges data center computing infrastructures and smart grid frequency/inertia support. Very responsive editorial board. | **High** |
| **#2** | **Sustainable Computing: Informatics and Systems (SUSCOM)** | Elsevier | IF: **6.2** / CS: **13.2** | **Free ($0)** | **1 day** | **Strong Fit:** Specifically covers data center energy management, UPS batteries, green computing, and cyber-physical electrical systems. | **High** |
| **#3** | **Control Engineering Practice (CEP)** | Elsevier | IF: **5.3** / CS: **9.1** | **Free ($0)** | **2 days** | **Strong Control Focus:** Since the core contribution is an *adaptive event-triggered distributed control* algorithm, CEP focuses heavily on applied control engineering in industrial power systems. | **High** |
| **#4** | **Computers & Electrical Engineering (CAEE)** | Elsevier | IF: **5.5** / CS: **12.2** | **Free ($0)** | **22 days** | **Broad Fit:** Well-suited for joint computer engineering (data center networks/triggering) and electrical power engineering topics. More accommodating acceptance threshold. | **Very High** |
| **#5** | **Journal of Systems Architecture (JSA)** | Elsevier | IF: **5.3** / CS: **9.1** | **Free ($0)** | **1 day** | **Fast alternative:** Covers distributed cyber-physical systems, real-time response, and embedded control. | **Moderate to High** |

---

## 3. Comparison of Publishing Options

| Journal Name | Subscription Option Available? | APC if Open Access | Turnaround (1st Decision) | Why this journal? |
|:---|:---:|:---:|:---:|:---|
| **Sustainable Energy, Grids and Networks** | **Yes ($0 charge)** | $2,780 | 6 days | Direct intersection of energy grids and computing loads. Highly recommended for this paper. |
| **Sustainable Computing (SUSCOM)** | **Yes ($0 charge)** | $3,460 | 1 day | Direct intersection of data center infrastructure & sustainability. |
| **Control Engineering Practice** | **Yes ($0 charge)** | $3,070 | 2 days | Solid choice if addressing reviewer comments regarding PMU window estimation and RoCoF delays. |
| **Computers & Electrical Engineering** | **Yes ($0 charge)** | $3,270 | 22 days | Highest likelihood of acceptance among general electrical/computer engineering journals. |

---

## 4. Key Revisions Needed Before Resubmission

Regardless of which journal you submit to, fixing the EPSR reviewer concerns is crucial:
1. **Realistic PMU/Estimation Filtering**: Clarify that frequency/RoCoF isn't instantaneous; model or mention a ~2-cycle PMU frequency estimation filter and RoCoF smoothing.
2. **Gain Saturation ($K_d = 55$ vs $P_{\max} = 0.06$ MW)**: Justify or normalize the gain, or add anti-windup/saturation dynamics.
3. **Fix Fig. 2(c)**: Ensure panel (c) shows the cumulative message comparison and idle-period inset instead of battery degradation.
4. **Clarify Text Numbers**: Ensure the text matches Fig. 3(a) (~3.5 mHz nadir variation).
5. **Clean Up Phantom References**: Remove or define "Section 2.3 of the original numerical study", "0.02% target", and "target 20 ms window".
