# 🧪 End-to-End A/B Testing Framework
### Microsoft Copilot Onboarding Flow Experiment

> **A production-grade experimentation analysis** covering the full lifecycle of an A/B test — from hypothesis formulation and power analysis through frequentist testing, Bayesian inference, causal reasoning, and a final ship/no-ship decision framework.

---

## 📌 Business Context

**Product:** Microsoft Copilot (SaaS Productivity Tool)  
**Question:** Does a redesigned onboarding flow with contextual tooltips and a guided first-task experience increase 7-day feature activation?  
**Stakes:** A 2pp lift in activation across 10K daily new users = ~73,000 additional activated users/year

---

## 🎯 What This Project Covers

| Section | Concept | Tools |
|---|---|---|
| 1 | Experiment Design & Hypothesis Formulation | — |
| 2 | Power Analysis & Sample Size Calculation | `statsmodels`, Cohen's h |
| 3 | Realistic Data Simulation (noise, bots, missingness) | `numpy`, `pandas` |
| 4 | Exploratory Data Analysis (EDA) | `matplotlib`, `seaborn` |
| 5 | Sample Ratio Mismatch (SRM) Detection | Chi-square test |
| 6 | Frequentist Hypothesis Testing | Z-test, bootstrap CI |
| 7 | CUPED Variance Reduction | Regression adjustment |
| 8 | Bayesian A/B Testing | Beta-Binomial, expected loss |
| 9 | Causal Inference: Difference-in-Differences | DiD estimator |
| 10 | Heterogeneous Treatment Effects (HTE) | Segment subgroup analysis |
| 11 | Decision Framework & Ship Recommendation | — |

---

## 📊 Key Results

```
Control activation rate:    31.52%
Treatment activation rate:  34.15%
Absolute lift:              +2.63pp  (95% CI: 1.23pp – 4.03pp)
Relative lift:              +8.35%
p-value:                    0.0002  ✅ Statistically significant
P(Treatment > Control):     99.99%  (Bayesian)
```

**Decision: 🚀 SHIP TREATMENT**

---

## 🔬 Methods Deep Dive

### Power Analysis
Computed minimum sample size to detect a 2pp MDE at 80% power and α=0.05 using Cohen's h effect size for proportions. Generated power curves showing the tradeoff between MDE, sample size, and statistical power.

### Frequentist Testing
Two-proportion z-test with bootstrapped confidence intervals on the lift. Confirmed no Sample Ratio Mismatch (SRM) via chi-square test on group assignment — a critical check often skipped in naive analyses.

### CUPED Variance Reduction
Applied Controlled-experiment Using Pre-Experiment Data (CUPED) to reduce metric variance using users' prior engagement score as a covariate. This technique, developed at Microsoft Research, increases statistical sensitivity without collecting more data.

### Bayesian Testing
Modeled activation as a Beta-Binomial process. Used a weakly informative prior Beta(2,5) and updated to posteriors given observed data. Decision criterion: expected loss minimization — ship treatment when E[loss|ship trt] < E[loss|keep ctrl].

### Difference-in-Differences
Constructed a pre/post panel to isolate causal effect from natural time trends, validating the parallel trends assumption between groups in the pre-period.

### Heterogeneous Treatment Effects
Segmented analysis by user type revealed the treatment lifts most for **Education** (+3.30%) and **Consumer** (+3.60%) segments. SMB lift (1.81%) was directionally positive but not statistically significant — suggesting a targeted rollout strategy.

---

## 🗂️ Repository Structure

```
ab_testing_project/
│
├── ab_testing_analysis.py       # Main analysis script (11 sections)
│
├── 01_power_analysis.png        # Power curves & MDE tradeoff
├── 02_eda_dashboard.png         # EDA: activation, retention, session depth
├── 03_frequentist_testing.png   # Z-test results, bootstrap lift distribution
├── 04_bayesian_testing.png      # Posterior distributions, credible intervals
├── 05_diff_in_diff.png          # DiD visualization with counterfactual
├── 06_hte_analysis.png          # Forest plot of segment-level effects
│
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/eeshanigundi/ab-testing-framework.git
cd ab-testing-framework

# Install dependencies
pip install numpy pandas scipy matplotlib seaborn statsmodels

# Run full analysis
python ab_testing_analysis.py
```

---

## 💡 Key Learnings & Design Decisions

**Why CUPED?**  
Standard z-tests ignore pre-experiment user behavior. CUPED removes this noise, effectively giving us more statistical power without running the experiment longer — critical when experiment runtime is constrained by business timelines.

**Why both Frequentist AND Bayesian?**  
Frequentist testing answers "is this result unlikely under the null?" Bayesian testing answers "what should we believe, and what decision minimizes expected loss?" Both matter for different stakeholders — engineers want p-values, business partners want probabilities and loss estimates.

**Why SRM check first?**  
A sample ratio mismatch means your randomization is broken. Any downstream analysis on compromised assignment is invalid. This is one of the most common silent failures in production experimentation systems.

**Why segment HTE analysis?**  
Aggregate results can mask heterogeneity. An average 2.63pp lift hides a 3.60pp lift for Consumer users and only 1.81pp for SMB — a fact that should drive rollout sequencing and targeting strategy.

---

## 🛠️ Skills Demonstrated

`A/B Experiment Design` · `Power Analysis` · `Frequentist Statistics` · `Bayesian Inference` · `Causal Inference` · `CUPED` · `Difference-in-Differences` · `Heterogeneous Treatment Effects` · `Python` · `Statistical Storytelling` · `Product Decision Frameworks`

---

*Built as part of a portfolio demonstrating applied experimentation for product data science roles.*
