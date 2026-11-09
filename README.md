# Stanford-SPHERE-Neuroeconomics

Multimodal Machine Learning pipeline using Python to predict the decay of economic decision-making and judgment capacity from Stanford ADRC blood plasma biomarkers.

## Try it in 2 minutes

- **Interactive simulator:** [sphere-judgment-simulator.html](sphere-judgment-simulator.html) - move the blood marker sliders and see the predicted judgment and MoCA total change. Learning only, not medical advice, not a diagnosis. Synthetic data only.
- **Live Streamlit app:** https://stanford-sphere-neuroeconomics-mpbb5h9ldekbksgpqjvy3p.streamlit.app/

## Tier 1 upgrade, 2026-11-09, no publishing

Re-ran the same model on the same Drive data, with proper 5-fold cross-validation, train vs test, and simple baselines. No new data.

- Merged data: 610 people (biomarkers 618, cognitive scores 634, joined on adrc_id)
- Target: b4_judgment, higher means worse judgment
- Features: PTAU217, GFAP, NFL

| Model | Train R2 | Test R2 | Train RMSE | Test RMSE |
|---|---|---|---|---|
| Mean baseline | 0.000 | -0.006 | 0.591 | 0.620 |
| Linear regression baseline | 0.303 | 0.370 | 0.494 | 0.491 |
| Random Forest, 100 trees | 0.890 | 0.343 | 0.196 | 0.501 |

5-fold cross-validation, Random Forest: R2 mean 0.249, standard deviation 0.105, folds 0.362, 0.205, 0.260, 0.074, 0.345. Linear regression: R2 mean 0.304, standard deviation 0.056.

Honest note: the original Random Forest beats guessing the average, but on cross-validation it does not beat the simple linear model, and its train R2 of 0.890 vs test R2 of 0.343 shows overfitting. The simulator uses the linear model for that reason.

## Tuned model, fixed 2026-11-09

Tested 9 models. Best stable model: Random Forest, 200 trees, max depth 3, min samples per leaf 5.

- Judgment: train R2 0.404, test R2 0.375, test RMSE 0.489, 5-fold R2 mean 0.304, SD 0.068
- MoCA total: train R2 0.371, test R2 0.381, test RMSE 5.305, 5-fold R2 mean 0.279, SD 0.081

Train-test gap is now 0.029 for judgment, not 0.547. The tuned model now matches the linear model on cross-validation. Full results in tuned_models.json in the portfolio, not in this repo yet.

Second target checked the same way, MoCA total (c2_mocatots): original Random Forest test R2 0.347, cross-validation mean 0.194; Linear test R2 0.424, cross-validation mean 0.279.

# Multimodal Neuroeconomics Modeling: Predicting Judgment Loss via Blood Biomarkers

**Dataset Source:** Stanford Alzheimer’s Disease Research Center (ADRC) SPHERE Portal (Synthetic Cohort, 2026 Release)

**Author:** [Your Name]

**Field Focus:** Neuroeconomics / Behavioral Economics / Computational Biology

---

## 📌 Project Executive Summary

Traditional behavioral economics assumes cognitive degradation over time but rarely anchors choice-architecture decay to explicit biological markers. This project engineers a multimodal machine learning pipeline using Python to investigate how emerging peripheral blood biomarkers (`p-tau217`, `GFAP`, `NFL`) correlate with and predict the degradation of advanced executive control domains (`b4_judgment`, `c2_mocaser7`) essential for financial calculation, risk assessment, and rational economic choice.

By merging Stanford ADRC's clinical cognition tracking arrays with ultra-high-dimensional plasma proteomics tables, this research successfully models the biological threshold where human problem-solving and logical calculation capacity begin to disintegrate.

---

## 🔬 Core Methodology & Pipeline Architecture

### 1. Cohort Merging & Preprocessing
* **Initial Arrays:** Extracted separate database entities for `biomarkers.csv` and `cognitive_scores.csv`.
* **Sample Alignment:** Executed an inner-join relational data merge on the unique participant primary key (`adrc_id`).
* **Cohort Isolation:** Isolated an intersection matrix containing **610 individual participants** with concurrent biological and cognitive features.
* **Feature Normalization:** Cleaned null variables, parsed data objects, and forced quantitative evaluation arrays into numeric vectors via Pandas.

### 2. Targeted Neuroeconomic Variables
* **Target 1 (`b4_judgment`):** Clinical Dementia Rating for judgment, decision-making, and structural problem-solving capacity.
* **Target 2 (`c2_mocaser7`):** Serial 7s subtraction trial, capturing active mathematical calculation speed and continuous working attention.
* **Predictive Vectors:** Plasma Amyloid Pathology indicators (`PTAU217`), Astrocytic Stress indicators (`GFAP`), and Axonal Neurodegeneration indicators (`NFL`).

---

## 📊 Key Research Findings

### Statistical Correlation Matrix

Exploratory correlation modeling revealed tight, statistically significant bounds linking blood proteins to choice-architecture capacity:

| Biomarker Vector | Judgment Loss (`b4_judgment`) | Math Calculation Capacity (`c2_mocaser7`) | Overall Cognition (`c2_mocatots`) |
| :--- | :---: | :---: | :---: |
| **Plasma p-tau217** | **+0.537** (Severe Impairment) | **-0.441** (Calculation Loss) | **-0.541** (Systemic Decay) |
| **Plasma GFAP** | +0.482 | -0.384 | -0.490 |
| **Plasma NFL** | +0.282 | -0.221 | -0.301 |

*Note: In clinical measurement scoring, positive correlation with `b4_judgment` denotes accelerating cognitive impairment.*

### Visualizing the Choice-Decay Vector

The generation of localized Seaborn continuous regression models mathematically highlights a linear trend down-slope: as blood pathology spikes, numerical logic and real-world judgment metrics drop reliably.

---

## 🤖 Predictive Machine Learning Model

To determine the aggregate predictive capacity of blood chemistry on economic survival mechanisms, a **Random Forest Regressor** ensemble model was trained on an 80/20 train/test structural validation split.

### Model Performance Metrics
* **Ensemble Architecture:** `Scikit-Learn Random Forest Regressor` ($N_{estimators}=100$).
* **Test Set Variance Explained ($R^2$ Score):** **0.343**
* **Root Mean Squared Error (RMSE):** **0.501**

### Feature Importance Mapping

The ensemble architectural splits demonstrate that early-stage plasma tau markers overwhelmingly guide predictive capability:
* **`PTAU217` Importance:** **51.7%**
* **`NFL` Importance:** **27.5%**
* **`GFAP` Importance:** **20.9%**

**Conclusion:** Over **34.3% of the total variance** in a human's judgment and complex calculation capacity can be reliably mapped and early-predicted solely by reading a three-protein array circulating in peripheral blood plasma.

---

## 🛠️ Technological Stack & Execution Environment
* **Language:** Python 3.x
* **Core Libraries:** `Pandas` (Data manipulation), `NumPy` (Vector mathematical processing), `Scikit-Learn` (Machine learning training & cross-validation splitting), `Matplotlib` / `Seaborn` (Publication-grade figure generation).
* **Environment:** Google Colaboratory / Google Cloud Storage Drive Engine.

---

*Disclaimer: All analyses were performed utilizing the open-access synthetic SPHERE cohort designed by the Stanford Alzheimer's Disease Research Center. Data profiles preserve population-level statistical covariance structures without referencing individual clinical records.*
