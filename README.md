#  Credit Card Fraud Detection : Enhanced ML Comparison

> An end-to-end machine learning benchmark for **credit card fraud detection** on a severely imbalanced dataset.
> Five classifiers are trained, cross-validated, and evaluated using rigorous fraud-specific metrics :
> PR-AUC, F1, MCC, and ROC-AUC, with exportable result tables.

---

##  Table of Contents

- [Project Overview](#-project-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Key Features](#-key-features)
- [Technologies Used](#-technologies-used)
- [How It Works](#-how-it-works)
- [Results & Visualisations](#-results--visualisations)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Performance Metrics](#-performance-metrics)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)

---

##  Project Overview

This project provides a rigorous, end-to-end comparison of five machine learning classifiers applied to the **ULB Credit Card Fraud Detection dataset** from Kaggle, one of the most widely used benchmarks for imbalanced classification.

A model that always predicts "legitimate" achieves **99.83% accuracy** while catching **zero frauds**. This project addresses that pitfall directly: every design decision, from evaluation metric selection to imbalance handling, is chosen to reflect real-world fraud detection requirements rather than inflated accuracy scores.

**Who is this for?**

| Audience | Use Case |
|---|---|
|  Financial institutions | Baseline benchmark for fraud classification pipelines |
|  Researchers & students | Study of imbalanced classification and proper evaluation |
|  Data scientists | Reusable template for multi-model comparison with fraud-aware metrics |
|  ML practitioners | Reference for class imbalance strategies without data leakage |

---

##  Pipeline Architecture

The project follows a linear, reproducible pipeline with five clearly separated stages:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Data       │    │  Exploratory │    │ Preprocessing│    │    Model     │    │  Evaluation  │
│   Loading    │ ─► │   Analysis   │ ─► │  & Splitting │ ─► │  Training &  │ ─► │  & Export    │
│              │    │              │    │              │    │     CV       │    │              │
│ • Load CSV   │    │ • Class dist │    │ • Deduplicate│    │ • 5 models   │    │ • 7 metrics  │
│ • Validate   │    │ • Amount EDA │    │ • Scale      │    │ • 3-fold CV  │    │ • 7 figures  │
│ • Overview   │    │ • Imbalance  │    │ • Stratified │    │ • Full train │    │ • CSV export │
│              │    │   analysis   │    │   70/30 split│    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

All design decisions, metric choice, imbalance handling, validation strategy are documented inline in the notebook with explicit justifications.

---

##  Key Features

- **Five-Model Benchmark** : Decision Tree, KNN, Linear SVM, Random Forest, and XGBoost are trained and evaluated on identical data splits for a fair, reproducible comparison.
- **Leakage-Free Imbalance Handling** : Class imbalance is corrected via `class_weight='balanced'` and `scale_pos_weight` rather than oversampling, eliminating any risk of data leakage into the test set.
- **Fraud-Appropriate Metrics** : PR-AUC is the primary metric; F1, MCC, Recall, and ROC-AUC are computed as secondary metrics, each capturing a distinct failure mode relevant to fraud.
- **Stratified Cross-Validation** : 3-fold stratified CV on a representative 30% training subsample provides honest generalisation estimates with controlled compute cost.
- **Publication-Ready Visualisations** : Seven figures are generated and saved at 130 DPI: class distribution, amount histograms, ROC curves, PR curves, metric comparison bar chart, confusion matrices, and XGBoost feature importances.
- **Exportable Results** : Test-set and cross-validation metrics are saved as CSV files for reproducibility and downstream reporting.
- **Optional SMOTE Block** : A commented-out SMOTE oversampling block is included in preprocessing for easy experimentation without modifying the core pipeline.

---

##  Technologies Used

| Category | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Machine Learning** | [Scikit-learn](https://scikit-learn.org/) — Decision Tree, KNN, LinearSVC, Random Forest |
| **Gradient Boosting** | [XGBoost](https://xgboost.readthedocs.io/) — `XGBClassifier` with `scale_pos_weight` |
| **Data Processing** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Visualisation** | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |
| **Environment** | [Jupyter Notebook](https://jupyter.org/) / Google Colab |

---

##  How It Works

The pipeline is structured as nine numbered notebook sections, executed top-to-bottom:

### Step 1 — Data Loading & Overview
- Reads `creditcard.csv` into a Pandas DataFrame
- Reports total transactions, fraud count and ratio, amount range, duplicates, and missing values
- Provides a concise sanity check before any transformation is applied

### Step 2 — Exploratory Data Analysis
- Plots the class distribution on a log scale to expose the severity of the imbalance
- Renders transaction amount histograms with median annotations, split by class (Normal vs. Fraud)
- Demonstrates why accuracy alone is a misleading metric in this setting

### Step 3 — Preprocessing
- Drops duplicate rows and the `Time` column (low signal for most classifiers)
- Applies `StandardScaler` to the `Amount` feature only
- Performs a **stratified 70/30 train-test split**, preserving the fraud prevalence in both sets

### Step 4 — Model Definitions
Five classifiers are instantiated with explicit imbalance correction:

| Model | Imbalance Strategy | Notes |
|---|---|---|
| Decision Tree | `class_weight='balanced'` | Unconstrained depth, Gini criterion |
| KNN | None (baseline) | k=5, parallel execution |
| SVM (Linear) | `class_weight='balanced'` | `LinearSVC` + `CalibratedClassifierCV` for probability output |
| Random Forest | `class_weight='balanced'` | 200 estimators, max depth 4 |
| XGBoost | `scale_pos_weight = neg/pos` | 200 estimators, max depth 4 |

### Step 5 — Stratified Cross-Validation
- Runs 3-fold stratified CV on a 30% stratified subsample of the training set
- Scores: F1, ROC-AUC, PR-AUC, Recall, Precision — reported as mean ± std per model

### Step 6 — Full Training & Test Evaluation
- Each model is fit on the complete training set
- Seven metrics are computed on the held-out test set: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC
- A styled summary table highlights the best and worst value per metric in green and red

### Step 7 — Classification Reports
- Per-class precision, recall, and F1 printed for every model at 4-digit precision

### Step 8 — Visualisations
- Seven figures generated and saved as PNG files (see [Results & Visualisations](#-results--visualisations))

### Step 9 — Export
- `results_test_metrics.csv` and `results_cv_metrics.csv` saved to the working directory

---

##  Results & Visualisations

Seven publication-ready figures are generated automatically during the run:

| Figure | Description |
|---|---|
| `fig1_class_distribution.png` | Bar chart of Normal vs. Fraud transaction counts (log scale) |
| `fig2_amount_distribution.png` | Amount histograms with median lines, split by class |
| `fig3_roc_curves.png` | ROC curves for all five models with AUC scores in the legend |
| `fig4_pr_curves.png` | Precision-Recall curves with baseline prevalence reference line |
| `fig5_metrics_comparison.png` | Grouped bar chart comparing all key metrics across models |
| `fig6_confusion_matrices.png` | Normalised confusion matrices side-by-side for all five models |
| `fig7_feature_importance.png` | XGBoost top-15 feature importances by mean gain |

> **Why PR curves over ROC?** With only 0.172% fraudulent transactions, ROC curves are overly optimistic. PR curves directly reflect model performance on the minority class and are the recommended diagnostic for severe class imbalance.

---

##  Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Dataset

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and place `creditcard.csv` in the project root (or update the `PATH` variable at the top of the notebook).

### Run the Notebook

```bash
jupyter notebook Credit_Card_Fraud_Detection_Enhanced_ML_Comparison.ipynb
```

Execute all cells top-to-bottom. Results tables, charts, and CSV exports will be generated automatically in the working directory.

### Run on Google Colab

Upload the notebook to [Google Colab](https://colab.research.google.com/), upload `creditcard.csv` to the Colab file system, and update the `PATH` variable accordingly. All dependencies are pre-installed in the Colab environment.

### Optional — Enable SMOTE Oversampling

An optional SMOTE block is included in the preprocessing section (commented out). To activate it:

```bash
pip install imbalanced-learn
```

Then uncomment the SMOTE block in Section 3 of the notebook.

---

##  Project Structure

```
├── Credit_Card_Fraud_Detection_Enhanced_ML_Comparison.ipynb   # Full pipeline notebook
├── creditcard.csv                                              # Dataset (download from Kaggle)
├── results_test_metrics.csv                                    # Per-model test set scores (generated)
├── results_cv_metrics.csv                                      # Cross-validation scores (generated)
├── fig1_class_distribution.png                                 # Generated figure
├── fig2_amount_distribution.png                                # Generated figure
├── fig3_roc_curves.png                                         # Generated figure
├── fig4_pr_curves.png                                          # Generated figure
├── fig5_metrics_comparison.png                                 # Generated figure
├── fig6_confusion_matrices.png                                 # Generated figure
├── fig7_feature_importance.png                                 # Generated figure
└── README.md
```

---

##  Dataset

This project uses the publicly available **[ULB Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**:

| Property | Value |
|---|---|
| **Total Transactions** | 284,807 |
| **Fraud Cases** | 492 (~0.172%) |
| **Features** | 28 PCA-transformed (`V1`–`V28`) + `Time`, `Amount`, `Class` |
| **Source** | European cardholders, September 2013 |
| **Challenge** | Severely imbalanced — 578 legitimate transactions for every fraud |

---

##  Performance Metrics

Seven metrics are computed per model on the held-out test set:

| Metric | What It Measures | Role in This Project |
|---|---|---|
| **Accuracy** | Overall correct predictions | Included for reference only — misleading here |
| **Precision** | Of predicted frauds, how many are real? | Controls false alarm rate |
| **Recall** | Of real frauds, how many are caught? | Most critical — missed fraud = financial loss |
| **F1** | Harmonic mean of Precision and Recall | Single balanced score |
| **ROC-AUC** | Ranking quality across all thresholds | General discrimination ability |
| **PR-AUC** | Area under the Precision-Recall curve | **Primary metric** — best for imbalanced data |
| **MCC** | Matthews Correlation Coefficient | Balanced even with extreme class sizes |

---

##  Future Work

Future improvements include hyperparameter tuning with Optuna, additional classifiers such as LightGBM and Isolation Forest, and leakage-free SMOTE integration for better imbalance handling. 

Explainability could be enhanced with SHAP-based feature attribution, while threshold tuning and cost-sensitive learning would make evaluation more realistic. 

Longer term, a FastAPI scoring endpoint and a Streamlit dashboard would move the project closer to production readiness.

---

##  Contributing

Contributions are welcome. Feel free to open an issue or submit a pull request for:

- Additional classifiers (LightGBM, CatBoost, Isolation Forest)
- SMOTE or hybrid resampling strategies
- Threshold tuning and cost-sensitive evaluation
- Interactive visualisation with Plotly or a Streamlit dashboard

---

##  License

> This project is released for **educational and research purposes**.

## Note from me :)

> This project was developed as part of an applied portfolio effort in machine learning and financial data science,
combining rigorous evaluation methodology with practical imbalanced classification techniques.
Contributions, suggestions, and feedback are welcome, feel free to explore!
