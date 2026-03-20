# ML for Turtle Age Prediction

Predicts turtle age (measured by ring count) from physical measurements using classical ML regression models. Based on a dataset of 4,177 turtles with features including length, diameter, height, weight variants, and sex.

## Problem

Counting rings on a turtle's shell cross-section is the ground-truth method for determining age — but it's destructive and time-consuming. This project asks: can physical measurements predict ring count accurately enough to serve as a non-invasive alternative?

## Approach

1. **EDA** — scatter plots, correlation analysis, outlier removal (4 samples with invalid height values removed)
2. **Gender analysis** — one-way ANOVA + pairwise t-tests confirmed sex has a statistically significant effect on ring count (infants have ~2–3 fewer rings than adults; p < 0.001)
3. **Feature engineering** — one-hot encoding for sex, polynomial feature expansion for non-linear models
4. **Model selection** — trained on 80% of data, validated on 10%, tested on 10%; stratified splits by ring count bins to preserve distribution
5. **Cross-validation** — 10-fold CV on best models to estimate generalization

## Models

| Model | Hyperparameters | MSE | MAE | R² |
|---|---|---|---|---|
| Linear Regression | — | 5.27 | 1.63 | 0.507 |
| Polynomial Regression | degree=2 | 4.66 | 1.51 | **0.564** |
| KNN | k=6 | 4.88 | 1.51 | 0.543 |

**Best model: Polynomial Regression (degree=2)** — R² of 0.564, MAE of ~1.5 rings on the held-out test set.

10-fold CV confirmed stability: mean MSE 4.45 ± 0.60, mean R² 0.569 ± 0.047.

## Key Findings

- Shell weight and height are the strongest individual predictors of ring count (r = 0.63 and 0.61)
- All physical features are highly correlated with each other (r > 0.84), but only moderately with ring count
- Polynomial degree=2 captures non-linear relationships without overfitting; higher degrees (≥4) overfit severely
- Prediction errors are larger for turtles at the extremes of the size distribution

## 📄 [Full PDF Report](./Report.pdf)
