# XGBoost Model Training Report

## Model Information
- **Model Name**: XGBoost_20251022_215204
- **Algorithm**: XGBoost
- **Training Date**: 20251022_215226
- **Platform**: Diabetes Risk Analysis Platform

## Model Performance Metrics
- **accuracy**: 0.8495
- **precision**: 0.8051
- **recall**: 0.8495
- **f1_score**: 0.8112
- **balanced_accuracy**: 0.3872
- **matthews_corrcoef**: 0.2555
- **roc_auc**: 0.8208

## Model Parameters
- **n_estimators**: 200
- **max_depth**: 6
- **learning_rate**: 0.1
- **random_state**: 42

## Usage Instructions

This model can be loaded using joblib:

```python
import joblib
model = joblib.load('XGBoost_20251022_215204')
predictions = model.predict(X_test)
```

## Model Validation
- Cross-validation performed with 5-fold CV
- Training data: BRFSS 2015 Dataset
- Features: 21 health indicators
- Target: Diabetes risk levels (0, 1, 2)

Generated automatically by Diabetes Risk Analysis Platform
