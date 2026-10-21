# MKU_main
Target repository

## Model training experiments

Run the parameter experiments with:

```bash
python -m model_training.train_experiments
```

The script uses the same Seaborn `tips` data and preprocessing pipeline as
`starter/baseline_model.py`. It compares linear regression with different
intercept and positivity settings, printing each experiment's settings,
learned coefficients, and test metrics.
