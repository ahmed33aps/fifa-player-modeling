# Validation

The repaired train.py entry point passed fresh execution. The original exploratory notebooks are retained and are not all reproduced end to end.

Measured local baseline output:

```json
{
  "rows": 19337,
  "split": "random 80/20 seed 42",
  "mae_value_millions": 0.19880302487702953,
  "rmse_value_millions": 1.5539546521547862,
  "r2": 0.9563399938469141,
  "tier_accuracy": 0.8151499482936918,
  "train_only_rating_cutoffs": [
    58.0,
    63.0,
    68.0
  ],
  "caution": "Tier excludes its defining rating and total stats from classifier inputs. No future-season generalization claim."
}

```

These are a single seeded holdout result, not future-performance guarantees.
