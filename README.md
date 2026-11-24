# FIFA player modeling

Player-value regression and performance-tier classification, with preprocessing and model comparisons.

## Contents

- `note_book_2.ipynb`
- `Fifa.csv`
- `EDA_preprosecing.ipynb`
- `Handle_Script.py`

## Run

Use Python 3.11+ in a virtual environment, then install dependencies:

```sh
python -m pip install -r requirements.txt
jupyter notebook
```

Open a notebook from this repository and run it from the repository root.

## Review notes

- Organized from the supplied project archives; duplicate folders, cache files and assignment handouts were not counted as separate projects.
- Saved notebook outputs were cleared so old results are not mistaken for a current successful run.
- The datasets are included in this repository in the repository root, so the project can run after cloning. They were added at the repository owner's request. Their original sources, licenses and terms were not independently verified, and no license is assigned to them here; credit and rights remain with the original data providers.
- This is a portfolio/learning project, not a validated decision system. See `VALIDATION.md` for checks and remaining limits.

## Repaired baseline

Run `python train.py` for a tested train-only preprocessing baseline. Original notebooks are retained as exploratory source and are not all reproduced end to end. The tier classifier omits the rating that defines its target.
