# Titanic Python Workflow

This repository is configured as a Python-only data science project. It uses the local `Titanic.csv` dataset to run a minimal end-to-end binary classification workflow for passenger survival.

## Project Structure

```text
.
|-- Titanic.csv
|-- README.md
|-- requirements.txt
|-- outputs/
|-- src/
|   `-- titanic_workflow.py
`-- .gitignore
```

## Setup

### Exact setup used in this workspace

This repository was rebuilt using the Python installation at `C:\Users\hp\AppData\Local\Programs\Python\Python314`:

```powershell
& 'C:\Users\hp\AppData\Local\Programs\Python\Python314\python.exe' -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Standard setup if `python` is already on `PATH`

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run The Workflow

```powershell
& .\.venv\Scripts\python.exe .\src\titanic_workflow.py
```

## What The Script Does

- Loads `Titanic.csv` with pandas
- Prints column names, dtypes, and missing-value counts
- Uses `Survived` as the binary target
- Drops identifier and high-cardinality text columns: `PassengerId`, `Name`, `Ticket`, `Cabin`
- Splits the data into train and test sets
- Applies numeric and categorical preprocessing with scikit-learn
- Trains a baseline logistic regression model
- Evaluates with accuracy and ROC-AUC
- Writes outputs to `outputs/`

## Outputs

After a successful run, you should see:

- `outputs/metrics.json`
- `outputs/roc_curve.png`
