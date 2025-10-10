# SRIM–SBI

A simple pipeline that combines **SRIM (Stopping and Range of Ions in Matter)** simulations with **simulation-based inference (SBI)** to model ion implantation. 

---

## What It Does

- Preprocesses SRIM output data  
- Trains a probabilistic posterior using [`sbi`]
- Samples from the posterior and runs new SRIM simulations  
- Summarizes results and performs posterior predictive checks (PPC)

---

## Structure

```
src/
 ├── utils/
 │   ├── data_utils.py      # preprocess, make_x_test
 │   ├── sbi_runner.py      # prior, inference, posterior
 │   ├── srim_utils.py      # run SRIM for posterior samples
 │   ├── srim_parser.py     # parse and summarize SRIM outputs
 │   └── analysis_utils.py  # PPC metrics and plots
 └── scripts/
     ├── train.py           # end-to-end training + SRIM + PPC
     └── ppc.py             # summarize + PPC only
```

---

## Example Workflow

```bash
# 1. Train posterior + run SRIM
python -m src.scripts.train

# 2. (Optional) Only summarize + PPC after SRIM is done
python -m src.scripts.ppc
```

---

##  Requirements

```
numpy
pandas
torch
sbi
srim
matplotlib
tqdm
scipy
```
