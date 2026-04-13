# Data Exploration with Pandas: College Major vs Your Salary

Explores post-university salary outcomes by major using pandas — cleaning data, ranking by earning potential, measuring salary risk, and comparing degree categories with group aggregation.

A 51-major dataset (sourced from a Wall Street Journal / PayScale survey) is loaded into a pandas DataFrame, cleaned of missing values, and explored across three angles: which majors pay the most on entry, which carry the least long-term salary risk, and how STEM, Business, and HASS graduates compare on average. The analysis moves from individual-major lookups to category-level pivot-style summaries using `.groupby()`.

The project is structured as a set of Jupyter notebooks split into theory (annotated lesson notes) and practice (hands-on exercises). There are no external services, no credentials, and no database — the only dependency is pandas and a single CSV file.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Analysis Flow](#2-analysis-flow)
3. [Features](#3-features)
4. [Dataset Schema](#4-dataset-schema)
5. [Architecture](#5-architecture)
6. [Notebook Reference](#6-notebook-reference)
7. [Configuration Reference](#7-configuration-reference)
8. [Course Context](#8-course-context)
9. [Dependencies](#9-dependencies)

---

## 1. Quick Start

```bash
git clone https://github.com/xavier-oc-programming/day-72-pandas-college-salary.git
cd day-72-pandas-college-salary
pip install -r requirements.txt
jupyter notebook
```

Open any notebook in `theory/` to read the lesson or any notebook in `practice/` to work through the exercises. The data file is at `data/salaries_by_college_major.csv` — practice notebooks reference it via `../data/salaries_by_college_major.csv`.

---

## 2. Analysis Flow

```
pipeline
    │
    │  ── [Ingestion] ────────────────────────────────────────────────
    ├── pd.read_csv()  →  salaries_by_college_major.csv  →  df (52 rows × 6 columns)
    │
    │  ── [Inspection] ───────────────────────────────────────────────
    ├── .head() / .tail()        →  preview rows, spot the NaN footer row
    ├── .shape / .columns        →  confirm 52 rows × 6 columns and column names
    ├── .isna() / .isna().sum()  →  NaN confirmed to last row only
    │
    │  ── [Cleaning] ─────────────────────────────────────────────────
    ├── df.dropna()  →  clean_df  (NaN footer removed — 51 majors remain)
    │
    │  ── [Lookup] ───────────────────────────────────────────────────
    ├── .max() / .idxmax()      →  row index of highest starting salary major
    ├── .min() / .idxmin()      →  row index of lowest starting / mid-career salary major
    ├── clean_df.loc[idx, col]  →  major name retrieved at that index
    │
    │  ── [Feature Engineering] ──────────────────────────────────────
    ├── 90th Percentile − 10th Percentile       →  spread_col (salary risk proxy)
    ├── clean_df.insert(1, 'Spread', spread_col) →  Spread column added to clean_df
    │
    │  ── [Ranking] ──────────────────────────────────────────────────
    ├── .sort_values('Spread')                                        →  low_risk
    ├── .sort_values('Mid-Career 90th Percentile Salary', asc=False)  →  highest_potential
    ├── .sort_values('Spread', ascending=False)                       →  greatest_salary_spread
    └── .sort_values(['Spread', '90th Percentile'], ascending=False)  →  high_risk_high_reward
```

---

## 3. Features

- Load and inspect the raw salary dataset (shape, columns, null counts)
- Detect and remove the trailing NaN row with `.isna()` / `.dropna()`
- Find the highest and lowest starting salary majors with `.idxmax()` / `.idxmin()`
- Find the highest and lowest mid-career salary majors
- Add a computed `Spread` column (90th − 10th percentile) as a salary-risk proxy
- Rank majors by lowest risk (smallest spread)
- Rank majors by highest earning potential (top 90th percentile salary)
- Compare high-risk / high-reward degree combinations
- Aggregate all salary columns by degree category (STEM / Business / HASS) using `.groupby()`
- Format float output to comma-separated 2 decimal places

---

## 4. Dataset Schema

**File:** `data/salaries_by_college_major.csv`
**Rows:** 51 majors (52 in raw file — last row is NaN footer, removed on clean)

| Column | Type | Description |
|---|---|---|
| Undergraduate Major | str | Major name |
| Starting Median Salary | float | Median salary at career start |
| Mid-Career Median Salary | float | Median salary at mid-career |
| Mid-Career 10th Percentile Salary | float | Lower-bound mid-career salary |
| Mid-Career 90th Percentile Salary | float | Upper-bound mid-career salary |
| Group | str | Degree category: STEM / Business / HASS |

**Computed column (added at runtime):**

| Column | Formula | Description |
|---|---|---|
| Spread | 90th Percentile − 10th Percentile | Salary range — proxy for earnings risk |

---

## 5. Architecture

```
Day_72_Data_Exploration_with_Pandas_College_Major_vs_Your_Salary/
│
├── data/
│   └── salaries_by_college_major.csv     # WSJ/PayScale seed dataset
│
├── theory/                               # Annotated lesson notebooks
│   ├── 00__Day_72_Goals_...ipynb         # Objectives and what you will build
│   ├── 01__Getting_Set_Up_...ipynb       # Jupyter setup and notebook format
│   ├── 02__Upload_the_Data_...ipynb      # pd.read_csv(), DataFrame basics
│   ├── 03__Preliminary_Data_...ipynb     # .shape, .columns, .isna(), .dropna()
│   ├── 04__Accessing_Columns_...ipynb    # Column access, .max(), .idxmax(), .loc[]
│   ├── 05__Solution_...ipynb             # Solution: highest / lowest earners
│   ├── 06__Sorting_Values_...ipynb       # .sort_values(), .insert(), spread column
│   ├── 07__Solution_...ipynb             # Solution: highest potential
│   ├── 08__Grouping_and_Pivoting_...ipynb # .groupby(), .mean(numeric_only=True)
│   └── 09__Learning_Points_...ipynb      # Full concept summary
│
├── practice/                             # Hands-on exercise notebooks
│   ├── A_03_Prelim_Data.ipynb            # Exercise: inspect and clean the dataset
│   ├── A_04_Accessing_Cols.ipynb         # Exercise: column access and extreme values
│   └── A_06_Sorting_vals.ipynb           # Exercise: sorting, spread, groupby
│
├── docs/
│   └── COURSE_NOTES.md                   # Original course exercise brief
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 6. Notebook Reference

### theory/

| Notebook | Key methods covered |
|---|---|
| 00__Day_72_Goals | Overview only |
| 01__Getting_Set_Up | Jupyter format, notebook vs IDE |
| 02__Upload_the_Data | `pd.read_csv()`, DataFrame creation |
| 03__Preliminary_Data | `.head()`, `.tail()`, `.shape`, `.columns`, `.isna()`, `.dropna()` |
| 04__Accessing_Columns | `df['col']`, `.max()`, `.idxmax()`, `.loc[idx, col]` |
| 05__Solution | Worked solution: highest / lowest salaries |
| 06__Sorting_Values | `.sort_values()`, `.insert()`, arithmetic on columns |
| 07__Solution | Worked solution: highest earning potential |
| 08__Grouping_and_Pivoting | `.groupby()`, `.mean(numeric_only=True)`, display formatting |
| 09__Learning_Points | Concept summary |

### practice/

| Notebook | Exercises |
|---|---|
| A_03_Prelim_Data | Load CSV, inspect shape and columns, detect and remove NaN rows |
| A_04_Accessing_Cols | Access salary columns, find extreme values, look up majors by index |
| A_06_Sorting_vals | Add spread column, sort by risk, sort by potential, compare high-risk/high-reward |

---

## 7. Configuration Reference

There is no `config.py` — the only constant worth noting is the CSV path, set inline in each practice notebook:

| Value | Location | Description |
|---|---|---|
| `../data/salaries_by_college_major.csv` | practice notebooks | Path to the seed dataset relative to the notebook |
| `pd.options.display.float_format` | theory/08 | Set to `'{:,.2f}'.format` for readable salary output |

---

## 8. Course Context

**100 Days of Code: The Complete Python Pro Bootcamp** — Day 72.
Topics: pandas DataFrames, data cleaning, column arithmetic, sorting, indexing with `.loc`, `.groupby()` aggregation.

See [docs/COURSE_NOTES.md](docs/COURSE_NOTES.md) for the original exercise brief and key concepts.

---

## 9. Dependencies

| Module | Used in | Purpose |
|---|---|---|
| `pandas` | all practice notebooks | DataFrame creation, cleaning, analysis, aggregation |
| `notebook` | — | Jupyter notebook server for running `.ipynb` files |
