# Course Notes — Day 72: Data Exploration with Pandas

**Course:** 100 Days of Code: The Complete Python Pro Bootcamp
**Day:** 72
**Topic:** Advanced Data Exploration with Pandas

---

## Original Exercise Brief

Analyse a real-world dataset of post-university salary outcomes by undergraduate major.
The dataset (sourced from a Wall Street Journal / PayScale survey) covers 51 majors grouped
into three categories: STEM, Business, and HASS (Humanities, Arts, Social Sciences).

### Tasks set by the course

1. Load `salaries_by_college_major.csv` into a pandas DataFrame.
2. Inspect the dataset: shape, column names, missing values.
3. Clean the data by removing rows with NaN values.
4. Identify the highest and lowest starting and mid-career salaries using `.max()`, `.min()`, `.idxmax()`, `.idxmin()`.
5. Access individual cells with `df.loc[]`.
6. Add a `Spread` column representing salary risk (90th − 10th percentile).
7. Sort by spread to find the lowest-risk majors.
8. Sort by 90th percentile salary to find the highest-earning-potential majors.
9. Group by degree category and compute per-group averages with `.groupby()` + `.mean()`.
10. Format float output to 2 decimal places with `pd.options.display.float_format`.

### Key concepts covered

- `pd.read_csv()` — loading tabular data
- `.head()`, `.tail()`, `.shape`, `.columns` — DataFrame inspection
- `.isna()`, `.dropna()` — detecting and handling missing data
- Column access: `df['col']`, `df[['col1', 'col2']]`
- Cell access: `df.loc[idx, 'col']`
- `.max()`, `.min()`, `.idxmax()`, `.idxmin()` — extreme values
- `.sort_values()` — ranking rows
- `.insert()` — adding a computed column at a specific position
- `.groupby()` + `.mean(numeric_only=True)` — pivot-table-style aggregation
- `pd.options.display.float_format` — display formatting

---

## Lesson Notes — From the Course

### 0. Day 72 Goals

The project uses real PayScale salary data to answer meaningful questions about college major choices:

- Which degrees have the **highest starting salaries**?
- Which majors have the **lowest earnings** after college?
- Which degrees have the **highest earning potential**?
- Which majors represent the **lowest risk** from an earnings standpoint?
- Do **STEM**, **Business**, or **HASS** graduates earn more on average?

Core pandas skills reinforced: `read_csv`, `head`, `info`, `describe`, `isna`, `dropna`, column/row selection, `sort_values`, `groupby`, `pivot_table`.

---

### 1. Getting Set Up for Data Science

Python notebooks are the standard environment for data science. They divide code into cells that execute independently and display output immediately below each cell.

**Google Colab** provides an online Jupyter-style notebook with no local setup. To create one: Google Drive → New → More → Google Colaboratory.

**Local alternative:** install Anaconda, which bundles Jupyter Notebook.

Run a cell with **Shift + Enter**. Autocompletion: `Ctrl+Space` (Windows) / `Cmd+Space` (Mac).

---

### 2. Loading Data

Upload the CSV to Colab via the Files sidebar (drag and drop), then:

```python
import pandas as pd
df = pd.read_csv('salaries_by_college_major.csv')
df.head()
```

For a local notebook with the CSV in the same directory, the path is identical. Use a relative or absolute path if the file is elsewhere.

`df.head()` previews the first five rows — enough to verify column names, data types, and structure before proceeding.

---

### 3. Preliminary Data Exploration and Data Cleaning

**Answering structural questions first:**

```python
df.shape      # (51, 6) — 51 rows, 6 columns
df.columns    # Index of column names
df.isna()     # Boolean DataFrame — True where value is NaN
df.isna().sum()  # Count of NaN values per column
df.tail()     # Last 5 rows — reveals the PayScale footer row
```

**The junk row:** the last row contains `"Source: PayScale Inc."` as the major name and NaN for every salary column. It is not data and must be removed before analysis.

**Removing it:**

```python
clean_df = df.dropna()   # removes any row with at least one NaN
clean_df.tail()          # verify footer is gone
```

`dropna()` is appropriate here because the only NaN row is the footer. In datasets with real missing values, you would need a more targeted approach.

---

### 4. Accessing Columns and Individual Cells

**Single column → Series:**

```python
clean_df['Starting Median Salary']
```

**Maximum value in a column:**

```python
clean_df['Starting Median Salary'].max()     # the value
clean_df['Starting Median Salary'].idxmax()  # the row index (e.g. 43)
```

**Retrieve the major at that index:**

```python
clean_df['Undergraduate Major'].loc[43]      # 'Physician Assistant'
clean_df.loc[43, 'Undergraduate Major']      # preferred — more explicit
```

**Retrieve the entire row:**

```python
clean_df.loc[43]   # returns a Series with all columns for row 43
```

`.loc` is preferred over chained `df['col'][idx]` because it is explicit, safer with slices, and consistent with pandas best practices.

**Challenge answers:**

```python
# Highest mid-career salary
clean_df.loc[clean_df['Mid-Career Median Salary'].idxmax(),
             ['Undergraduate Major', 'Mid-Career Median Salary']]
# → Chemical Engineering, $107,000

# Lowest starting salary
clean_df.loc[clean_df['Starting Median Salary'].idxmin(),
             ['Undergraduate Major', 'Starting Median Salary']]
# → Spanish, $34,000

# Lowest mid-career salary
clean_df.loc[clean_df['Mid-Career Median Salary'].idxmin(),
             ['Undergraduate Major', 'Mid-Career Median Salary']]
# → Education, $52,000
```

---

### 6. Sorting Values and Adding Columns

**Salary risk concept:** a low-risk major has a small spread between the 10th and 90th percentile mid-career salaries — earnings are predictable. A high-spread major has wide variability; the top earners do very well, but the bottom earners do not.

**Computing spread:**

```python
spread_col = (
    clean_df['Mid-Career 90th Percentile Salary']
    - clean_df['Mid-Career 10th Percentile Salary']
)
# Alternative using .subtract():
spread_col = clean_df['Mid-Career 90th Percentile Salary'].subtract(
    clean_df['Mid-Career 10th Percentile Salary']
)
```

Pandas arithmetic is vectorised — subtracting two columns produces a new Series aligned by index, with no loop required.

**Inserting the column:**

```python
clean_df.insert(1, 'Spread', spread_col)
```

`insert(position, name, data)` places the column at a specific index. Position `1` makes it the second column, improving readability.

**Sorting to find low-risk majors:**

```python
low_risk = clean_df.sort_values('Spread')   # ascending by default
low_risk[['Undergraduate Major', 'Spread']].head()
# → Nursing ($50,700), Physician Assistant ($57,600), ...
```

**Finding highest potential:**

```python
highest_potential = clean_df.sort_values(
    'Mid-Career 90th Percentile Salary', ascending=False
)
highest_potential[['Undergraduate Major', 'Mid-Career 90th Percentile Salary']].head()
# → Economics ($210,000), Finance ($195,000), ...
```

---

### 8. Grouping and Pivoting Data with Pandas

**Why groupby:** moving from individual majors to degree categories (STEM / Business / HASS) reveals group-level patterns hidden at the individual level — the pandas equivalent of an Excel Pivot Table.

```python
clean_df.groupby('Group').count()                        # majors per group
clean_df.groupby('Group').mean(numeric_only=True)        # average salaries per group
```

`numeric_only=True` prevents a `TypeError` when non-numeric columns (like the major name) would otherwise be included in the aggregation.

**Formatting output as currency:**

```python
pd.options.display.float_format = '{:,.2f}'.format
```

This affects display only — the underlying float values are unchanged.

**Reading the result:** each row is a degree category (Business, HASS, STEM). Each column shows the average of that salary metric for the group. STEM averages higher across all metrics.

---

### 9. Learning Points and Summary

Key methods consolidated from Day 72:

| Method | Purpose |
|---|---|
| `.head()` / `.tail()` | Preview first/last rows |
| `.shape` | Row and column count |
| `.columns` | Column name index |
| `.isna()` / `.dropna()` | Detect and remove NaN values |
| `df['col']` / `df[['c1','c2']]` | Column selection |
| `df.loc[idx, 'col']` | Label-based cell access |
| `.max()` / `.min()` | Extreme values |
| `.idxmax()` / `.idxmin()` | Row index of extreme values |
| `.sort_values()` | Rank rows by column |
| `.insert()` | Add computed column at position |
| `.groupby()` + `.mean()` | Pivot-table-style aggregation |
| `pd.options.display.float_format` | Currency display formatting |

---

## Extended Analysis — Beyond the Course

The following operations are used in `analysis.py` and `analysis.ipynb` but were not covered in the original course material. Each was developed independently to extend the analysis beyond the course scope.

### Inspection

| Operation | What it does | Usage |
|---|---|---|
| `df.info()` | Prints dtypes, non-null counts, memory | `df.info()` |
| `df.describe()` | Statistical summary (count, mean, std, quartiles) | `df.describe()` |
| `df.dtypes` | Series of dtype per column | `print(df.dtypes)` |
| `df.nunique()` | Number of distinct values per column | `df.nunique()` |
| `df['col'].value_counts()` | Frequency of each unique value | `df['Group'].value_counts()` |
| `df.duplicated().sum()` | Count of duplicate rows | `df.duplicated().sum()` |
| `df.memory_usage(deep=True)` | Bytes consumed per column | `df.memory_usage(deep=True)` |

### Cleaning

| Operation | What it does | Usage |
|---|---|---|
| `reset_index(drop=True)` | Resets integer index after dropping rows | `clean_df.reset_index(drop=True)` |
| `rename(columns={})` | Renames columns using a dictionary | `df.rename(columns={'Old': 'New'})` |
| `fillna(value)` | Fills NaN with a specified value | `df[cols].fillna(0)` (demo on copy) |

### Feature Engineering

| Operation | What it does | Usage |
|---|---|---|
| `col / col` (Safety) | Element-wise division of two columns | `clean_df['Start'] / clean_df['Spread']` |
| `rank(ascending=False)` | Assigns rank 1 to highest value | `clean_df['Start'].rank(ascending=False)` |
| `pd.cut(col, bins, labels)` | Bins continuous values into labelled categories | `pd.cut(clean_df['Start'], bins=3, labels=[...])` |
| Boolean column (`col == val`) | Creates a True/False column | `clean_df['Group'] == 'STEM'` |
| `groupby().transform('mean')` | Broadcasts per-group mean back to each row | `clean_df.groupby('Group')['Start'].transform('mean')` |

### Selection & Filtering

| Operation | What it does | Usage |
|---|---|---|
| `iloc[r:r, c:c]` | Integer-position slice | `clean_df.iloc[0:5, 0:3]` |
| `isin([list])` | Membership filter | `clean_df[clean_df['Group'].isin(['STEM', 'Business'])]` |
| `query('expr')` | SQL-like string filter | `clean_df.query('Group == "STEM" and Start > 50000')` |
| `between(lo, hi)` | Inclusive range filter | `clean_df[clean_df['Start'].between(40000, 60000)]` |
| `str.contains('pattern')` | Substring filter on string column | `clean_df[clean_df['Major'].str.contains('Engineering')]` |

### Aggregation

| Operation | What it does | Usage |
|---|---|---|
| `sum()` | Column total | `clean_df['Mid'].sum()` |
| `median()` | Middle value | `clean_df['Mid'].median()` |
| `std()` | Standard deviation | `clean_df['Start'].std()` |
| `var()` | Variance | `clean_df['Spread'].var()` |
| `quantile(q)` | Value at percentile q | `clean_df['Start'].quantile(0.25)` |
| `agg(['mean','median','std'])` | Multiple functions in one call | `clean_df[cols].agg(['mean', 'median', 'std'])` |

### GroupBy (extended)

| Operation | What it does | Usage |
|---|---|---|
| `groupby().median()` | Median per group | `clean_df.groupby('Group').median(numeric_only=True)` |
| `groupby().agg({'col': ['fn']})` | Different functions per column | `clean_df.groupby('Group').agg({'Start': ['mean','min','max']})` |
| `groupby()['col'].nlargest(n)` | Top n rows per group | `clean_df.groupby('Group')['Mid'].nlargest(3)` |
| `groupby().apply(lambda)` | Arbitrary per-group function | `clean_df.groupby('Group').apply(lambda g: g.nlargest(2, 'Start'))` |

### Sorting & Ranking

| Operation | What it does | Usage |
|---|---|---|
| `nlargest(n, col)` | Top n rows by column | `clean_df.nlargest(5, 'P90')` |
| `nsmallest(n, col)` | Bottom n rows by column | `clean_df.nsmallest(5, 'Spread')` |
| `sort_index()` | Sort by index (alphabetical after groupby) | `group_means.sort_index()` |

### Reshaping & Pivoting

| Operation | What it does | Usage |
|---|---|---|
| `pivot_table(values, index, aggfunc)` | Cross-tabulation with aggregation | `clean_df.pivot_table(values='Mid', index='Group', aggfunc=['mean','max','min'])` |
| `set_index(col)` | Promotes column to row index | `clean_df.set_index('Major').loc['Economics']` |
| `.T` | Transposes rows and columns | `group_means.T` |
| `melt(id_vars, value_vars, var_name, value_name)` | Wide to long format | `clean_df.melt(id_vars=['Major','Group'], value_vars=['Start','Mid','P10','P90'])` |

### Apply & Map

| Operation | What it does | Usage |
|---|---|---|
| `apply(func, axis=1)` | Applies a function row-by-row | `clean_df.apply(lambda row: 'High' if row['Mid'] > 90000 else ..., axis=1)` |
| `map(dict)` | Replaces values via dictionary lookup | `clean_df['Group'].map({'STEM': 'Science & Technology', ...})` |
| `str.upper()` | Uppercase string column | `clean_df['Major'].str.upper()` |
| `str.len()` | Length of each string | `clean_df['Major'].str.len()` |
| `str.split().str[0]` | First word of each string | `clean_df['Major'].str.split().str[0]` |
| `pipe(func)` | Chains transformations without intermediate variables | `clean_df.pipe(lambda d: d[d['Group']=='STEM']).pipe(...)` |

### Correlation

| Operation | What it does | Usage |
|---|---|---|
| `corr()` | Pairwise Pearson correlation matrix | `clean_df[cols].corr()` |
| `np.triu(ones, k=1)` | Upper-triangle mask (avoids duplicate pairs) | `np.triu(np.ones(corr.shape), k=1).astype(bool)` |
| `where(mask).stack()` | Extracts masked values as a flat Series | `corr.where(mask).stack().sort_values(ascending=False)` |

### Export

| Operation | What it does | Usage |
|---|---|---|
| `to_csv(index=False)` | Writes DataFrame to CSV without row numbers | `clean_df.to_csv('data/clean_salaries.csv', index=False)` |
| `to_json(orient='records', indent=2)` | Writes one JSON object per row | `clean_df.to_json('data/salaries.json', orient='records', indent=2)` |

### Visualisation (matplotlib + seaborn)

| Operation | What it does | Usage |
|---|---|---|
| `sns.set_theme()` | Sets a consistent base style | `sns.set_theme(style='whitegrid', palette='muted')` |
| `ax.barh()` | Horizontal bar chart | `ax.barh(majors, values, color=colours)` |
| `ax.invert_yaxis()` | Places highest value at top | Ensures top-ranked major appears first |
| `np.polyfit / np.polyval` | Linear regression coefficients and evaluation | Per-group trend lines on scatter chart |
| `ax.annotate()` | Labels individual data points | Labels the 5 biggest outliers on the scatter |
| `sns.heatmap()` | Annotated colour-coded matrix | Correlation heatmap with `annot=True, fmt='.2f'` |
| `ax.boxplot(patch_artist=True)` | Box plot with filled boxes | Distribution per group with group colours |
| `ax.axvline(0)` | Vertical reference line | Zero line on rank-change chart |
| `mticker.FuncFormatter` | Custom axis tick format | Currency labels (`$X,XXX`) on all salary axes |
| `matplotlib.patches.Patch` | Custom legend entry | Group colour swatches on bar charts |
| `plt.savefig(path, dpi=150)` | Saves chart to file | All 7 charts saved to `plots/` |

### Live Summary DataFrame (Section 15)

| Operation | What it does | Usage |
|---|---|---|
| `pd.DataFrame([tuples], columns=[...])` | Builds a display DataFrame from computed row tuples | Used in Section 15 to show live key stats that update on every run |
| f-string in cell values | Formats computed floats as readable strings inside the DataFrame | `f"${value:,.0f}"` for currency, `f"{pct:.1f}%"` for growth |

**Key concept:** building a DataFrame from computed values rather than hardcoding them means the summary table stays in sync with the data automatically on every run — no manual updates needed.
