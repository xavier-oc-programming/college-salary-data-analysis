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
