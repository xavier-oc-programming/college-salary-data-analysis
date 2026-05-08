import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch
from pathlib import Path

PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)
DATA_PATH = Path('data/salaries_by_college_major.csv')

GROUP_COLOURS = {'STEM': '#2E75B6', 'Business': '#ED7D31', 'HASS': '#70AD47'}


def main():

    # ── 1. LOAD ───────────────────────────────────────────────────────────────
    print('=' * 70)
    print('1. LOAD')
    print('=' * 70)

    df = pd.read_csv(DATA_PATH)
    print(f'Loaded {len(df)} rows from {DATA_PATH}')

    # ── 2. INSPECTION & PROFILING ─────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('2. INSPECTION & PROFILING')
    print('=' * 70)

    print('\n--- df.shape ---')
    print(df.shape)

    print('\n--- df.info() ---')
    df.info()

    print('\n--- df.describe() ---')
    print(df.describe())

    print('\n--- df.dtypes ---')
    print(df.dtypes)

    print('\n--- df.nunique() ---')
    print(df.nunique())

    print("\n--- df['Group'].value_counts() ---")
    print(df['Group'].value_counts())

    print('\n--- df.duplicated().sum() ---')
    print(df.duplicated().sum())

    print('\n--- df.memory_usage(deep=True) ---')
    print(df.memory_usage(deep=True))

    print('\n--- df.isna() ---')
    print(df.isna())

    print('\n--- df.isna().sum() ---')
    print(df.isna().sum())

    print('\n--- df.head() ---')
    print(df.head())

    print('\n--- df.tail() ---')
    print(df.tail())

    # ── 3. CLEANING ───────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('3. CLEANING')
    print('=' * 70)

    clean_df = df.dropna()
    # Reset so index runs 0–50 after removing the footer row
    clean_df = clean_df.reset_index(drop=True)
    clean_df = clean_df.rename(columns={
        'Undergraduate Major':               'Major',
        'Starting Median Salary':            'Start',
        'Mid-Career Median Salary':          'Mid',
        'Mid-Career 10th Percentile Salary': 'P10',
        'Mid-Career 90th Percentile Salary': 'P90',
    })

    print('\n--- cleaned df.head() ---')
    print(clean_df.head())

    # fillna on a copy shows the alternative — the source data is unchanged
    fillna_demo = df.copy()
    fillna_demo[['Starting Median Salary', 'Mid-Career Median Salary']] = (
        fillna_demo[['Starting Median Salary', 'Mid-Career Median Salary']].fillna(0)
    )
    print('\n--- fillna demo (copy only, not applied to clean_df) ---')
    print(fillna_demo.tail(3))

    # ── 4. FEATURE ENGINEERING ────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('4. FEATURE ENGINEERING')
    print('=' * 70)

    clean_df['Spread'] = clean_df['P90'] - clean_df['P10']
    clean_df['Growth'] = (clean_df['Mid'] - clean_df['Start']) / clean_df['Start'] * 100
    # High safety means start salary is large relative to the earnings spread
    clean_df['Safety'] = clean_df['Start'] / clean_df['Spread']
    # ascending=False so rank 1 = highest salary
    clean_df['Start_Rank'] = clean_df['Start'].rank(ascending=False)
    clean_df['Mid_Rank']   = clean_df['Mid'].rank(ascending=False)
    # Positive = climbed in ranking by mid-career; negative = fell
    clean_df['Rank_Change'] = clean_df['Start_Rank'] - clean_df['Mid_Rank']
    clean_df['Salary_Band'] = pd.cut(
        clean_df['Start'], bins=3, labels=['Low', 'Medium', 'High']
    )
    clean_df['is_STEM'] = clean_df['Group'] == 'STEM'
    # transform preserves the original row count so we can add it as a column
    clean_df['Group_Avg_Start'] = clean_df.groupby('Group')['Start'].transform('mean')
    clean_df['vs_group_avg']    = clean_df['Start'] - clean_df['Group_Avg_Start']

    print(clean_df[[
        'Major', 'Spread', 'Growth', 'Safety', 'Rank_Change',
        'Salary_Band', 'Group_Avg_Start', 'vs_group_avg',
    ]].head(10).to_string())

    # ── 5. SELECTION & FILTERING ──────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('5. SELECTION & FILTERING')
    print('=' * 70)

    print("\n--- df['Major'] (single column) ---")
    print(clean_df['Major'].to_string())

    print("\n--- df[['Major', 'Start', 'Mid']] (multiple columns) ---")
    print(clean_df[['Major', 'Start', 'Mid']].head().to_string())

    print("\n--- df.loc[0, 'Major'] (label-based cell) ---")
    print(clean_df.loc[0, 'Major'])

    print('\n--- df.iloc[0:5, 0:3] (integer slice) ---')
    print(clean_df.iloc[0:5, 0:3].to_string())

    print("\n--- df[df['Start'] > 50000] ---")
    print(clean_df[clean_df['Start'] > 50000][['Major', 'Start', 'Group']].to_string())

    print("\n--- df[(df['Group'] == 'STEM') & (df['Start'] > 55000)] ---")
    print(
        clean_df[(clean_df['Group'] == 'STEM') & (clean_df['Start'] > 55000)]
        [['Major', 'Start']].to_string()
    )

    print("\n--- df[df['Group'].isin(['STEM', 'Business'])] ---")
    print(
        clean_df[clean_df['Group'].isin(['STEM', 'Business'])]
        [['Major', 'Group']].head(8).to_string()
    )

    print('\n--- df.query(\'Group == "STEM" and Start > 50000\') ---')
    print(
        clean_df.query('Group == "STEM" and Start > 50000')
        [['Major', 'Start']].to_string()
    )

    print("\n--- df[df['Start'].between(40000, 60000)] ---")
    print(
        clean_df[clean_df['Start'].between(40000, 60000)]
        [['Major', 'Start']].head(8).to_string()
    )

    print("\n--- df[df['Major'].str.contains('Engineering')] ---")
    print(
        clean_df[clean_df['Major'].str.contains('Engineering')]
        [['Major', 'Start', 'Mid']].to_string()
    )

    # ── 6. AGGREGATION ────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('6. AGGREGATION')
    print('=' * 70)

    print("\n--- df['Mid'].sum() ---")
    print(f"${clean_df['Mid'].sum():,.0f}")

    print("\n--- df['Start'].mean() ---")
    print(f"${clean_df['Start'].mean():,.0f}")

    print("\n--- df['Mid'].median() ---")
    print(f"${clean_df['Mid'].median():,.0f}")

    print("\n--- df['Start'].std() ---")
    print(f"${clean_df['Start'].std():,.0f}")

    print("\n--- df['Spread'].var() ---")
    print(f"{clean_df['Spread'].var():,.0f}")

    print("\n--- df['Mid'].min() / df['Mid'].max() ---")
    print(f"Min: ${clean_df['Mid'].min():,.0f}  |  Max: ${clean_df['Mid'].max():,.0f}")

    print("\n--- df['Mid'].idxmin() / df['Mid'].idxmax() ---")
    print(f"idxmin: {clean_df['Mid'].idxmin()}  |  idxmax: {clean_df['Mid'].idxmax()}")

    print("\n--- df['Start'].quantile(0.25) / quantile(0.75) ---")
    print(
        f"Q1: ${clean_df['Start'].quantile(0.25):,.0f}  |  "
        f"Q3: ${clean_df['Start'].quantile(0.75):,.0f}"
    )

    print("\n--- df[['Start', 'Mid', 'Spread']].agg(['mean', 'median', 'std']) ---")
    print(clean_df[['Start', 'Mid', 'Spread']].agg(['mean', 'median', 'std']).to_string())

    print('\n--- df.describe() ---')
    print(clean_df[['Start', 'Mid', 'P10', 'P90', 'Spread', 'Growth']].describe().to_string())

    # ── 7. GROUPBY ────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('7. GROUPBY')
    print('=' * 70)

    print('\n--- groupby mean ---')
    print(
        clean_df.groupby('Group')
        .mean(numeric_only=True)[['Start', 'Mid', 'Spread', 'Growth']]
        .to_string()
    )

    print('\n--- groupby median ---')
    print(
        clean_df.groupby('Group')
        .median(numeric_only=True)[['Start', 'Mid', 'Spread']]
        .to_string()
    )

    print('\n--- groupby min ---')
    print(
        clean_df.groupby('Group')
        .min(numeric_only=True)[['Start', 'Mid']]
        .to_string()
    )

    print('\n--- groupby max ---')
    print(
        clean_df.groupby('Group')
        .max(numeric_only=True)[['Start', 'Mid']]
        .to_string()
    )

    print('\n--- groupby count ---')
    print(clean_df.groupby('Group').count()[['Major']].to_string())

    print("\n--- groupby agg: Start + Spread ---")
    print(
        clean_df.groupby('Group')
        .agg({'Start': ['mean', 'min', 'max'], 'Spread': 'mean'})
        .to_string()
    )

    print("\n--- groupby nlargest top 3 Mid per group ---")
    print(clean_df.groupby('Group')['Mid'].nlargest(3).to_string())

    print('\n--- groupby apply: top 2 Start per group ---')
    print(
        clean_df
        .groupby('Group', group_keys=False)
        .apply(lambda grp: grp.nlargest(2, 'Start'))
        [['Major', 'Group', 'Start']]
        .to_string()
    )

    print('\n--- transform mean: Major, Group, Start, Group_Avg_Start, vs_group_avg ---')
    print(
        clean_df[['Major', 'Group', 'Start', 'Group_Avg_Start', 'vs_group_avg']]
        .head(10)
        .to_string()
    )

    stem_avg_mid    = clean_df[clean_df['Group'] == 'STEM']['Mid'].mean()
    hass_avg_mid    = clean_df[clean_df['Group'] == 'HASS']['Mid'].mean()
    stem_avg_growth = clean_df[clean_df['Group'] == 'STEM']['Growth'].mean()
    hass_avg_growth = clean_df[clean_df['Group'] == 'HASS']['Growth'].mean()

    # ── 8. SORTING & RANKING ──────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('8. SORTING & RANKING')
    print('=' * 70)

    print('\n--- top 10 by Mid salary ---')
    print(
        clean_df.sort_values('Mid', ascending=False)
        [['Major', 'Group', 'Mid']].head(10).to_string()
    )

    print('\n--- sort by Group asc, Start desc ---')
    print(
        clean_df.sort_values(['Group', 'Start'], ascending=[True, False])
        [['Major', 'Group', 'Start']].head(10).to_string()
    )

    print('\n--- lowest spread (safest majors) ---')
    print(clean_df.sort_values('Spread')[['Major', 'Spread']].head(10).to_string())

    print('\n--- nlargest 5 by P90 ---')
    print(clean_df.nlargest(5, 'P90')[['Major', 'P90']].to_string())

    print('\n--- nsmallest 5 by Spread ---')
    print(clean_df.nsmallest(5, 'Spread')[['Major', 'Spread']].to_string())

    print('\n--- group means sorted by index ---')
    print(
        clean_df.groupby('Group')
        .mean(numeric_only=True)[['Start', 'Mid']]
        .sort_index()
        .to_string()
    )

    print('\n--- top 10 rank climbers (Start → Mid) ---')
    print(
        clean_df.sort_values('Rank_Change', ascending=False)
        [['Major', 'Group', 'Start_Rank', 'Mid_Rank', 'Rank_Change']]
        .head(10)
        .to_string()
    )

    # ── 9. RESHAPING & PIVOTING ───────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('9. RESHAPING & PIVOTING')
    print('=' * 70)

    print('\n--- pivot_table: Mid by Group ---')
    print(
        clean_df.pivot_table(
            values='Mid', index='Group', aggfunc=['mean', 'max', 'min']
        ).to_string()
    )

    print('\n--- set_index Major → loc Economics ---')
    print(clean_df.set_index('Major').loc['Economics'].to_string())

    print('\n--- groupby mean transposed ---')
    print(
        clean_df.groupby('Group')
        .mean(numeric_only=True)[['Start', 'Mid', 'Spread', 'Growth']]
        .T
        .to_string()
    )

    melted = clean_df.melt(
        id_vars=['Major', 'Group'],
        value_vars=['Start', 'Mid', 'P10', 'P90'],
        var_name='Salary_Type',
        value_name='Amount',
    )
    print('\n--- melt: first 10 rows ---')
    print(melted.head(10).to_string())

    # ── 10. APPLY & MAP ───────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('10. APPLY & MAP')
    print('=' * 70)

    tier_labels = clean_df.apply(
        lambda row: 'High'   if row['Mid'] > 90000 else
                    'Medium' if row['Mid'] > 65000 else 'Low',
        axis=1,
    )
    print('\n--- apply: salary tier counts ---')
    print(tier_labels.value_counts().to_string())

    group_full = clean_df['Group'].map({
        'STEM':     'Science & Technology',
        'Business': 'Business & Commerce',
        'HASS':     'Humanities & Social Sciences',
    })
    print('\n--- map: Group full name counts ---')
    print(group_full.value_counts().to_string())

    print('\n--- str.upper (first 5) ---')
    print(clean_df['Major'].str.upper().head(5).to_string())

    print("\n--- str.contains 'Engineering' ---")
    print(clean_df[clean_df['Major'].str.contains('Engineering')]['Major'].tolist())

    print('\n--- str.len sorted descending (top 5) ---')
    print(clean_df['Major'].str.len().sort_values(ascending=False).head(5).to_string())

    print('\n--- str.split first word (first 10) ---')
    print(clean_df['Major'].str.split().str[0].head(10).to_string())

    print('\n--- pipe: STEM sorted by Mid ---')
    stem_sorted = (
        clean_df
        .pipe(lambda d: d[d['Group'] == 'STEM'])
        .pipe(lambda d: d.sort_values('Mid', ascending=False))
    )
    print(stem_sorted[['Major', 'Mid']].head(5).to_string())

    # ── 11. DISPLAY & STYLING ─────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('11. DISPLAY & STYLING')
    print('=' * 70)

    pd.options.display.float_format = '{:,.0f}'.format
    pd.options.display.max_rows = 51

    print('\n--- top 10 by Mid-Career salary (formatted) ---')
    print(
        clean_df.sort_values('Mid', ascending=False)
        [['Major', 'Group', 'Start', 'Mid']].head(10).to_string()
    )

    print('\n--- group summary table (formatted) ---')
    print(
        clean_df.groupby('Group')
        .mean(numeric_only=True)[['Start', 'Mid', 'Spread', 'Growth']]
        .to_string()
    )

    print('\n--- rank change: climbers and fallers ---')
    rank_table = clean_df[
        ['Major', 'Group', 'Start_Rank', 'Mid_Rank', 'Rank_Change']
    ].copy()
    rank_table['Direction'] = rank_table['Rank_Change'].apply(
        lambda x: 'Climber' if x > 0 else ('Faller' if x < 0 else 'Stable')
    )
    print(rank_table.sort_values('Rank_Change', ascending=False).head(15).to_string())

    pd.reset_option('display.float_format')

    # ── 12. CORRELATION ANALYSIS ──────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('12. CORRELATION ANALYSIS')
    print('=' * 70)

    corr_cols   = ['Start', 'Mid', 'P10', 'P90', 'Spread', 'Growth']
    corr_matrix = clean_df[corr_cols].corr()

    print('\n--- correlation matrix ---')
    print(corr_matrix.to_string())

    pearson_start_mid = corr_matrix.loc['Start', 'Mid']
    print(f'\nPearson correlation (Start vs Mid): {pearson_start_mid:.3f}')
    print(
        'Interpretation: strong positive correlation — majors with high starting '
        'salaries tend to have high mid-career salaries too.'
    )

    # Upper-triangle mask avoids counting each pair twice
    upper_mask  = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_pairs  = (
        corr_matrix.where(upper_mask)
        .stack()
        .sort_values(ascending=False)
    )
    print('\n--- two most correlated pairs ---')
    print(corr_pairs.head(2).to_string())

    stem_avg_start   = clean_df[clean_df['Group'] == 'STEM']['Start'].mean()
    hass_beating_stem = clean_df[
        (clean_df['Group'] == 'HASS') & (clean_df['Start'] > stem_avg_start)
    ]
    print(f'\n--- HASS majors with Start > STEM avg (${stem_avg_start:,.0f}) ---')
    print(
        hass_beating_stem[['Major', 'Start']].to_string()
        if not hass_beating_stem.empty else '  (none)'
    )

    # ── 13. EXPORT ────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('13. EXPORT')
    print('=' * 70)

    clean_df.to_csv('data/clean_salaries.csv', index=False)
    print('Saved: data/clean_salaries.csv')

    clean_df.groupby('Group').mean(numeric_only=True).to_csv('data/group_summary.csv')
    print('Saved: data/group_summary.csv')

    clean_df.to_json('data/salaries.json', orient='records', indent=2)
    print('Saved: data/salaries.json')

    # ── 14. VISUALISATIONS ────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('14. VISUALISATIONS')
    print('=' * 70)

    sns.set_theme(style='whitegrid', palette='muted')

    # ── Chart 1: top 10 mid-career salary ────────────────────────────────────
    top10_mid = clean_df.sort_values('Mid', ascending=False).head(10)
    fig, ax   = plt.subplots(figsize=(10, 6))
    bar_colours = [GROUP_COLOURS[g] for g in top10_mid['Group']]
    bars = ax.barh(top10_mid['Major'], top10_mid['Mid'], color=bar_colours)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'${x:,.0f}')
    )
    for bar, val in zip(bars, top10_mid['Mid']):
        ax.text(
            bar.get_width() + 500,
            bar.get_y() + bar.get_height() / 2,
            f'${val:,.0f}', va='center', fontsize=9,
        )
    legend_handles = [
        Patch(color=col, label=grp) for grp, col in GROUP_COLOURS.items()
        if grp in top10_mid['Group'].values
    ]
    ax.legend(handles=legend_handles, loc='lower right')
    ax.set_title('Top 10 Majors by Mid-Career Median Salary')
    ax.set_xlabel('Mid-Career Median Salary')
    plt.tight_layout()
    chart1_path = PLOTS_DIR / '01_top10_midcareer.png'
    plt.savefig(chart1_path, dpi=150)
    plt.close()
    print(f'Saved: {chart1_path}')

    # ── Chart 2: average salary by degree group ───────────────────────────────
    groups       = ['STEM', 'Business', 'HASS']
    metrics      = ['Start', 'Mid', 'P10', 'P90']
    group_means  = clean_df.groupby('Group')[metrics].mean()
    x            = np.arange(len(metrics))
    bar_width    = 0.25
    fig, ax      = plt.subplots(figsize=(11, 6))
    for i, grp in enumerate(groups):
        ax.bar(
            x + i * bar_width, group_means.loc[grp, metrics],
            width=bar_width, label=grp, color=GROUP_COLOURS[grp],
        )
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(metrics)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f'${y:,.0f}')
    )
    ax.set_title('Average Salary by Degree Group')
    ax.set_xlabel('Salary Metric')
    ax.set_ylabel('Average Salary')
    ax.legend()
    plt.tight_layout()
    chart2_path = PLOTS_DIR / '02_group_comparison.png'
    plt.savefig(chart2_path, dpi=150)
    plt.close()
    print(f'Saved: {chart2_path}')

    # ── Chart 3: scatter Starting vs Mid-Career with regression per group ─────
    fig, ax = plt.subplots(figsize=(10, 7))
    for grp in groups:
        subset = clean_df[clean_df['Group'] == grp]
        ax.scatter(
            subset['Start'], subset['Mid'],
            color=GROUP_COLOURS[grp], label=grp, s=60, alpha=0.8,
        )
        # Linear regression for this group's trend line
        coeffs  = np.polyfit(subset['Start'], subset['Mid'], 1)
        x_line  = np.linspace(subset['Start'].min(), subset['Start'].max(), 100)
        ax.plot(
            x_line, np.polyval(coeffs, x_line),
            color=GROUP_COLOURS[grp], linewidth=1.5, linestyle='--', alpha=0.6,
        )
    # Label the 5 majors furthest above the overall regression line
    overall_coeffs   = np.polyfit(clean_df['Start'], clean_df['Mid'], 1)
    clean_df['_res'] = clean_df['Mid'] - np.polyval(overall_coeffs, clean_df['Start'])
    for _, row in clean_df.nlargest(5, '_res').iterrows():
        ax.annotate(
            row['Major'], (row['Start'], row['Mid']),
            textcoords='offset points', xytext=(6, 4), fontsize=8,
        )
    clean_df.drop(columns=['_res'], inplace=True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'${y:,.0f}'))
    ax.set_title('Starting vs Mid-Career Salary by Major')
    ax.set_xlabel('Starting Median Salary')
    ax.set_ylabel('Mid-Career Median Salary')
    ax.legend()
    plt.tight_layout()
    chart3_path = PLOTS_DIR / '03_scatter_salary_trajectory.png'
    plt.savefig(chart3_path, dpi=150)
    plt.close()
    print(f'Saved: {chart3_path}')

    # ── Chart 4: top 10 salary growth % ──────────────────────────────────────
    top10_growth    = clean_df.sort_values('Growth', ascending=False).head(10)
    growth_colours  = [GROUP_COLOURS[g] for g in top10_growth['Group']]
    fig, ax         = plt.subplots(figsize=(10, 6))
    growth_bars     = ax.barh(
        top10_growth['Major'], top10_growth['Growth'], color=growth_colours
    )
    ax.invert_yaxis()
    for bar, val in zip(growth_bars, top10_growth['Growth']):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}%', va='center', fontsize=9,
        )
    legend_handles = [
        Patch(color=col, label=grp) for grp, col in GROUP_COLOURS.items()
        if grp in top10_growth['Group'].values
    ]
    ax.legend(handles=legend_handles, loc='lower right')
    ax.set_title('Top 10 Majors by Salary Growth (Start → Mid-Career)')
    ax.set_xlabel('Growth (%)')
    plt.tight_layout()
    chart4_path = PLOTS_DIR / '04_top10_growth.png'
    plt.savefig(chart4_path, dpi=150)
    plt.close()
    print(f'Saved: {chart4_path}')

    # ── Chart 5: correlation heatmap ──────────────────────────────────────────
    corr_data = clean_df[['Start', 'Mid', 'P10', 'P90', 'Spread', 'Growth']].corr()
    fig, ax   = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_data, annot=True, fmt='.2f',
        cmap='coolwarm', center=0, ax=ax, linewidths=0.5,
    )
    ax.set_title('Salary Metric Correlation Heatmap')
    plt.tight_layout()
    chart5_path = PLOTS_DIR / '05_correlation_heatmap.png'
    plt.savefig(chart5_path, dpi=150)
    plt.close()
    print(f'Saved: {chart5_path}')

    # ── Chart 6: box plot mid-career salary by group ──────────────────────────
    fig, ax      = plt.subplots(figsize=(8, 6))
    group_order  = ['STEM', 'Business', 'HASS']
    group_data   = [clean_df[clean_df['Group'] == g]['Mid'].values for g in group_order]
    bp           = ax.boxplot(group_data, patch_artist=True, labels=group_order)
    for patch, grp in zip(bp['boxes'], group_order):
        patch.set_facecolor(GROUP_COLOURS[grp])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'${y:,.0f}'))
    ax.set_title('Mid-Career Salary Distribution by Degree Group')
    ax.set_xlabel('Degree Group')
    ax.set_ylabel('Mid-Career Median Salary')
    plt.tight_layout()
    chart6_path = PLOTS_DIR / '06_boxplot_by_group.png'
    plt.savefig(chart6_path, dpi=150)
    plt.close()
    print(f'Saved: {chart6_path}')

    # ── Chart 7: rank change climbers and fallers ─────────────────────────────
    climbers   = clean_df.sort_values('Rank_Change', ascending=False).head(10)
    fallers    = clean_df.sort_values('Rank_Change').head(10)
    chart_data = (
        pd.concat([climbers, fallers])
        .drop_duplicates('Major')
        .sort_values('Rank_Change')
    )
    # Red for fallers, dark green for climbers
    bar_colours_rank = [
        '#C00000' if v < 0 else '#375623' for v in chart_data['Rank_Change']
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(chart_data['Major'], chart_data['Rank_Change'], color=bar_colours_rank)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Salary Rank Change: Starting → Mid-Career')
    ax.set_xlabel('Rank Change (positive = climbed by mid-career)')
    plt.tight_layout()
    chart7_path = PLOTS_DIR / '07_rank_change.png'
    plt.savefig(chart7_path, dpi=150)
    plt.close()
    print(f'Saved: {chart7_path}')

    # ── 15. FINAL SUMMARY ─────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('15. FINAL SUMMARY')
    print('=' * 70)

    highest_mid_row = clean_df.loc[clean_df['Mid'].idxmax()]
    lowest_mid_row  = clean_df.loc[clean_df['Mid'].idxmin()]
    biggest_grower  = clean_df.loc[clean_df['Growth'].idxmax()]
    top_climber     = clean_df.loc[clean_df['Rank_Change'].idxmax()]
    safest_major    = clean_df.loc[clean_df['Spread'].idxmin()]
    highest_p90_row = clean_df.loc[clean_df['P90'].idxmax()]

    print(f"""
Total majors analysed          : {len(clean_df)}
Highest mid-career salary      : {highest_mid_row['Major']} (${highest_mid_row['Mid']:,.0f})
Lowest mid-career salary       : {lowest_mid_row['Major']} (${lowest_mid_row['Mid']:,.0f})
Biggest salary grower          : {biggest_grower['Major']} ({biggest_grower['Growth']:.2f}%)
Top rank climber (Start→Mid)   : {top_climber['Major']} (+{top_climber['Rank_Change']:.0f} positions)
Safest major (lowest spread)   : {safest_major['Major']} (spread ${safest_major['Spread']:,.0f})
Highest earning ceiling (P90)  : {highest_p90_row['Major']} (${highest_p90_row['P90']:,.0f})
Pearson r (Start vs Mid)       : {pearson_start_mid:.3f}
STEM vs HASS mid-career gap    : ${stem_avg_mid - hass_avg_mid:,.0f}
STEM vs HASS avg Growth gap    : {stem_avg_growth:.2f}% vs {hass_avg_growth:.2f}% ({stem_avg_growth - hass_avg_growth:.2f}pp difference)
""")


if __name__ == '__main__':
    main()
