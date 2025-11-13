import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Open two-start-two-choice-results.csv as a pandas dataframe
tstc_df = pd.read_parquet('data/processed/two-start-two-choice-results.parquet')
# set panda display options
pd.set_option('display.max_columns', None,
              'display.max_rows', None,
              'display.expand_frame_repr', False)

# ANALYSIZE: Compare performance between training conditions and look for sex effects,
# away vs. towards, and interaction effects therein.
summary_df = tstc_df[tstc_df['training_type'].isin(['PI', 'PI+VC_f2', 'VC'])].copy()
_sex = summary_df['sex'].astype(str).str.strip().str.lower()
sex_mapping = {'f': 'Female', 'm': 'Male', 'female': 'Female', 'male': 'Male'}
summary_df['sex'] = _sex.map(sex_mapping).fillna(summary_df['sex'])
TRAINING_ORDER = ['PI','PI+VC_f2', 'VC']

# Creating to dataframes one with PI, PI+VC_f2, VC and one with only PI+VC_f2 and VC
# This is the analysize the effect of cue approach training seperately from PI training
# that had no cue.

trials_to_acq_sct_df = summary_df[['sex', 'cue_approach', 'training_type', 'trials_to_acq']].copy()
trials_to_acq_sct_log_df = trials_to_acq_sct_df.copy()
trials_to_acq_sct_log_df['trials_to_acq'] = np.log(trials_to_acq_sct_log_df['trials_to_acq'])

trials_to_acq_st_df = trials_to_acq_sct_df[trials_to_acq_sct_df['training_type'] != 'PI'].copy()

DV = 'trials_to_acq'
FACTORS_SCT = ['sex', 'cue_approach', 'training_type'] #CA: cue approach
FACTORS_ST = ['sex', 'training_type'] #TTA: trials to acquisition


# CHECK ASSUMPTIONS FUNCTIONS for ANOVA
def _levene_groups_column(df: pd.DataFrame, factors: list[str] | None=None) -> pd.Series:
    # Create a grouping column for Levene's test based on factors
    return df[factors].astype(str).agg('_'.join, axis=1)


def check_assumptions(df: pd.DataFrame,
                      factors: list[str] | None=None,
                      dv: str = None
                      ) -> dict:
    from scipy import stats
    from statsmodels.stats.diagnostic import lilliefors
    # Basic DV checks
    desc = df[dv].describe()
    skew = stats.skew(df[dv], nan_policy="omit")
    kurt = stats.kurtosis(df[dv], fisher=True, nan_policy="omit")

    # OLS residuals for normality (fit saturated ANOVA model)
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    formula = f'{dv} ~ ' + " * ".join([f'C({f})' for f in factors])
    model = ols(formula, data=df).fit()
    resid = model.resid.dropna()

    # Shapiro (n<=5000 recommended) and Lilliefors (Kolmogorov–Smirnov adj)
    shapiro_p = np.nan
    if len(resid) <= 5000:
        shapiro_p = stats.shapiro(resid)[1]
    try:
        lillie_stat, lillie_p = lilliefors(resid.values, dist='norm')
    except Exception:
        lillie_p = np.nan

    # Levene across cells
    from scipy.stats import levene
    grp = _levene_groups_column(df)
    groups = [df.loc[grp==g, DV].dropna().values for g in sorted(grp.unique())]
    # filter empty groups
    groups = [g for g in groups if len(g)>1]
    lev = levene(*groups, center='median') if len(groups)>=2 else None

    return {
        "dv_describe": desc.to_dict(),
        "skew": float(skew) if np.isfinite(skew) else np.nan,
        "kurtosis_fisher": float(kurt) if np.isfinite(kurt) else np.nan,
        "resid_shapiro_p": float(shapiro_p) if shapiro_p==shapiro_p else np.nan,
        "resid_lilliefors_p": float(lillie_p) if lillie_p==lillie_p else np.nan,
        "levene_W": float(lev.statistic) if lev else np.nan,
        "levene_p": float(lev.pvalue) if lev else np.nan,
        "n": int(df[DV].notna().sum())
    }


sct_assumptions = check_assumptions(trials_to_acq_sct_df, FACTORS_SCT, DV)
st_assumptions = check_assumptions(trials_to_acq_st_df, FACTORS_ST, DV)

print("Assumptions Check - SCT (with Cue Approach):")
print(sct_assumptions)
print("\nAssumptions Check - ST (without PI):")
print(st_assumptions)

# Create log-transformed versions of the dataframes
trials_to_acq_sct_log_df = trials_to_acq_sct_df.copy()
trials_to_acq_sct_log_df['trials_to_acq'] = np.log(trials_to_acq_sct_log_df['trials_to_acq'])
trials_to_acq_st_log_df = trials_to_acq_sct_log_df[trials_to_acq_sct_log_df['training_type'] != 'PI'].copy()

def run_anova_python(
        df: pd.DataFrame, 
        factors: list[str] | None = None,
        dv: str = None
        ) -> pd.DataFrame:
    if factors is None or dv is None:
        raise RuntimeError["Factors and DV must be provided for run_anova_python()."]
    
    formula = f'{dv} ~ ' + " * ".join  ([f'C({f})' for f in factors])
    model = ols(formula, data=df).fit()
    aov = anova_lm(model, typ=3)
    # tidy up
    aov = aov.reset_index().rename(columns={"index":"term"})
    aov["term"] = aov["term"].str.replace("C\\(|\\)", "", regex=True)
    return aov


anova_results_tta_df = run_anova_python(trials_to_acq_df, FACTORS_ST, DV)
anova_results_tta_log_df = run_anova_python(trials_to_acq_log_df, FACTORS_ST, DV)
anova_results_CA_df = run_anova_python(cue_approach_df, FACTORS_SCT, DV)
anova_results_CA_log_df = run_anova_python(cue_approach_log_df, FACTORS_SCT, DV)

print("ANOVA Results (Original DV):")
print(anova_results_tta_df)
print("\nANOVA Results (Log-Transformed DV):")
print(anova_results_tta_log_df)
print("\nANOVA Results with Cue Approach (Original DV):")
print(anova_results_CA_df)
print("\nANOVA Results with Cue Approach (Log-Transformed DV):")
print(anova_results_CA_log_df)


def run_bonferroni_posthoc_python(
    df: pd.DataFrame,
    factors: list[str] | None = None,
    dv: str = None,
    equal_var: bool = False
) -> pd.DataFrame:
    """
    Compute pairwise t-tests for each categorical factor with Bonferroni correction.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing the dependent variable and factors.
    factors : list[str], optional
        Factor names to test. Defaults to factors.
    dv : str
        Dependent-variable column name.
    equal_var : bool
        Passed to scipy.stats.ttest_ind (False = Welch's t-test).

    Returns
    -------
    pandas.DataFrame
        Concatenated post-hoc results with raw and Bonferroni-adjusted p-values.
    """
    if factors is None or dv is None:
        raise RuntimeError["Factors and DV must be provided for run_bonferroni_posthoc_python()."]

    all_rows = []
    for factor in factors:
        levels = df[factor].dropna().unique()
        pairs = combinations(sorted(levels), 2)
        rows = []
        for a, b in pairs:
            group_a = df[df[factor] == a][dv].dropna()
            group_b = df[df[factor] == b][dv].dropna()
            if len(group_a) < 2 or len(group_b) < 2:
                continue
            t_stat, p_val = scipy_stats.ttest_ind(group_a, group_b, equal_var=equal_var)
            rows.append({
                'factor': factor,
                'contrast': f'{a} vs {b}',
                't_stat': t_stat,
                'p_raw': p_val,
                'n_a': len(group_a),
                'n_b': len(group_b),
                'mean_a': group_a.mean(),
                'mean_b': group_b.mean()
            })
        if not rows:
            continue
        p_vals = [r['p_raw'] for r in rows]
        reject, p_adj, _, _ = multipletests(p_vals, method='bonferroni')
        for r, p_corr, rej in zip(rows, p_adj, reject):
            r['p_bonf'] = p_corr
            r['reject_bonf'] = bool(rej)
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


python_posthoc_acq_log_df = run_bonferroni_posthoc_python(trials_to_acq_log_df, FACTORS_ST, DV)
print("\nPython Bonferroni Post-hoc acq-log (Log-Transformed DV):")
print(python_posthoc_acq_log_df)
python_posthoc_ca_log_df = run_bonferroni_posthoc_python(cue_approach_log_df, FACTORS_SCT, DV)
print("\nPython Bonferroni Post-hoc cue-approach (Log-Transformed DV):")
print(python_posthoc_ca_log_df)

def plot_bars_with_sex_scatter(
        ax,
        data: pd.DataFrame,
        dv: str,
        training_order: list[str],
        title: str,
        training_col: str = 'training_type',
        sex_col: str = 'sex',
        palette_name: str = 'Blues'):
    """Bar chart of DV means with male/female scatter overlays."""
    working = data.copy()
    sex_mapping = {'f': 'Female', 'm': 'Male', 'female': 'Female', 'male': 'Male'}
    working['sex_clean'] = (
        working[sex_col].astype(str).str.strip().str.lower().map(sex_mapping).fillna(working[sex_col])
    )
    order = [cat for cat in training_order if cat in working[training_col].unique()]
    palette = sns.color_palette(palette_name, len(order))
    sns.barplot(
        data=working,
        x=training_col,
        y=dv,
        hue=training_col,
        estimator=np.mean,
        errorbar='se',
        palette=palette,
        edgecolor='black',
        ax=ax,
        order=order,
        legend=False,
        hue_order=order
    )

    sex_styles = {
        'Female': {'facecolors': 'none', 'edgecolors': '#123A63', 'linewidths': 1.5, 'zorder': 5},
        'Male': {'facecolors': '#123A63', 'edgecolors': '#0B203E', 'linewidths': 1, 'zorder': 5}
    }
    rng = np.random.default_rng(12345)
    seen = set()

    for sex, style in sex_styles.items():
        subset = working[working['sex_clean'] == sex]
        if subset.empty:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(subset))
        codes = pd.Categorical(subset[training_col], categories=order, ordered=True).codes.astype(float)
        x_positions = codes + jitter
        ax.scatter(
            x_positions,
            subset[dv],
            marker='o',
            s=55,
            label=sex if sex not in seen else None,
            clip_on=False,
            **style
        )
        seen.add(sex)

    ax.set_title(title)
    ax.set_xlabel('Training Type')
    ax.set_ylabel(f'{dv.replace("_", " ").title()}' + (' (ln)' if 'Log' in title else ''))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)


sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
plot_bars_with_sex_scatter(axes[0], trials_to_acq_df, DV, TRAINING_ORDER, 'Trials to Acquisition (Raw)')
plot_bars_with_sex_scatter(axes[1], trials_to_acq_log_df, DV, TRAINING_ORDER, 'Trials to Acquisition (Log)')
handles, labels = axes[0].get_legend_handles_labels()
if handles:
    axes[0].legend(handles, labels, title='Sex', frameon=False, loc='upper left')
plt.tight_layout()
plt.show()
