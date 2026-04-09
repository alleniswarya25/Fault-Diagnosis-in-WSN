import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

EXCLUDE_COLS = [
    'Reading', 'Label', 'Fault_Type', 'Severity_Level',
    'Fault_Type_Name', 'Severity_Name', 'Hierarchical_Label', 'Data_Source',
]


def validate_synthetic_data(df: pd.DataFrame) -> dict:
    original = df[df['Data_Source'] == 'Original'].copy()
    synthetic = df[df['Data_Source'] == 'Synthetic'].copy()

    if synthetic.empty:
        print("No synthetic data found. Skipping validation.")
        return {}

    feature_columns = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_COLS
    ]

    results = {}

    ks_scores = []
    for col in feature_columns:
        stat, pval = stats.ks_2samp(original[col].dropna(), synthetic[col].dropna())
        ks_scores.append(1 - stat)

    results['mean_ks_complement'] = float(np.mean(ks_scores))

    corr_orig = original[feature_columns].corr().values
    corr_synt = synthetic[feature_columns].corr().values
    corr_sim = float(np.corrcoef(corr_orig.flatten(), corr_synt.flatten())[0, 1])
    results['correlation_similarity'] = corr_sim

    coverage_scores = []
    for col in feature_columns:
        orig_min, orig_max = original[col].min(), original[col].max()
        synt_vals = synthetic[col]
        in_range = ((synt_vals >= orig_min) & (synt_vals <= orig_max)).mean()
        coverage_scores.append(float(in_range))
    results['mean_range_coverage'] = float(np.mean(coverage_scores))

    fault_orig = original['Fault_Type'].value_counts(normalize=True).sort_index()
    fault_synt = synthetic['Fault_Type'].value_counts(normalize=True).sort_index()
    fault_synt = fault_synt.reindex(fault_orig.index, fill_value=0)
    results['fault_class_coverage'] = float((fault_synt > 0).mean())

    sev_orig = original['Severity_Level'].value_counts(normalize=True).sort_index()
    sev_synt = synthetic['Severity_Level'].value_counts(normalize=True).sort_index()
    sev_synt = sev_synt.reindex(sev_orig.index, fill_value=0)
    results['severity_class_coverage'] = float((sev_synt > 0).mean())

    print("\nSynthetic Data Validation Report")
    print("=" * 40)
    print(f"  Original samples:      {len(original)}")
    print(f"  Synthetic samples:     {len(synthetic)}")
    print(f"  KS Complement (mean):  {results['mean_ks_complement']:.4f}")
    print(f"  Correlation Similarity:{results['correlation_similarity']:.4f}")
    print(f"  Range Coverage:        {results['mean_range_coverage']:.4f}")
    print(f"  Fault Class Coverage:  {results['fault_class_coverage']:.4f}")
    print(f"  Severity Coverage:     {results['severity_class_coverage']:.4f}")

    return results


if __name__ == '__main__':
    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'hierarchical_fault_dataset_augmented.csv'
    )
    df = pd.read_csv(data_path)
    validate_synthetic_data(df)
