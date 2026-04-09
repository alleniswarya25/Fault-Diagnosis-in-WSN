import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

INPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'hierarchical_fault_dataset.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'hierarchical_fault_dataset_augmented.csv')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FAULT_TYPES = {
    0: 'Normal',
    1: 'Hardover',
    2: 'Drift',
    3: 'Spike',
    4: 'Erratic',
    5: 'Stuck',
}

SEVERITY_MAP = {
    0: ('None',     0),
    1: ('Low',      1),
    2: ('Medium',   2),
    3: ('High',     3),
    4: ('Critical', 4),
}


def _assign_severity(fault_type: int, magnitude: float) -> int:
    if fault_type == 0:
        return 0
    if magnitude < 0.25:
        return 1
    elif magnitude < 0.50:
        return 2
    elif magnitude < 0.75:
        return 3
    else:
        return 4


def inject_hardover(series: pd.Series, magnitude: float = 0.6) -> pd.Series:
    result = series.copy().astype(float)
    col_range = series.max() - series.min()
    shift = magnitude * col_range * np.random.choice([-1, 1])
    result += shift
    return result


def inject_drift(series: pd.Series, magnitude: float = 0.4) -> pd.Series:
    result = series.copy().astype(float)
    n = len(result)
    drift = np.linspace(0, magnitude * (series.max() - series.min()), n)
    result += drift
    return result


def inject_spike(series: pd.Series, magnitude: float = 0.8, spike_frac: float = 0.05) -> pd.Series:
    result = series.copy().astype(float)
    n_spikes = max(1, int(len(result) * spike_frac))
    spike_idx = np.random.choice(len(result), n_spikes, replace=False)
    col_range = series.max() - series.min()
    result.iloc[spike_idx] += magnitude * col_range * np.random.choice([-1, 1], n_spikes)
    return result


def inject_erratic(series: pd.Series, magnitude: float = 0.5) -> pd.Series:
    result = series.copy().astype(float)
    col_std = series.std()
    noise = np.random.normal(0, magnitude * col_std * 3, len(result))
    result += noise
    return result


def inject_stuck(series: pd.Series, magnitude: float = 0.5, stuck_frac: float = 0.3) -> pd.Series:
    result = series.copy().astype(float)
    stuck_val = series.sample(1).values[0]
    n_stuck = int(len(result) * stuck_frac)
    start = np.random.randint(0, len(result) - n_stuck)
    result.iloc[start:start + n_stuck] = stuck_val
    return result


INJECTORS = {
    1: inject_hardover,
    2: inject_drift,
    3: inject_spike,
    4: inject_erratic,
    5: inject_stuck,
}


def inject_faults(df: pd.DataFrame, feature_columns: list, n_per_class: int = 200) -> pd.DataFrame:
    injected_rows = []

    for fault_type, fault_name in FAULT_TYPES.items():
        if fault_type == 0:
            continue

        injector = INJECTORS[fault_type]
        magnitudes = np.linspace(0.2, 0.9, n_per_class)

        for i, magnitude in enumerate(magnitudes):
            base_row = df[df['Fault_Type'] == 0].sample(1, random_state=i).copy()
            row = base_row.copy()

            for col in feature_columns:
                if col in row.columns:
                    row[col] = injector(row[col], magnitude).values

            severity = _assign_severity(fault_type, magnitude)
            severity_name, severity_level = SEVERITY_MAP[severity]

            row['Fault_Type'] = fault_type
            row['Fault_Type_Name'] = fault_name
            row['Severity_Level'] = severity_level
            row['Severity_Name'] = severity_name
            row['Hierarchical_Label'] = f"{fault_name}_{severity_name}"
            row['Label'] = 1
            row['Data_Source'] = 'Injected'

            injected_rows.append(row)

    if not injected_rows:
        return df

    df_injected = pd.concat(injected_rows, ignore_index=True)
    print(f"Injected {len(df_injected)} fault samples across {len(FAULT_TYPES) - 1} fault types")
    return df_injected


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input file not found: {INPUT_PATH}")
        print("Place 'hierarchical_fault_dataset.csv' in the data/ folder.")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)

    exclude_cols = ['Reading', 'Label', 'Fault_Type', 'Severity_Level',
                    'Fault_Type_Name', 'Severity_Name', 'Hierarchical_Label', 'Data_Source']
    feature_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

    for col in feature_columns:
        df[col] = df[col].fillna(df[col].median())

    if 'Data_Source' not in df.columns:
        df['Data_Source'] = 'Original'

    print(f"Base dataset: {len(df)} samples")
    print(f"Feature columns: {feature_columns}\n")

    df_injected = inject_faults(df, feature_columns, n_per_class=200)

    df_combined = pd.concat([df, df_injected], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df_combined['Reading'] = range(len(df_combined))

    print(f"\nFinal dataset: {len(df_combined)} samples")
    print(f"Fault type distribution:\n{df_combined['Fault_Type_Name'].value_counts()}")
    print(f"Severity distribution:\n{df_combined['Severity_Name'].value_counts()}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
