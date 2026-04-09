import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from ctgan import CTGAN
except ImportError:
    print("ctgan not found. Install with: pip install ctgan")
    sys.exit(1)

INPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'hierarchical_fault_dataset.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'hierarchical_fault_dataset_augmented.csv')

DISCRETE_COLUMNS = ['Fault_Type', 'Severity_Level', 'Fault_Type_Name', 'Severity_Name', 'Hierarchical_Label']

CTGAN_EPOCHS = 300
TARGET_SYNTHETIC_PER_CLASS = 300
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


def _get_feature_columns(df):
    exclude = ['Reading', 'Label', 'Fault_Type', 'Severity_Level',
               'Fault_Type_Name', 'Severity_Name', 'Hierarchical_Label', 'Data_Source']
    return [c for c in df.columns if c not in exclude]


def generate_synthetic_data(df_original: pd.DataFrame) -> pd.DataFrame:
    print(f"Original dataset: {len(df_original)} samples")
    print(f"Fault type distribution:\n{df_original['Fault_Type_Name'].value_counts()}\n")

    feature_columns = _get_feature_columns(df_original)
    train_cols = feature_columns + DISCRETE_COLUMNS

    available_cols = [c for c in train_cols if c in df_original.columns]
    available_discrete = [c for c in DISCRETE_COLUMNS if c in df_original.columns]

    ctgan = CTGAN(epochs=CTGAN_EPOCHS, verbose=True)
    ctgan.fit(df_original[available_cols], available_discrete)

    fault_classes = df_original['Fault_Type'].unique()
    synthetic_parts = []

    for fault_class in sorted(fault_classes):
        class_df = df_original[df_original['Fault_Type'] == fault_class]
        n_existing = len(class_df)
        n_generate = max(TARGET_SYNTHETIC_PER_CLASS - n_existing, 0)

        if n_generate == 0:
            continue

        samples = ctgan.sample(n_generate * 3)
        samples = samples[samples['Fault_Type'] == fault_class].head(n_generate)

        if len(samples) < n_generate:
            extra = ctgan.sample((n_generate - len(samples)) * 5)
            extra = extra[extra['Fault_Type'] == fault_class].head(n_generate - len(samples))
            samples = pd.concat([samples, extra], ignore_index=True)

        samples = samples.head(n_generate)
        samples['Data_Source'] = 'Synthetic'
        synthetic_parts.append(samples)
        print(f"  Class {fault_class}: generated {len(samples)} synthetic samples")

    if not synthetic_parts:
        print("No synthetic samples generated.")
        return df_original

    df_synthetic = pd.concat(synthetic_parts, ignore_index=True)
    df_original_tagged = df_original.copy()
    df_original_tagged['Data_Source'] = 'Original'

    df_augmented = pd.concat([df_original_tagged, df_synthetic], ignore_index=True)
    df_augmented = df_augmented.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df_augmented['Reading'] = range(len(df_augmented))

    print(f"\nAugmented dataset: {len(df_augmented)} samples")
    print(f"  Original: {len(df_original_tagged)}")
    print(f"  Synthetic: {len(df_synthetic)}")
    print(f"\nFault type distribution after augmentation:\n{df_augmented['Fault_Type_Name'].value_counts()}")

    return df_augmented


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input file not found: {INPUT_PATH}")
        print("Place 'hierarchical_fault_dataset.csv' in the data/ folder.")
        sys.exit(1)

    df_original = pd.read_csv(INPUT_PATH)

    feature_columns = _get_feature_columns(df_original)
    for col in feature_columns:
        df_original[col] = df_original[col].fillna(df_original[col].median())

    df_augmented = generate_synthetic_data(df_original)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_augmented.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved augmented dataset to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
