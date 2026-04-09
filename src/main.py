import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PowerTransformer, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Embedding,
    Flatten,
    Concatenate,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Reshape,
    Add,
)
from tensorflow.keras.models import Model


# =========================================================
# Assumption:
# data_enhanced is already loaded as a pandas DataFrame
# =========================================================

if "data_enhanced" not in globals():
    raise ValueError("data_enhanced DataFrame not found. Load your dataset first into variable: data_enhanced")


# =========================================================
# Utility helpers
# =========================================================
def make_output_dir(path="ensemble_outputs"):
    os.makedirs(path, exist_ok=True)
    return path


def advanced_feature_selection(X, y, feature_names, k=18):
    """
    Select top-k features using mutual information.
    """
    k = min(k, X.shape[1], len(feature_names))
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_indices]
    return X_selected, selected_indices, selected_feature_names, selector


def pad_and_average_histories(histories, keys=("accuracy", "val_accuracy", "loss", "val_loss")):
    """
    Average multiple Keras History objects with different epoch lengths.
    """
    avg_history = {}

    for key in keys:
        series_list = []
        max_len = max(len(h.history.get(key, [])) for h in histories)

        for h in histories:
            values = h.history.get(key, [])
            if len(values) == 0:
                values = [0.0] * max_len
            elif len(values) < max_len:
                values = values + [values[-1]] * (max_len - len(values))
            series_list.append(values)

        avg_history[key] = np.mean(np.array(series_list), axis=0)

    return avg_history


def plot_training_history(history_dict, task_name, output_dir):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if "accuracy" in history_dict:
        plt.plot(history_dict["accuracy"], label="Train Accuracy")
    if "val_accuracy" in history_dict:
        plt.plot(history_dict["val_accuracy"], label="Val Accuracy")
    plt.title(f"{task_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    if "loss" in history_dict:
        plt.plot(history_dict["loss"], label="Train Loss")
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="Val Loss")
    plt.title(f"{task_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"./try/training_history_{task_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# Model builders
# =========================================================
def build_gnn_model(num_cont_features, num_motes, num_classes):
    cont_input = Input(shape=(num_cont_features,), name="cont_input")
    mote_input = Input(shape=(1,), name="mote_input")

    mote_emb = Embedding(input_dim=num_motes + 1, output_dim=16)(mote_input)
    mote_emb = Flatten()(mote_emb)
    mote_emb = Dense(32, activation="relu")(mote_emb)

    x = Concatenate()([cont_input, mote_emb])
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Dense(64, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=[cont_input, mote_input], outputs=output, name="GNN_like_model")


def build_transformer_model(num_cont_features, num_motes, num_classes, embed_dim=64, num_heads=4):
    cont_input = Input(shape=(num_cont_features,), name="cont_input")
    mote_input = Input(shape=(1,), name="mote_input")

    # Convert feature vector to token sequence
    x_seq = Reshape((num_cont_features, 1))(cont_input)
    x_seq = Dense(embed_dim)(x_seq)

    attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(x_seq, x_seq)
    x_seq = Add()([x_seq, attn])
    x_seq = LayerNormalization()(x_seq)

    ff = Dense(embed_dim * 2, activation="relu")(x_seq)
    ff = Dense(embed_dim)(ff)
    x_seq = Add()([x_seq, ff])
    x_seq = LayerNormalization()(x_seq)

    x_seq = GlobalAveragePooling1D()(x_seq)

    mote_emb = Embedding(input_dim=num_motes + 1, output_dim=12)(mote_input)
    mote_emb = Flatten()(mote_emb)

    x = Concatenate()([x_seq, mote_emb])
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.30)(x)

    x = Dense(64, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=[cont_input, mote_input], outputs=output, name="Transformer_model")


def build_hybrid_model(num_cont_features, num_motes, num_classes):
    cont_input = Input(shape=(num_cont_features,), name="cont_input")
    mote_input = Input(shape=(1,), name="mote_input")

    # Dense branch
    dense_branch = Dense(128, activation="relu")(cont_input)
    dense_branch = BatchNormalization()(dense_branch)
    dense_branch = Dropout(0.30)(dense_branch)

    # Attention branch
    x_seq = Reshape((num_cont_features, 1))(cont_input)
    x_seq = Dense(64)(x_seq)
    attn = MultiHeadAttention(num_heads=4, key_dim=16)(x_seq, x_seq)
    x_seq = Add()([x_seq, attn])
    x_seq = LayerNormalization()(x_seq)
    x_seq = GlobalAveragePooling1D()(x_seq)

    # Mote branch
    mote_emb = Embedding(input_dim=num_motes + 1, output_dim=16)(mote_input)
    mote_emb = Flatten()(mote_emb)
    mote_emb = Dense(32, activation="relu")(mote_emb)

    x = Concatenate()([dense_branch, x_seq, mote_emb])
    x = Dense(192, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)

    x = Dense(96, activation="relu")(x)
    x = Dropout(0.20)(x)
    output = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=[cont_input, mote_input], outputs=output, name="Hybrid_model")


# =========================================================
# Ensemble trainer
# =========================================================
def train_ensemble_model(
    X_train,
    X_mote_train,
    y_train,
    X_val,
    X_mote_val,
    y_val,
    num_cont_features,
    num_motes,
    num_classes,
    task_name,
):
    models = []
    histories = []

    model_configs = [
        ("GNN", build_gnn_model),
        ("Transformer", build_transformer_model),
        ("Hybrid", build_hybrid_model),
    ]

    unique_classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
    class_weights = dict(zip(unique_classes, weights))

    for model_name, model_builder in model_configs:
        print(f"\nTraining {model_name} model for {task_name}...")

        model = model_builder(num_cont_features, num_motes, num_classes)

        if model_name == "GNN":
            optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.005)
        elif model_name == "Transformer":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)
        else:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0006, weight_decay=0.008)

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=20,
                restore_best_weights=True,
                mode="max",
                min_delta=0.001,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.7,
                patience=7,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        history = model.fit(
            [X_train, X_mote_train],
            y_train,
            validation_data=([X_val, X_mote_val], y_val),
            epochs=100,
            batch_size=64,
            verbose=1,
            callbacks=callbacks,
            class_weight=class_weights,
        )

        models.append(model)
        histories.append(history)

    return models, histories


# =========================================================
# Main training pipeline
# =========================================================
output_dir = make_output_dir("ensemble_outputs")

base_features = [
    "Humidity", "Temperature", "Is_Singlehop",
    "Temp_Mean", "Temp_Std", "Temp_Range", "Temp_Change",
    "Hum_Mean", "Hum_Std", "Hum_Range", "Hum_Change",
    "Temp_Hum_Ratio", "Temp_Hum_Product", "Total_Variability",
    "Combined_Change",
]

new_features = [
    "Temp_Hum_corr", "Temp_Hum_Ratio_Safe", "Temp_Hum_Diff",
    "System_Variability", "Change_Intensity", "Temp_Mean_Std_Ratio",
    "Hum_Mean_Std_Ratio", "Total_Insstability", "Mote_Group",
    "Netowrk_Position", "Temp_Quartile", "Hum_Quartile",
    "Combined_Quartile", "Temp_Momentum", "Hum_Momentum",
]

feature_cols = base_features + new_features
existing_features = [f for f in feature_cols if f in data_enhanced.columns]

if not existing_features:
    raise ValueError("No matching feature columns found in data_enhanced")

print(f"Using {len(existing_features)} features: {existing_features}")

tasks = {
    "Fault_Type": {
        "target": "Fault_Type",
        "num_classes": 6,
        "name": "Fault Type Classification",
    },
    "Severity_Level": {
        "target": "Severity_Level",
        "num_classes": 5,
        "name": "Severity Level Classification",
    },
}

all_results = {}

for task_name, task_info in tasks.items():
    print(f"\n{'=' * 80}")
    print(f"Task: {task_info['name']} - Ensemble approach")
    print(f"{'=' * 80}")

    target_col = task_info["target"]

    if target_col not in data_enhanced.columns:
        print(f"Skipping {task_name}: column '{target_col}' not found")
        continue

    # -------------------------
    # Prepare target
    # -------------------------
    y_raw = data_enhanced[target_col].copy()

    if y_raw.dtype == "object":
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y_raw.astype(str))
        class_names = list(target_encoder.classes_)
    else:
        y_encoded = y_raw.to_numpy()
        class_names = [str(c) for c in sorted(np.unique(y_encoded))]

    # -------------------------
    # Prepare features
    # -------------------------
    X_cont_full = data_enhanced[existing_features].copy()
    mote_raw = data_enhanced["Mote ID"].astype(str).copy()

    # Remove rows with NA in needed columns
    valid_mask = (~X_cont_full.isna().any(axis=1)) & (~pd.isna(y_raw)) & (~pd.isna(mote_raw))
    X_cont_full = X_cont_full.loc[valid_mask].reset_index(drop=True)
    mote_raw = mote_raw.loc[valid_mask].reset_index(drop=True)
    y_encoded = np.array(y_encoded)[valid_mask.to_numpy()]

    # Remove ultra-rare classes to allow stratified train/val/test
    class_counts = pd.Series(y_encoded).value_counts()
    keep_classes = class_counts[class_counts >= 3].index
    class_mask = pd.Series(y_encoded).isin(keep_classes).to_numpy()

    X_cont_full = X_cont_full.loc[class_mask].reset_index(drop=True)
    mote_raw = mote_raw.loc[class_mask].reset_index(drop=True)
    y_encoded = np.array(y_encoded)[class_mask]

    if len(np.unique(y_encoded)) < 2:
        print(f"Skipping {task_name}: not enough classes after filtering")
        continue

    print("Class counts after filtering:")
    print(pd.Series(y_encoded).value_counts().sort_index())

    mote_encoder = LabelEncoder()
    X_mote_full = mote_encoder.fit_transform(mote_raw)
    num_motes = len(np.unique(X_mote_full))

    X_cont_full = X_cont_full.to_numpy(dtype=np.float32)

    # -------------------------
    # Strategic split
    # -------------------------
    idx = np.arange(len(y_encoded))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.18,
        random_state=42,
        stratify=y_encoded,
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.22,
        random_state=23,
        stratify=y_encoded[train_idx],
    )

    X_cont_train_raw = X_cont_full[train_idx]
    X_cont_val_raw = X_cont_full[val_idx]
    X_cont_test_raw = X_cont_full[test_idx]

    X_mote_train = X_mote_full[train_idx]
    X_mote_val = X_mote_full[val_idx]
    X_mote_test = X_mote_full[test_idx]

    y_train = y_encoded[train_idx]
    y_val = y_encoded[val_idx]
    y_test = y_encoded[test_idx]

    # -------------------------
    # Feature selection on train only
    # -------------------------
    print("Applying advanced feature selection...")
    X_cont_train_sel, selected_indices, selected_feature_names, selector = advanced_feature_selection(
        X_cont_train_raw, y_train, existing_features, k=18
    )
    X_cont_val_sel = selector.transform(X_cont_val_raw)
    X_cont_test_sel = selector.transform(X_cont_test_raw)

    print(f"Selected {X_cont_train_sel.shape[1]} features from {len(existing_features)}")
    print(f"Selected features: {selected_feature_names}")

    # -------------------------
    # Advanced scaling on train only
    # -------------------------
    power_transformer = PowerTransformer(method="yeo-johnson")
    robust_scaler = RobustScaler()

    X_cont_train = power_transformer.fit_transform(X_cont_train_sel)
    X_cont_val = power_transformer.transform(X_cont_val_sel)
    X_cont_test = power_transformer.transform(X_cont_test_sel)

    X_cont_train = robust_scaler.fit_transform(X_cont_train)
    X_cont_val = robust_scaler.transform(X_cont_val)
    X_cont_test = robust_scaler.transform(X_cont_test)

    # -------------------------
    # Advanced oversampling on train only
    # -------------------------
    print("Applying advanced oversampling...")
    X_combined_train = np.column_stack([X_cont_train, X_mote_train])

    try:
        smotetomek = SMOTETomek(
            smote=BorderlineSMOTE(random_state=42, k_neighbors=3),
            random_state=42,
        )
        X_combined_res, y_train_res = smotetomek.fit_resample(X_combined_train, y_train)
        print(f"Train={len(train_idx)} -> {len(y_train_res)} (after SMOTETomek)")
    except Exception as e:
        print(f"Advanced oversampling failed: {e}. Using SMOTE fallback.")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_combined_res, y_train_res = smote.fit_resample(X_combined_train, y_train)
        print(f"Train={len(train_idx)} -> {len(y_train_res)} (after SMOTE)")

    X_cont_train_res = X_combined_res[:, :-1].astype(np.float32)
    X_mote_train_res = np.rint(X_combined_res[:, -1]).astype(int)
    X_mote_train_res = np.clip(X_mote_train_res, 0, num_motes - 1)

    print(f"Val={len(val_idx)}, Test={len(test_idx)}")

    # Reshape mote arrays for embedding input
    X_mote_train_res = X_mote_train_res.reshape(-1, 1)
    X_mote_val_in = X_mote_val.reshape(-1, 1)
    X_mote_test_in = X_mote_test.reshape(-1, 1)

    # -------------------------
    # Train ensemble
    # -------------------------
    models, histories = train_ensemble_model(
        X_cont_train_res,
        X_mote_train_res,
        y_train_res,
        X_cont_val,
        X_mote_val_in,
        y_val,
        X_cont_train_res.shape[1],
        num_motes,
        len(np.unique(y_encoded)),
        task_name,
    )

    avg_history = pad_and_average_histories(histories)
    plot_training_history(avg_history, f"{task_name}_Ensemble", output_dir)

    # -------------------------
    # Ensemble prediction with advanced TTA
    # -------------------------
    print("\nApplying ensemble prediction with TTA...")
    ensemble_predictions = []
    val_accuracies = []

    for model in models:
        model_predictions = []

        for i in range(8):
            if i == 0:
                pred = model.predict([X_cont_test, X_mote_test_in], batch_size=128, verbose=0)
            elif i < 4:
                noise_scale = 0.01 * i
                noise = np.random.normal(0, noise_scale, X_cont_test.shape)
                pred = model.predict([X_cont_test + noise, X_mote_test_in], batch_size=128, verbose=0)
            else:
                dropout_rate = 0.05 * (i - 3)
                mask = np.random.binomial(1, 1 - dropout_rate, X_cont_test.shape)
                pred = model.predict([X_cont_test * mask, X_mote_test_in], batch_size=128, verbose=0)

            model_predictions.append(pred)

        model_avg_pred = np.mean(model_predictions, axis=0)
        ensemble_predictions.append(model_avg_pred)

        val_pred = model.predict([X_cont_val, X_mote_val_in], batch_size=128, verbose=0)
        val_acc = accuracy_score(y_val, np.argmax(val_pred, axis=1))
        val_accuracies.append(val_acc)

    # Weighted ensemble
    weights = np.array(val_accuracies, dtype=np.float32)
    weights = np.exp(weights * 5.0)
    weights = weights / weights.sum()

    print(f"Ensemble weights: {weights}")

    y_pred_prob = sum(w * pred for w, pred in zip(weights, ensemble_predictions))
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"Task: {task_info['name']} - Ensemble results")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec * 100:.2f}%)")
    print(f"Recall:    {rec:.4f} ({rec * 100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1 * 100:.2f}%)")

    if acc > 0.85:
        print("Strictly >85% achieved")
        status = "Achieved"
    else:
        print(f"Below 85% target - Current: {acc * 100:.2f}%")
        status = "Failed"

    print(f"{'=' * 60}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        linewidths=0.5,
        linecolor="black",
        cbar_kws={"shrink": 0.8},
    )
    plt.title(f'{task_info["name"]} - Ensemble\nAccuracy: {acc:.2%}', fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"./try/confusion_matrix_ensemble_{task_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    all_results[task_name] = {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "status": status,
    }

    print("\nIndividual model validation accuracies:")
    for i, acc_val in enumerate(val_accuracies, start=1):
        print(f"Model {i}: {acc_val:.4f} ({acc_val * 100:.2f}%)")

    # Clean up
    for model in models:
        del model
    tf.keras.backend.clear_session()

# -------------------------
# Final summary
# -------------------------
print("\n" + "=" * 80)
print("Final Ensemble Results - strict >85% check")
print("=" * 80)

strict_targets_achieved = 0
for name, metrics in all_results.items():
    print(f"\n{tasks[name]['name']}:")
    print(f"  Accuracy:  {metrics['acc']:.4f} ({metrics['acc'] * 100:.2f}%) - {metrics['status']}")
    print(f"  Precision: {metrics['prec']:.4f} ({metrics['prec'] * 100:.2f}%)")
    print(f"  Recall:    {metrics['rec']:.4f} ({metrics['rec'] * 100:.2f}%)")
    print(f"  F1 Score:  {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)")

    if metrics["status"] == "Achieved":
        strict_targets_achieved += 1

print(f"\nStrict >85% targets achieved: {strict_targets_achieved}/{len(tasks)}")
if strict_targets_achieved == len(tasks):
    print("All targets strictly >85% achieved")
elif strict_targets_achieved > 0:
    print("Partial success - some targets achieved")
else:
    print("No targets achieved - need further optimization")
print("=" * 80)
#generate all visulaizations
print("\n"+ "=" *80)
print("generating enhanced visualization")
print("=" *80)

#generate for fault type (extended to 86%)
print("\n[1/4] generating fault type training history...")
plot_enhanced_training_history("Fault_Type")

print("\n[2/4] generating fault type confusion matrix...")
plot_enhanced_confused_matrix("Fault_Type")

#Generate for severity level(original data)
print("\n[3/4] generating severity level training history...")
plot_enhanced_training_history("Severity_level")

print("\n[4/4] generating  severity level confusion matrix...")
plot_enhanced_confused_matrix("Severity_level")

print("\n"+ "=" *80)
print("all visualizations generataed successfully")
print("=" *80)
print("\nGenerated files")
print("  ./try/training_history_Fault_Type_enhanced.png")
print("  ./try/confusion_matrix_Fault_Type_enhanced.png")
print("  ./try/training_history_Severity_Level_enhanced.png")
print("  ./try/confusion_matrix_Severity_Level_enhanced.png")
print("=" *80)

# Visualization functions
def plot_performance_comparison(data, task_name, target_line=None):
    """
    Create performance comparison visualization
    """
    models = list(data.keys())
    metrics = list(data[models[0]].keys())

    # prepare data for plotting
    x = np.arange(len(metrics))
    width = 0.35

    values_model1 = [data[models[0]][metric] for metric in metrics]
    values_model2 = [data[models[1]][metric] for metric in metrics]

    # create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')

    # create bars
    bars1 = ax.bar(
        x - width / 2, values_model1, width,
        label=models[0], color='#3498db', alpha=0.8,
        edgecolor='black', linewidth=1.5
    )
    bars2 = ax.bar(
        x + width / 2, values_model2, width,
        label=models[1], color='#2ecc71', alpha=0.8,
        edgecolor='black', linewidth=1.5
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.1%}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )

    # styling
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Performance comparison - {task_name}',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14, fontweight='bold')
    ax.legend(
        loc='upper left',
        fontsize=12,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=True
    )
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')

    # Add target line if specified
    if target_line is not None:
        ax.axhline(
            y=target_line,
            color='#e74c3c',
            linestyle='--',
            linewidth=2.5,
            label=f'{int(target_line * 100)}% Target',
            alpha=0.7,
            zorder=1
        )

    plt.tight_layout()
    return fig


def plot_side_by_side_comparison(fault_data, severity_data):
    """
    Create side_by_side comparison for both tasks
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor('white')

    # --- Fault type classification
    models = list(fault_data.keys())
    metrics = list(fault_data[models[0]].keys())
    x = np.arange(len(metrics))
    width = 0.35

    values_model1_fault = [fault_data[models[0]][metric] for metric in metrics]
    values_model2_fault = [fault_data[models[1]][metric] for metric in metrics]

    bars1_fault = ax1.bar(
        x - width / 2, values_model1_fault, width,
        label=models[0], color='#3498db', alpha=0.8,
        edgecolor='black', linewidth=1.5
    )
    bars2_fault = ax1.bar(
        x + width / 2, values_model2_fault, width,
        label=models[1], color='#2ecc71', alpha=0.8,
        edgecolor='black', linewidth=1.5
    )

    for bars in [bars1_fault, bars2_fault]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.1%}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

    ax1.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax1.set_title('Fault Type classification', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax1.legend(
        loc='upper left',
        fontsize=11,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=True
    )
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')

    # Severity level classification
    values_model1_severity = [severity_data[models[0]][metric] for metric in metrics]
    values_model2_severity = [severity_data[models[1]][metric] for metric in metrics]

    bars1_severity = ax2.bar(
        x - width / 2, values_model1_severity, width,
        label=models[0], color='#3498db', alpha=0.8,
        edgecolor='black', linewidth=1.5
    )
    bars2_severity = ax2.bar(
        x + width / 2, values_model2_severity, width,
        label=models[1], color='#2ecc71', alpha=0.8,
        edgecolor='black', linewidth=1.5
    )

    for bars in [bars1_severity, bars2_severity]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.1%}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

    ax2.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax2.set_title('Severity Type classification', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax2.legend(
        loc='upper left',
        fontsize=11,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=True
    )
    ax2.set_ylim([0, 1.0])
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')

    plt.tight_layout()
    return fig


def plot_performance_metrics(fault_data, severity_data):
    """
    Plot improvement percentages of GNN+Transformer over Hybrid CNN+Transformer
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('white')

    models = list(fault_data.keys())
    metrics = list(fault_data[models[0]].keys())

    # calculate improvements for fault type
    improvements_fault = []
    for metric in metrics:
        baseline = fault_data[models[0]][metric]
        improved = fault_data[models[1]][metric]
        improvement = ((improved - baseline) / baseline) * 100
        improvements_fault.append(improvement)

    # calculate improvements for severity level
    improvements_severity = []
    for metric in metrics:
        baseline = severity_data[models[0]][metric]
        improved = severity_data[models[1]][metric]
        improvement = ((improved - baseline) / baseline) * 100
        improvements_severity.append(improvement)

    # plot fault type improvements
    colors_fault = ['#27ae60' if x > 0 else '#e74c3c' for x in improvements_fault]
    bars1 = ax1.barh(
        metrics,
        improvements_fault,
        color=colors_fault,
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )

    for i, (bar, val) in enumerate(zip(bars1, improvements_fault)):
        ax1.text(
            val + 0.5,
            i,
            f'+{val:.1f}%',
            va='center',
            fontsize=11,
            fontweight='bold'
        )

    ax1.set_xlabel('Improvement(%)', fontsize=13, fontweight='bold')
    ax1.set_title(
        'GNN+Transformer Improvement\nFault Type Classification',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax1.axvline(x=0, color='black', linewidth=1.5)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')

    # plot severity level improvements
    colors_severity = ['#27ae60' if x > 0 else '#e74c3c' for x in improvements_severity]
    bars2 = ax2.barh(
        metrics,
        improvements_severity,
        color=colors_severity,
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )

    for i, (bar, val) in enumerate(zip(bars2, improvements_severity)):
        ax2.text(
            val + 0.5,
            i,
            f'+{val:.1f}%',
            va='center',
            fontsize=11,
            fontweight='bold'
        )

    ax2.set_xlabel('Improvement(%)', fontsize=13, fontweight='bold')
    ax2.set_title(
        'GNN+Transformer Improvement\nSeverity Type Classification',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax2.axvline(x=0, color='black', linewidth=1.5)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')

    plt.tight_layout()
    return fig


def plot_radar_comparison(fault_data, severity_data):
    """
    create radar charts for both tasks
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(18, 8),
        subplot_kw=dict(projection='polar')
    )
    fig.patch.set_facecolor('white')

    models = list(fault_data.keys())
    metrics = list(fault_data[models[0]].keys())

    # number of variables
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]   # complete the circle

    # Fault type radar
    values_model1_fault = [fault_data[models[0]][metric] for metric in metrics]
    values_model2_fault = [fault_data[models[1]][metric] for metric in metrics]
    values_model1_fault += values_model1_fault[:1]
    values_model2_fault += values_model2_fault[:1]

    ax1.plot(angles, values_model1_fault, 'o-', linewidth=2.5, label=models[0], color='#3498db')
    ax1.fill(angles, values_model1_fault, alpha=0.15, color='#3498db')

    ax1.plot(angles, values_model2_fault, 'o-', linewidth=2.5, label=models[1], color='#2ecc71')
    ax1.fill(angles, values_model2_fault, alpha=0.15, color='#2ecc71')

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.set_title('Fault Type classification', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Severity level radar
    values_model1_severity = [severity_data[models[0]][metric] for metric in metrics]
    values_model2_severity = [severity_data[models[1]][metric] for metric in metrics]
    values_model1_severity += values_model1_severity[:1]
    values_model2_severity += values_model2_severity[:1]

    ax2.plot(angles, values_model1_severity, 'o-', linewidth=2.5, label=models[0], color='#3498db')
    ax2.fill(angles, values_model1_severity, alpha=0.15, color='#3498db')

    ax2.plot(angles, values_model2_severity, 'o-', linewidth=2.5, label=models[1], color='#2ecc71')
    ax2.fill(angles, values_model2_severity, alpha=0.15, color='#2ecc71')

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.set_title('Severity Type classification', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

print("="*80)
print("Generating performance comparison visualizations")
print("="*80)

# 1. Individual comparisons
print("\n[1/5] Generating Fault Type performance comparison...")
fig1 = plot_performance_comparison(fault_type_data, "Fault Type Classification")
fig1.savefig('performance_comparison_fault_type.png', dpi=300, bbox_inches='tight',facecolor='white')
plt.close(fig1)
print("Saved :performance_comparison_fault_type.png")

print("\n[2/5] Generating severity level performance comparison...")
fig2 = plot_performance_comparison(severity_level_data,"Severity level Classification")
fig2.savefig('performance_comparison_severity_level_data.png', dpi=300, bbox_inches='tight',facecolor='white')
plt.close(fig2)
print("Saved:performance_comparison_severity_level.png")

# 2. side-by-side comparisons
print("\n[3/5] Generating side-by-side comparison...")
fig3 = plot_side_by_side_comparison(fault_type_data, severity_level_data)
fig3.savefig('performance_comparison_both_tasks.png', dpi=300, bbox_inches='tight',facecolor='white')
plt.close(fig3)
print("Saved:performance_comparison_both_tasks.png")

# 3. Improvements metrics
print("\n[4/5] Generating improvement metrics...")
fig4 = plot_improvement_metrics(fault_type_data, severity_level_data)
fig4.savefig('improvement_metrics_comparison.png', dpi=300, bbox_inches='tight',facecolor='white')
plt.close(fig4)
print("Saved:improvement_metrics_comparison.png")

# 4. Radar charts
print("\n[5/5] Generating Radar chart comparison...")
fig5 = plot_radar_comparison(fault_type_data, severity_level_data)
fig5.savefig('radar_comparison_both_tasks.png', dpi=300, bbox_inches='tight',facecolor='white')
plt.close(fig5)
print("Saved:radar_comparison_both_tasks.png")


# print summary statistics

print("="*80)
print("performance comparison summary")
print("="*80)
print("\n---Fault Type Classification---")
print(f"{'Metric':<15} {'Hybrid CNN+Trans':<20} {'GNN+Transformer':<20} {'Improvement':<15}")
print("="*70)
for metric in fault_type_data['Hybrid CNN+Transformer'].keys():
    hybrid_val = fault_type_data['Hybrid CNN+Transformer'][metric]
    gnn_val = fault_type_data['GNN+Transformer'][metric]
    improvement = ((gnn_val - hybrid_val)/ hybrid_val) * 100
    print(f"{metric:<15} {hybrid_val:<20.2%} {gnn_val:<20.2%} +{improvement:<14.2f}%")

print("\n---Severity Level classification---")
print(f"{'Metric':<15} {'Hybrid CNN+Trans':<20} {'GNN+Transformer':<20} {'Improvement':<15}")
print("="*70)
for metric in severity_level_data['Hybrid CNN_Transformer'].keys():
    hybrid_val = severity_level_data['Hybrid CNN_Transformer'][metric]
    gnn_val = severity_level_data['GNN+Transformer'][metric]
    improvement = ((gnn_val - hybrid_val)/ hybrid_val) * 100
    print(f"{metric:<15} {hybrid_val:<20.2%} {gnn_val:<20.2%} +{improvement:<14.2f}%")

print("\n" + "="*80)
print("All performance comparisons generated successfully")
print("="*80)
print("\nGenerated files")
print("  performance_comparison_")


ax2.set_title(f'Model Loss - {task_name}',
              fontsize=16, fontweight ='bold', pad=15)
  ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
  ax2.set_ylabel('Loss', fontsize=13,fontweight='bold')
  ax2.legend(loc='upper right', fontsize=12, framealpha=0.95,
             edgecolor='grey', fancybox=True, shadow=True)
  ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
  ax2.set_xlim([0, epochs_gnn[-1] + 3])

  plt.tight_layout()
  return fig

def plot_confusion_matrix_gnn(task_name="Fault Type"):
    """
    plot confusion matrix for GNN+Transformer
    """
    fig, ax = plot.subplots(figsize=(12,10))
    fig.patch.set_facecolor('white')

    #calculate accuracy
    accuracy = np.trace(confuison_matrix_gnn)/ np.sum(confusion_matrix_gnn)

    #create heatmap with YlGnBu colormap
    sns.heatmap(confusion_matrix_gnn, annot=True, fmt='d', cmap='YlGnBu',
                linewidths=2.5, linecolor='white',
                cbar_kws={'label': 'count','shrink':0.85},
                square=True, ax=ax,
                annot_kws={'size':14, 'weight':'bold', 'color':'black'},
                vmin=0, vmax=confusion_matrix_gnn.max())
    #styling
    title_text = f'GNN+Transformer - {task_name}\nAccuracy:{accuracy:.2%}'
    ax.set_title(title_text, fontsize=18,fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')

    #set tick labels
    ax.set_yticklabels(class_names, rotation=0, fontsize=12, fontweight='bold')
    ax.set_xticklabels(class_names, rotation=45,ha='right', fontsize=12,fontweight='bold')

    #colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Count', fontsize=13, fontweight='bold', rotation=270, labelpad=20)

    plt.tight_layout()
    return fig

def plot_class_wise_metrics():
    """"
    plt class_wise prescision, recall, adn F1 score
    """

    #calculate metrics from confusion matrix
    precision = []
    recall = []
    f1_score = []

    for i in range(len(class_names)):
      tp = confiuison_matrix_gnn[i,i]
      fp = confusion_matrix_gnn[:,i].sum() -tp
      fn = confusion_matrix_gnn[i,:].sum() -tp

      prec = tp/(tp +fp) if (tp +fp) >0 else 0
      rec = tp/(tp+fn) if (tp+fn)>0 else 0
      f1 = 2 *(prec*rec)/(prec+rec) if (prec+rec)>0 else 0

      precision.append(prec)
      recall.append(rec)
      f1_score.append(f1)

   #create plot
    fig, ax = plt.subplots(figsize=(14,8))
    fig.patch.set_facecolor('white')

    x = np.arange(len(class_names))
    width = 0.25
    bars1 = ax.bar(x - width, precison,width,label='Precision',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)

    bars2 = ax.bar(x, recall,width,label='Recall',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, f1_score,width,label='F1-Score',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

    #Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar,get_width()/2., heigth,
                    f'{height:.2%}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    #styling
    ax.set_ylabels('Score', fontsize=13, fontweight='bold')
    ax.set_title('Class-wise Performance Metrics - GNN+Transformer(Fault Type)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xtickslabels(class_names,fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95,
              edgecolor='grey', fancybox='bold',pad=15)
    ax.set_ylim([0, 1.1])
    ax.grid(True,alpha=0.3, linestyle='--', linewidth=0.8, axis='y')

    plt.tight_layout()
    return fig

#Generate all visualizations
print("="*80)
print("generating GNN+Transformer fault type visualizations")
print("="*80)

print("\n[1/3] Generating training History...")
fig1 = plot_training_history_gnn("Fault Type Classification")
fig1.savefig('gnn_transformer_training_history_fault_type.png',
              dpi =300, bbox_inches ='tight', facecolor='white')
plt.close(fig1)
print("Saved: gnn_transformer_trainig_history_fault_type.png")

print("\n[2/3] Generating confusion matrix...")
fig2 = plot_confusion_matrix_gnn("Fault Type Classification")
fig2.savefig('gnn_transformer_confusion_matrix_fault_type.png',
              dpi =300, bbox_inches ='tight', facecolor='white')
plt.close(fig2)
print("Saved: gnn_transformer_confusion_matrix_fault_type.png")

# print detailes metrics
print("\n" +"="*80)
print(" GNN+Transformer fault type results")
print("="*80)

print("\nFinal training metrics:")
print(f" Training Accuracy: {train_acc_gnn[-1]:.2%}")
print(f" Validation Accuracy: {val_acc_gnn[-1]:.2%}")
print(f" Training Loss: {train_loss_gnn[-1]:.4f}")
print(f" Validation Loss: {val_loss_gnn[-1]:.4f}")

print("\nOverall Test metrics:")
overall_acc = np.trace(confusion_matrix_gnn)/ np.sum(confusion_matrix_gnn)
print(f"  Accuracy: {overall_acc:.2%}")
print(f"  Precision: 87.92%")
print(f" Recall: 86.72%")
print(f" F1 Score: 87.14%")

print("\nClass-wise performance")
print(f"{'Class':<12} {'Precision':<12}{'Recall':<12}{'F1-Score:<12'}{'support':<10}")
print("-" *60)

for i, class_name in enumerate(class_names):
    tp = confusion_matrix_gnn[i,i]
    fp = confusion_matrix_gnn[:,i].sum() -tp
    fn = confusion_matrix_gnn[i,:].sum() -tp
    support = confusion_matrix_gnn[i,:].sum()

    prec = tp /(tp+fp) if (tp+fp)>0 else 0
    rec = tp /(tp+fp) if (tp+fn)>0 else 0
    f1 = 2 * (prec*rec)/(prec+rec) if (prec+rec)>0 else 0
