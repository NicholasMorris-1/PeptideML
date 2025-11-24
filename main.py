"""ML Pipeline for Peptide Coupling Quality Prediction"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

df = pd.read_csv('peptide_data.csv')

df['pre-chain'] = df['pre-chain'].fillna("").astype(str)
df['amino_acid'] = df['amino_acid'].fillna("").astype(str)

df['first_diff_norm'] = df.groupby('serial')['first_diff'].transform(
    lambda x: (x - x.mean()) / x.std()
)

def categorize_first_diff(x):
    if x <= -0.9:
        return "Excellent"
    elif x <= -0.7:
        return "Good"
    elif x <= -0.5:
        return "Moderate"
    elif x <= -0.2:
        return "Poor"
    else:
        return "Failed"

df['coupling_quality'] = df['first_diff'].apply(categorize_first_diff)

label_enc = LabelEncoder()
df['quality_encoded'] = label_enc.fit_transform(df['coupling_quality'])

# Amino acid properties
aa_props = {
    'A': [1.8,  0, 89.1],
    'C': [2.5,  0, 121.2],
    'D': [-3.5, -1, 133.1],
    'E': [-3.5, -1, 147.1],
    'F': [2.8,  0, 165.2],
    'G': [-0.4, 0, 75.1],
    'H': [-3.2,  1, 155.2],
    'I': [4.5,  0, 131.2],
    'K': [-3.9,  1, 146.2],
    'L': [3.8,  0, 131.2],
    'M': [1.9,  0, 149.2],
    'N': [-3.5,  0, 132.1],
    'P': [-1.6,  0, 115.1],
    'Q': [-3.5,  0, 146.2],
    'R': [-4.5,  1, 174.2],
    'S': [-0.8,  0, 105.1],
    'T': [-0.7,  0, 119.1],
    'V': [4.2,  0, 117.1],
    'W': [-0.9,  0, 204.2],
    'Y': [-1.3,  0, 181.2]
}

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_sequence_features(seq):
    if len(seq) == 0:
        return [0, 0, 0, 0, 0, 0, 0]
    props = np.array([aa_props.get(a, [0,0,0]) for a in seq])
    hydro, charge, mw = props[:,0], props[:,1], props[:,2]
    return [
        np.mean(hydro), np.std(hydro), np.sum(hydro),
        np.mean(charge), np.sum(charge),
        np.mean(mw), np.sum(mw)
    ]

features = []
for _, row in df.iterrows():
    pre = row['pre-chain']
    aa = row['amino_acid']

    pre_feats = compute_sequence_features(pre)
    aa_feats = aa_props.get(aa, [0,0,0])

    seq_len = len(pre)
    pre_hydro_mean = pre_feats[0]
    aa_hydro = aa_feats[0]
    delta_hydro = aa_hydro - pre_hydro_mean

    features.append(pre_feats + aa_feats + [seq_len, delta_hydro])

feat_names = [
    'pre_hydro_mean','pre_hydro_std','pre_hydro_sum',
    'pre_charge_mean','pre_charge_sum',
    'pre_mw_mean','pre_mw_sum',
    'aa_hydro','aa_charge','aa_mw',
    'seq_length','delta_hydro'
]

X = pd.DataFrame(features, columns=feat_names)
y = df['quality_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 1. CLASSICAL ML MODELS COMPARISON
# =============================================================================

print("=" * 80)
print("CLASSICAL ML MODELS COMPARISON")
print("=" * 80)

models = {
    'Random Forest (Improved)': RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=False
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
    'SVM (RBF)': SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=7,
        weights='distance'
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=200,
        random_state=42
    ),
    'Naive Bayes': GaussianNB()
}

results = []

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 50)

    # Use scaled data for distance-based models
    if name in ['SVM (RBF)', 'Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    # Train model
    model.fit(X_tr, y_train)

    # Predictions
    y_pred = model.predict(X_te)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    results.append({
        'Model': name,
        'Test Accuracy': acc,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_enc.classes_))

# =============================================================================
# 2. RESULTS SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("MODELS SUMMARY (SORTED BY TEST ACCURACY)")
print("=" * 80)

results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
print(results_df.to_string(index=False))

# =============================================================================
# 3. DEEP LEARNING APPROACH WITH TENSORFLOW
# =============================================================================

print("\n" + "=" * 80)
print("DEEP LEARNING MODEL (TensorFlow)")
print("=" * 80)

# Prepare data for deep learning
X_train_dl = X_train_scaled
X_test_dl = X_test_scaled
y_train_dl = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test_dl = tf.keras.utils.to_categorical(y_test, num_classes=5)

# Build neural network
def create_model(input_dim, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        # First block
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Second block
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Third block
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Create and train model
dl_model = create_model(X_train_dl.shape[1], 5)

print("\nModel Architecture:")
dl_model.summary()

# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

# Train
print("\nTraining Deep Learning Model...")
history = dl_model.fit(
    X_train_dl, y_train_dl,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate
y_pred_dl = dl_model.predict(X_test_dl)
y_pred_classes = np.argmax(y_pred_dl, axis=1)
y_test_classes = np.argmax(y_test_dl, axis=1)

dl_accuracy = accuracy_score(y_test_classes, y_pred_classes)

print(f"\nDeep Learning Test Accuracy: {dl_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=label_enc.classes_))

# =============================================================================
# 4. VISUALIZATIONS
# =============================================================================

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('dl_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Compare all models
all_results = results + [{
    'Model': 'Deep Learning (TensorFlow)',
    'Test Accuracy': dl_accuracy,
    'CV Mean': None,
    'CV Std': None
}]

comparison_df = pd.DataFrame(all_results).sort_values('Test Accuracy', ascending=False)

plt.figure(figsize=(12, 6))
plt.barh(comparison_df['Model'], comparison_df['Test Accuracy'])
plt.xlabel('Test Accuracy')
plt.title('Model Performance Comparison')
plt.xlim([0, 1])
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nTop 3 Models by Test Accuracy:")
print(comparison_df[['Model', 'Test Accuracy']].head(3).to_string(index=False))
