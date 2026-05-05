# ============================================================
# Pneumonia Detection from Chest X-Rays
# A Comparative Study: Custom CNN vs. ResNet50 (Transfer Learning)
# Author: Moparthi Aparna (ID: 700772027)
# ============================================================

import os
import json
import shutil
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1. Kaggle Authentication (using provided credentials)
# ------------------------------
# Replace with your Kaggle username and API key if different
KAGGLE_USERNAME = "Appuash"
KAGGLE_KEY = "KGAT_6128e8b6328f17e628fd4da860c153bd"

# Create .kaggle directory and store credentials
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
!chmod 600 ~/.kaggle/kaggle.json   # This line works in Colab; for local, use os.chmod

# Install required packages (uncomment if needed)
# !pip install -q kaggle tensorflow scikit-learn pandas

# ------------------------------
# 2. Download Dataset (if not already present)
# ------------------------------
if not os.path.exists('chest_xray_raw/chest_xray/train'):
    !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
    !unzip -q chest-xray-pneumonia.zip -d chest_xray_raw

# ------------------------------
# 3. Create 50/50 Train/Validation Split (fix original 16‑image validation set)
# ------------------------------
from sklearn.model_selection import train_test_split

original_train_dir = 'chest_xray_raw/chest_xray/train'
new_train_dir = 'chest_xray_split_50/train'
new_val_dir = 'chest_xray_split_50/val'
test_dir = 'chest_xray_raw/chest_xray/test'

# Remove old splits if they exist
shutil.rmtree(new_train_dir, ignore_errors=True)
shutil.rmtree(new_val_dir, ignore_errors=True)

# Stratified split per class (NORMAL and PNEUMONIA)
for class_name in os.listdir(original_train_dir):
    class_path = os.path.join(original_train_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    images = os.listdir(class_path)
    train_imgs, val_imgs = train_test_split(images, test_size=0.5, random_state=42)

    os.makedirs(os.path.join(new_train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(new_val_dir, class_name), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(new_train_dir, class_name, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(new_val_dir, class_name, img))

print(f"Train dir: {new_train_dir}, Val dir: {new_val_dir}, Test dir: {test_dir}")

# ------------------------------
# 4. Data Generators (with augmentation)
# ------------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Check GPU availability
print("GPU available:", tf.config.list_physical_devices('GPU'))

batch_size = 32
img_size = (224, 224)

# Data augmentation for training (only)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    new_train_dir, target_size=img_size, batch_size=batch_size,
    class_mode='binary', shuffle=True
)
val_gen = val_test_datagen.flow_from_directory(
    new_val_dir, target_size=img_size, batch_size=batch_size,
    class_mode='binary', shuffle=False
)
test_gen = val_test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size,
    class_mode='binary', shuffle=False
)

print(f"Train: {train_gen.samples}, Val: {val_gen.samples}, Test: {test_gen.samples}")

# ------------------------------
# 5. Model 1: Custom CNN (from scratch)
# ------------------------------
model_cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer='l2'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

print(model_cnn.summary())

print("Training Custom CNN (10 epochs)...")
history_cnn = model_cnn.fit(train_gen, epochs=10, validation_data=val_gen, verbose=1)

# Evaluate custom CNN on test set
y_true = test_gen.classes
y_pred_cnn_prob = model_cnn.predict(test_gen, verbose=0).flatten()
y_pred_cnn = (y_pred_cnn_prob > 0.5).astype(int)

acc_cnn = accuracy_score(y_true, y_pred_cnn)
prec_cnn = precision_score(y_true, y_pred_cnn)
rec_cnn = recall_score(y_true, y_pred_cnn)
f1_cnn = f1_score(y_true, y_pred_cnn)
auc_cnn = roc_auc_score(y_true, y_pred_cnn_prob)

print(f"\nCustom CNN → Acc: {acc_cnn:.4f}, Prec: {prec_cnn:.4f}, Rec: {rec_cnn:.4f}, F1: {f1_cnn:.4f}, AUC: {auc_cnn:.4f}")

# ------------------------------
# 6. Model 2: ResNet50 with Transfer Learning (two‑phase fine‑tuning)
# ------------------------------
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Separate generators with ResNet preprocessing
train_gen_resnet = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
).flow_from_directory(new_train_dir, target_size=img_size, batch_size=batch_size,
                      class_mode='binary', shuffle=True)

val_gen_resnet = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    new_val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

test_gen_resnet = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

# Load pretrained ResNet50 without top
base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False  # freeze base for phase 1

resnet_model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

resnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nPhase 1: Train only top layers (5 epochs)...")
resnet_model.fit(train_gen_resnet, epochs=5, validation_data=val_gen_resnet, verbose=1)

# Phase 2: unfreeze last 20 layers for fine‑tuning
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

resnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                     loss='binary_crossentropy', metrics=['accuracy'])

print("\nPhase 2: Fine‑tune last 20 layers (5 more epochs)...")
resnet_model.fit(train_gen_resnet, epochs=5, validation_data=val_gen_resnet, verbose=1)

# Evaluate ResNet50 on test set
y_pred_resnet_prob = resnet_model.predict(test_gen_resnet, verbose=0).flatten()
y_pred_resnet = (y_pred_resnet_prob > 0.5).astype(int)

acc_res = accuracy_score(y_true, y_pred_resnet)
prec_res = precision_score(y_true, y_pred_resnet)
rec_res = recall_score(y_true, y_pred_resnet)
f1_res = f1_score(y_true, y_pred_resnet)
auc_res = roc_auc_score(y_true, y_pred_resnet_prob)

print(f"\nResNet50 (fine‑tuned) → Acc: {acc_res:.4f}, Prec: {prec_res:.4f}, Rec: {rec_res:.4f}, F1: {f1_res:.4f}, AUC: {auc_res:.4f}")

# ------------------------------
# 7. Final Comparison Table
# ------------------------------
import pandas as pd
comparison = pd.DataFrame({
    'Model': ['Custom CNN', 'ResNet50 (fine‑tuned)'],
    'Accuracy': [acc_cnn, acc_res],
    'Precision': [prec_cnn, prec_res],
    'Recall': [rec_cnn, rec_res],
    'F1 Score': [f1_cnn, f1_res],
    'AUC-ROC': [auc_cnn, auc_res]
})

print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)
print(comparison.to_string(index=False))

# Save comparison to CSV for report
comparison.to_csv('model_comparison_final.csv', index=False)
print("\nResults saved to 'model_comparison_final.csv'")
print("✅ Project completed successfully.")