import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from glob import glob
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

# -------- Parameters --------
csv_path = 'data/HAM10000_metadata.csv'
img_dir = 'data/HAM10000_images'
BATCH_SIZE = 32
IMG_SIZE = 224  # DenseNet works best with 224x224
EPOCHS = 50
SEED = 42
pd.set_option('display.max_columns', None)

# ===== 1. LOAD DATA =====
df = pd.read_csv(csv_path)
df['age'].fillna((df['age'].mean()), inplace=True)

le = LabelEncoder()
df['label'] = le.fit_transform(df['dx'])

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(img_dir, '*.jpg'))}

df['path'] = df['image_id'].map(imageid_path_dict.get)

# Preprocess images for DenseNet
def load_and_process_image(path):
    img = Image.open(path).resize((IMG_SIZE, IMG_SIZE))
    img = img.convert('RGB')
    img = np.asarray(img)
    img = preprocess_input(img)
    return img

df['image'] = df['path'].map(load_and_process_image)

# ===== 2. TRAIN/VAL/TEST SPLIT (Group by lesion_id to avoid leakage) =====
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
trainval_idx, test_idx = next(gss_test.split(df, groups=df['lesion_id']))
trainval_df = df.iloc[trainval_idx]
test_df = df.iloc[test_idx]

val_size = 0.15 / 0.85
gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=SEED)
train_idx, val_idx = next(gss_val.split(trainval_df, groups=trainval_df['lesion_id']))
train_df = trainval_df.iloc[train_idx]
val_df = trainval_df.iloc[val_idx]

print(f"Train size: {len(train_df)/len(df):.2%}")
print(f"Val size:   {len(val_df)/len(df):.2%}")
print(f"Test size:  {len(test_df)/len(df):.2%}")

# ===== 3. Convert to arrays =====
x_train = np.stack(train_df['image'].values)
x_val = np.stack(val_df['image'].values)
x_test = np.stack(test_df['image'].values)

y_train = train_df['label'].values
y_val = val_df['label'].values
y_test = test_df['label'].values

num_classes = len(le.classes_)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# ===== 4. Augmentation =====
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)
datagen.fit(x_train)

# ===== 5. Class weights =====
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=np.argmax(y_train, axis=1)
)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}
print("Class Weights:", class_weights_dict)

# ===== 6. Build DenseNet Model =====
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # freeze for initial training

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Recall(name='recall')]
)
print(model.summary())

# ===== 7. Callbacks =====
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=1e-5)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ===== 8. Train frozen backbone =====
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    verbose=1,
    callbacks=[learning_rate_reduction, early_stop],
    class_weight=class_weights_dict
)

# ===== 9. Fine-tune top layers =====
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Recall(name='recall')]
)

fine_tune_history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    epochs=10,
    validation_data=(x_val, y_val),
    verbose=1,
    callbacks=[learning_rate_reduction, early_stop],
    class_weight=class_weights_dict
)

# ===== 10. Save model =====
model.save('DenseNet_HAM10000.h5')

# ===== 11. Evaluate =====
test_results = model.evaluate(x_test, y_test, verbose=1)
print(f"Test results - Loss: {test_results[0]}, Accuracy: {test_results[1]}, Recall: {test_results[2]}")

# ===== 12. Plot training metrics =====
def plot_training(history, title_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss' + title_suffix)
    axes[0].legend()

    axes[1].plot(history.history['accuracy'], label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_title('Accuracy' + title_suffix)
    axes[1].legend()

    axes[2].plot(history.history['recall'], label='Train Recall')
    axes[2].plot(history.history['val_recall'], label='Val Recall')
    axes[2].set_title('Recall' + title_suffix)
    axes[2].legend()

    plt.tight_layout()
    plt.show()
plot_training(history, " (Frozen)")
plot_training(fine_tune_history, " (Fine-tuned)")

# ===== 13. Confusion Matrix & Classification Report =====
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4)
print("Classification Report:\n", report)
