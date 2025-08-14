import os
import random

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from glob import glob
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# -------- Parameters --------
csv_path = 'data/HAM10000_metadata.csv'
img_dir = 'data/HAM10000_images'
BATCH_SIZE = 32
IMG_SIZE = 124
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 50
SEED = 42
pd.set_option('display.max_columns', None)


df = pd.read_csv('data/HAM10000_metadata.csv')

df['age'].fillna((df['age'].mean()), inplace=True)

le = LabelEncoder()
df['label'] = le.fit_transform(df['dx'])


imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(img_dir, '*.jpg'))}

df['path'] = df['image_id'].map(imageid_path_dict.get)

def remove_hairs_rgb(rgb_img, kernel_size=17, thresh=10, inpaint_radius=1):
    """
    rgb_img: uint8 RGB image (H, W, 3)
    kernel_size: size of structuring element for black-hat (odd, ~ hair thickness)
    thresh: binarization threshold for hair mask
    inpaint_radius: radius for cv2.inpaint (in pixels)
    """
    # Work in BGR for OpenCV, then convert back to RGB at the end
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # Hair detection via black-hat on grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Binary mask of hairs
    _, hair_mask = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)

    # Inpaint hairs
    inpainted = cv2.inpaint(bgr, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)

    # Back to RGB
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def load_and_process_image(path):
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img)  # uint8 [0..255]

    # # Kernel tuned to image size (roughly IMG_SIZE/12 is a good start for 124px)
    # k = max(3, (IMG_SIZE // 12) | 1)  # ensure odd
    # arr = remove_hairs_rgb(arr, kernel_size=k, thresh=10, inpaint_radius=1)

    # Normalize to [0,1] for the model
    return arr.astype(np.float32) / 255.0


# Pick 10 random samples from your dataframe
sample_paths = random.sample(df['path'].tolist(), 10)


plt.figure(figsize=(15, 8))

for i, path in enumerate(sample_paths):
    orig = np.asarray(Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
    k = max(3, (IMG_SIZE // 12) | 1)
    clean = remove_hairs_rgb(orig, kernel_size=k, thresh=10, inpaint_radius=1)

    # Original image
    plt.subplot(4, 5, 2*i+1)
    plt.imshow(orig)
    plt.axis('off')
    plt.title(f'Orig {i+1}')

    # Hair-removed image
    plt.subplot(4, 5, 2*i+2)
    plt.imshow(clean)
    plt.axis('off')
    plt.title(f'Clean {i+1}')

plt.tight_layout()
plt.show()

df['image'] = df['path'].map(load_and_process_image)

print(df['image'].map(lambda x: x.shape).value_counts())

gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
trainval_idx, test_idx = next(gss_test.split(df, groups=df['lesion_id']))
trainval_df = df.iloc[trainval_idx]
test_df = df.iloc[test_idx]

# Step 2: Split trainval_df into 70% train and 15% val relative to full dataset
# Since trainval_df is 85% of data, val_size relative to trainval_df = 15/85 ≈ 0.1765
val_size = 0.15 / 0.85  # ≈ 0.1765

gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=SEED)
train_idx, val_idx = next(gss_val.split(trainval_df, groups=trainval_df['lesion_id']))
train_df = trainval_df.iloc[train_idx]
val_df = trainval_df.iloc[val_idx]

# Verify split proportions:
print(f"Train size: {len(train_df)/len(df):.2%}")
print(f"Val size:   {len(val_df)/len(df):.2%}")
print(f"Test size:  {len(test_df)/len(df):.2%}")

# Verify no leakage of lesions between splits:
print("Overlap lesions between train and val:",
      set(train_df['lesion_id']).intersection(set(val_df['lesion_id'])))
print("Overlap lesions between trainval and test:",
      set(trainval_df['lesion_id']).intersection(set(test_df['lesion_id'])))

x_train = np.stack(train_df['image'].values)
x_val = np.stack(val_df['image'].values)
x_test = np.stack(test_df['image'].values)

y_train = train_df['label'].values
y_val = val_df['label'].values
y_test = test_df['label'].values

num_classes = 7  # number of classes

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Flatten images to (n_samples, IMG_SIZE*IMG_SIZE*3)
x_train_flat = x_train.reshape((x_train.shape[0], -1))
y_train_labels = np.argmax(y_train, axis=1)  # from one-hot to class indices

# Apply SMOTE
sm = SMOTE(random_state=SEED)
x_train_res, y_train_res = sm.fit_resample(x_train_flat, y_train_labels)

# Reshape back to image format
x_train_res = x_train_res.reshape((-1, IMG_SIZE, IMG_SIZE, 3))

# One-hot encode labels again
y_train_res = tf.keras.utils.to_categorical(y_train_res, num_classes)

model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
print(model.summary())


def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss for multi-class classification.
    gamma: focusing parameter
    alpha: balance parameter
    """

    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = tf.reduce_sum(weight * cross_entropy, axis=1)
        return tf.reduce_mean(loss)

    return focal_loss_fixed

# Compile the model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) , loss = focal_loss(2, 0.25), metrics=["accuracy", tf.keras.metrics.Recall(name='recall')])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=1e-5)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=np.argmax(y_train, axis=1)  # Convert one-hot back to class indices
)

# Convert to dict format expected by Keras
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

print("Class Weights:", class_weights_dict)

# Train with class weights
history = model.fit(
    datagen.flow(x_train_res, y_train_res, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    verbose=1,
    callbacks=[learning_rate_reduction, early_stop]
)

model.save('baseSmoteFocal.h5')

test_results = model.evaluate(x_test, y_test, verbose=1)
print(f"Test results - Loss: {test_results[0]}, Accuracy: {test_results[1]}, Recall: {test_results[2]}")

# ===== 1. TRAINING METRICS PLOTS =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Accuracy
axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

# Recall
axes[2].plot(history.history['recall'], label='Train Recall')
axes[2].plot(history.history['val_recall'], label='Val Recall')
axes[2].set_title('Recall')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Recall')
axes[2].legend()

plt.tight_layout()
plt.show()

# ===== 2. CONFUSION MATRIX =====
# Predict class labels
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ===== 3. CLASS-WISE METRICS =====
from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4)
print("Classification Report:\n", report)