import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# ------------------------------
# 1. Parameters
# ------------------------------
img_size = (48, 48)
batch_size = 32
epochs = 25

# ------------------------------
# 2. Load Dataset
# ------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    "train",
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale"   # since emotions dataset is usually gray images
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "test",
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale"
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ------------------------------
# 3. Prefetching for performance
# ------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ------------------------------
# 4. Build CNN Model
# ------------------------------
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(48, 48, 1)),  # normalize

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')  # output layer
])

# ------------------------------
# 5. Compile Model
# ------------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------
# 6. Train Model
# ------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# ------------------------------
# 7. Save Model
# ------------------------------
model.save("emotion_model.h5")
print("✅ Model saved as emotion_model.h5")
