import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. Load model & test dataset
# ------------------------------
img_size = (48, 48)
batch_size = 32

model = tf.keras.models.load_model("emotion_model.h5")

test_ds = tf.keras.utils.image_dataset_from_directory(
    "test",
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    shuffle=False
)

class_names = test_ds.class_names
print("Classes:", class_names)

# ------------------------------
# 2. Make predictions
# ------------------------------
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# ------------------------------
# 3. Classification Report
# ------------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ------------------------------
# 4. Confusion Matrix
# ------------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
