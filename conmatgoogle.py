import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Define the paths to the dataset directories
test_dir = 'dataset/test'

# Define image size and batch size
img_width, img_height = 224, 224
batch_size = 40

# Create data generator for testing
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Load the pre-trained Xception model and make predictions on the test set
model_path = 'googlenet_model.h5'
model = load_model(model_path)
test_predictions = model.predict(test_generator, steps=test_generator.samples // batch_size)

# Convert predictions from probabilities to class labels
test_pred_classes = np.argmax(test_predictions, axis=1)

# Get true labels
test_true_classes = test_generator.classes

# Get class names
class_names = list(test_generator.class_indices.keys())

# Create confusion matrix
confusion_mtx = confusion_matrix(test_true_classes, test_pred_classes)

# Create a DataFrame to hold the confusion matrix
df = pd.DataFrame(confusion_mtx, index=class_names, columns=class_names)
print(df)

fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=df.values, rowLabels=class_names, colLabels=class_names, loc='center')
fig.tight_layout()
plt.show()
