import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

test_dir = '../dataset/test'
img_width, img_height = 299, 299

# Create data classes
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
#    target_size=(img_width, img_height),
#    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Load the model
model = load_model('inceptionv3_model.h5')

# Load the image to be classified
img_path = '../dataset/test/Dynamo/IMG_20230301_011435.jpg_0.jpg'
img = image.load_img(img_path, target_size=(img_width, img_height))

# Preprocess the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# Predict the class of the image
preds = model.predict(x)
predicted_class_indices = np.argmax(preds, axis=1)
class_labels = list(test_generator.class_indices.keys())
predicted_classes = [class_labels[i] for i in predicted_class_indices]
print(predicted_classes)
