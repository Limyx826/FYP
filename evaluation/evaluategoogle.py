from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Define the paths to the dataset directories
test_dir = '../dataset/test'

# Define image size and batch size
img_width, img_height = 299, 299
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
model_path = '../googlenet_model.h5'
model = load_model(model_path)
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)

