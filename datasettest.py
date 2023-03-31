from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the paths to the dataset directories
test_dir = 'dataset/test'

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

class_names = list(test_generator.class_indices.keys())

print(class_names)