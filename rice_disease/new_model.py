import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.utils import class_weight

# Paths to the pre-augmented datasets
train_dir = "F:/Rice_leaf_Disease/split_leaf/train"
val_dir = "F:/Rice_leaf_Disease/split_leaf/validation"
test_dir = "F:/Rice_leaf_Disease/split_leaf/test"

# Constants
img_size = (224, 224)
batch_size = 32
num_classes = 4
input_shape = (224, 224, 3)

# Data Generators (No Augmentation, Only Rescaling)
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
)

# Debugging the generator output
x_batch, y_batch = next(train_generator)
print(f"Batch images shape: {x_batch.shape}")
print(f"Batch labels shape: {y_batch.shape}")

# Calculate class weights based on the training data
class_sample_count = np.array([train_generator.class_indices[label] for label in train_generator.class_indices.keys()])
class_weight_dict = class_weight.compute_class_weight(
    'balanced', 
    classes=np.unique(class_sample_count), 
    y=train_generator.classes
)
class_weights = dict(zip(train_generator.class_indices.values(), class_weight_dict))

# Model Definition
model = Sequential()
densenet_base = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
model.add(densenet_base)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(num_classes, activation='softmax'))

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model with Class Weights
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size,
    class_weight=class_weights  # Apply class weights
)

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the Model
model.save("Disease_dtct.h5")
print("Model saved successfully!")
