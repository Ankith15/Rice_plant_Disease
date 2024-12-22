from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
import random


# Directory paths
base_dir = r"F:\\Rice_leaf_Disease\\rice_disease\data"  # Replace with your dataset folder
target_count = 770 # Desired number of images per class

# Initialize ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

for class_name in os.listdir(base_dir):
    class_dir = os.path.join(base_dir, class_name)
    if os.path.isdir(class_dir):
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
        current_count = len(images)
        
        # If the current class has fewer images than the target
        if current_count < target_count:
            for i in range(target_count - current_count):
                # Select a random image and apply augmentation
                img_path = random.choice(images)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                img = np.expand_dims(img, axis=0)  # Expand dimensions for augmentation
                
                # Generate augmented image
                augmented_img = next(datagen.flow(img, batch_size=1))[0].astype(np.uint8)
                
                # Save the augmented image
                new_file_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_aug_{i}.jpg"
                cv2.imwrite(os.path.join(class_dir, new_file_name), augmented_img)

print("All classes are now balanced to 770 images each using augmentation.")
