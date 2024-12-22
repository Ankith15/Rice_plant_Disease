<<<<<<< HEAD
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# Define paths
input_dir = r"F:\\Rice_leaf_Disease\\rice_disease\\data"  # Original dataset folder
output_dir = r"F:\\Rice_leaf_Disease\\rice_disease\\aug_data"  # Folder to save augmented images
os.makedirs(output_dir, exist_ok=True)

# Augmentation configuration
datagen = ImageDataGenerator(
    rotation_range=30,
=======
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
>>>>>>> 24e7bc881ef71f707cdde85538050e41106027f0
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

<<<<<<< HEAD
# Number of augmented images per original image
num_augmented_images = 2  # Adjust this to the desired number

# Iterate through each class folder
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    print(f"Processing class: {class_name}")
    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    # Process each image in the class folder
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        if not os.path.isfile(image_path):
            continue
        
        # Load and preprocess the image
        img = load_img(image_path, target_size=(150, 150))  # Resize to 150x150
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)  # Reshape for augmentation
        
        # Generate and save augmented images
        count = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_class_path, 
                                  save_prefix="aug", save_format="jpeg"):
            count += 1
            if count >= num_augmented_images:
                break

    print(f"Augmented images for class '{class_name}' saved to {output_class_path}")

print("Data augmentation completed successfully!")
=======
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
>>>>>>> 24e7bc881ef71f707cdde85538050e41106027f0
