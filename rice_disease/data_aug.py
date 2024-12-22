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
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

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
