import kagglehub
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

print(tf.__version__)

def process_data():
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print("Path to dataset files:", path)
    
    # Define class names based on folder structure
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    images = []
    labels = []
    
    # To keep track of one example from each class
    class_examples = {}
    
    # Walk through dataset directory and load images
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(path, 'Training', class_name)
        if os.path.exists(class_path):
            print(f"Processing {class_name} images...")
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    # Load and resize image to 150x150
                    img = load_img(img_path, target_size=(150, 150))
                    # Convert to array and normalize
                    img_array = img_to_array(img) / 255.0
                    images.append(img_array)
                    labels.append(class_idx)
                    
                    # Save one example of each class for visualization
                    if class_name not in class_examples:
                        class_examples[class_name] = img_array
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Convert labels to one-hot encoding for the model
    y_categorical = to_categorical(y, num_classes=len(classes))
    
    # Split into train and validation sets
    X_train, X_val, y_train_cat, y_val_cat = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )
    
    # Also keep the original labels for visualization
    _, _, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset loaded: {len(X_train)} training images, {len(X_val)} validation images")
    print(f"Classes: {classes}")
    
    # Display some sample images
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(X_train[i])
        plt.title(classes[y_train[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('figures/sample_images.png')
    
    # Plot one image from each class
    plt.figure(figsize=(12, 4))
    for i, class_name in enumerate(classes):
        plt.subplot(1, 4, i+1)
        plt.imshow(class_examples[class_name])
        plt.title(class_name)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('figures/class_examples.png')
    
    return X_train, X_val, y_train_cat, y_val_cat, classes

def build_cnn_model(input_shape, num_classes):
    """
    Build and return a CNN model for image classification
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, classes = process_data()
    
    # Get input shape from the data
    input_shape = X_train.shape[1:]
    num_classes = len(classes)
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    # Build and display model
    model = build_cnn_model(input_shape, num_classes)
    model.summary()
    

    history = model.fit(
         X_train, y_train,
         validation_data=(X_val, y_val),
         epochs=5,
         batch_size=32
     )


