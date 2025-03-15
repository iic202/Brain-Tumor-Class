
import kagglehub
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Check TensorFlow version
# print(tf.__version__)

def process_data():

    # Get the dataset from Kaggle
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print("Path to dataset files:", path)
    
    # Define the classes
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    images = [] # Will store all the loaded images
    labels = [] # Will store the labels for each image
    
    # To keep track of one example from each class
    class_examples = {}
    
    # Walk through dataset directory and load images
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(path, 'Training', class_name) # Path to the class directory

        # Check if the class directory exists
        if os.path.exists(class_path):
            print(f"Processing {class_name} images...")
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_name)

                    img = load_img(img_path, target_size=(150, 150)) # Load and resize image to 150x150

                    # Convert image to numpy array and normalize pixel values
                    img_array = img_to_array(img) / 255.0
                    images.append(img_array)
                    labels.append(class_idx)
                    
                    # Save one example of each class for visualization
                    # This is for visualization purposes only
                    if class_name not in class_examples:
                        class_examples[class_name] = img_array
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Convert labels to one-hot encoding for the model
    y_categorical = to_categorical(y, num_classes=len(classes))
    
    # Split into train and validation sets (80% train, 20% validation)
    X_train, X_val, y_train_cat, y_val_cat = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )
    
    # Also keep the original labels for visualization
    _, _, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Print some information about the dataset
    print(f"Dataset loaded: {len(X_train)} training images, {len(X_val)} validation images")
    print(f"Classes: {classes}")
    
    # Plot one image from each class
    # 4 images in total for visualization purposes only
    # Just to make sure the data is loaded correctly and to have a look at the images
    plt.figure(figsize=(12, 4))
    for i, class_name in enumerate(classes):
        plt.subplot(1, 4, i+1)
        plt.imshow(class_examples[class_name])
        plt.title(class_name)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('figures/class_examples.png')
    plt.show()
    
    return X_train, X_val, y_train_cat, y_val_cat, classes

def build_cnn_model(input_shape, num_classes):
    """
    Build and return a CNN model for image classification
    """
    model = Sequential([
        # First convolutional block
        Input(shape=input_shape),
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

    model.summary()
    
    return model

def fit_model(model, X_train, y_train, X_val, y_val):
    """
    Fit the model on the training data and validate on the validation data
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32
    )
    
    return history

def plots():
    # Plot loss history and save file
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.legend()
    plt.savefig('figures/loss_plot.png')
    plt.show()

    # Plot accuracy history and save file
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig('figures/accuracy_plot.png')
    plt.show()

    #  Plot the conffusion matrix and save file 
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    confusion_mtx = tf.math.confusion_matrix(np.argmax(y_val, axis=1), y_pred_classes)
    
        # Create a better visualization with class labels
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Greens)
    plt.colorbar()
    plt.title('Confusion Matrix', fontsize=14)
    
         # Add class labels to the axes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    
    # Add numbers to each cell
    thresh = confusion_mtx.numpy().max() / 2.0
    for i in range(confusion_mtx.shape[0]):
        for j in range(confusion_mtx.shape[1]):
            plt.text(j, i, format(confusion_mtx[i, j].numpy(), 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_mtx[i, j].numpy() > thresh else "black")
    
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png')
    plt.show()
    
if __name__ == "__main__":

    # Load and process the data
    X_train, X_val, y_train, y_val, classes = process_data()
    
    # Get input shape from the data for the model
    input_shape = X_train.shape[1:]
    num_classes = len(classes)
    
    # Print the input shape and number of classes if needed
    # print(f"Input shape: {input_shape}")
    # print(f"Number of classes: {num_classes}")
    
    # Build and display model
    model = build_cnn_model(input_shape, num_classes)
    
    # Fit the model
    history = fit_model(model, X_train, y_train, X_val, y_val)
    
    # Save the model
    model.save('model/brain-tumor-mri.keras')

    # Plot accuracy, loss and confusion matrix
    plots()


