import os
import urllib.request
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image

# --- CONFIGURATION (UPGRADED) ---
DATA_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
DATA_DIR = "GTSRB/Final_Training/Images"
IMG_HEIGHT = 60  # DOUBLED RESOLUTION
IMG_WIDTH = 60   # DOUBLED RESOLUTION
CHANNELS = 3
NUM_CATEGORIES = 43
EPOCHS = 15      # Increased for better learning
BATCH_SIZE = 32

def download_and_extract_data():
    if not os.path.exists("GTSRB"):
        print("📥 Downloading GTSRB Dataset...")
        urllib.request.urlretrieve(DATA_URL, "traffic_data.zip")
        with zipfile.ZipFile("traffic_data.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✅ Dataset Extracted.")
    else:
        print("✅ Dataset found.")

def load_data(data_dir):
    images = []
    labels = []
    print(f"📂 Loading images (Resizing to {IMG_HEIGHT}x{IMG_WIDTH})...")
    for category in range(NUM_CATEGORIES):
        path = os.path.join(data_dir, f"{category:05d}")
        if not os.path.exists(path): continue
        for img_name in os.listdir(path):
            if img_name.endswith(".ppm"):
                try:
                    image = Image.open(os.path.join(path, img_name))
                    image = image.resize((IMG_WIDTH, IMG_HEIGHT)) # High Res
                    images.append(np.array(image))
                    labels.append(category)
                except: pass
    return np.array(images), np.array(labels)

def build_pro_model():
    """Deeper CNN for Higher Resolution"""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        BatchNormalization(),
        MaxPool2D((2, 2)),
        Dropout(0.2),

        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D((2, 2)),
        Dropout(0.2),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D((2, 2)),
        Dropout(0.3),

        # Block 4 (New for 60x60)
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CATEGORIES, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    download_and_extract_data()
    data, labels = load_data(DATA_DIR)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # One-Hot
    y_train = to_categorical(y_train, NUM_CATEGORIES)
    y_test = to_categorical(y_test, NUM_CATEGORIES)
    
    # Augmentation (The Secret Sauce)
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,
        fill_mode="nearest"
    )
    
    model = build_pro_model()
    print(f"🚀 Training High-Res Model on RTX 4060...")
    
    # Train with Augmentation
    history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        epochs=EPOCHS,
                        validation_data=(X_test, y_test))
    
    model.save("traffic_classifier.h5")
    print("✅ SUCCESS: High-Res Model Saved.")