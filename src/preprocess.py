import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def load_train_data(train_path, img_size=(100, 100)):
    images = []
    labels = []
    for fruit_folder in os.listdir(train_path):
        fruit_path = os.path.join(train_path, fruit_folder)
        if os.path.isdir(fruit_path):
            for img_file in os.listdir(fruit_path):
                img = Image.open(os.path.join(fruit_path, img_file))
                img = img.resize(img_size)
                img = np.array(img) / 255.0  # Normalisasi
                images.append(img)
                labels.append(fruit_folder)
    
    return np.array(images), np.array(labels)

def load_test_data(test_path, img_size=(100, 100)):
    images = []
    filenames = []
    for img_file in os.listdir(test_path):
        if img_file.endswith('.jpg'):
            img = Image.open(os.path.join(test_path, img_file))
            img = img.resize(img_size)
            img = np.array(img) / 255.0  # Normalisasi
            images.append(img)
            filenames.append(img_file)
    
    return np.array(images), np.array(filenames)

def encode_labels(labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    one_hot_encoded = to_categorical(integer_encoded)
    return one_hot_encoded, label_encoder

def display_sample_images(images, labels, n=5):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i])
        plt.title(labels[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_data_generator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, 
                          shear_range=0.15, zoom_range=0.15, horizontal_flip=True, fill_mode='nearest'):
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode,
        validation_split=0.2  # 20% data untuk validasi
    )

def preprocess_data(train_path, test_path, img_size=(100, 100), batch_size=32):
    # Create data generators
    train_datagen = create_data_generator()
    test_datagen = ImageDataGenerator(rescale=1./255)  # Hanya normalisasi untuk data test
    
    # Load and augment training data
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Load test data
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator
