from src.preprocess import preprocess_data
from src.model import create_model
from src.train import train_model
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    train_path = "data/train/"
    test_path = "data/test/"
    
    train_generator, validation_generator, test_generator = preprocess_data(train_path, test_path)
    
    num_classes = len(train_generator.class_indices)
    input_shape = train_generator.image_shape
    
    model = create_model(input_shape=input_shape, num_classes=num_classes)
    
    history = train_model(model, train_generator, validation_generator, epochs=50)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    # Evaluate model on test data
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Make predictions on test data
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Display some test images with predictions
    plt.figure(figsize=(15, 10))
    for i in range(15):
        plt.subplot(3, 5, i+1)
        img = test_generator[i][0][0]  # Get the first image from the batch
        plt.imshow(img)
        true_label = class_labels[true_classes[i]]
        pred_label = class_labels[predicted_classes[i]]
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
