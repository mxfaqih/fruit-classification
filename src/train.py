from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(model, train_generator, validation_generator, epochs=50):
    checkpoint = ModelCheckpoint('../models/best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping]
    )
    
    return history
