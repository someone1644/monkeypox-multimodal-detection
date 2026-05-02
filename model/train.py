import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def build_xception_model_with_metadata(image_size=(128, 128, 3), dropout_rate=0.5):
    image_input = Input(shape=image_size, name="image_input")
    base_model = Xception(weights='imagenet', include_top=False, input_shape=image_size)
    base_model.trainable = False
    x = base_model(image_input, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    metadata_input = Input(shape=(2,), name="metadata_input")
    combined = Concatenate()([x, metadata_input])
    combined_output = Dense(1, activation='sigmoid')(combined)
    model = Model(inputs=[image_input, metadata_input], outputs=combined_output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
def create_multi_input_generators(train_df, val_df, image_size=(128, 128), batch_size=32):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_img_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="class",
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    val_img_gen = datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filename",
        y_col="class",
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    print("Train samples:", train_img_gen.samples)
    print("Validation samples:", val_img_gen.samples)
    def multi_generator(image_gen, df):
        metadata_cols = ['fever', 'lymph_nodes']
        while True:
            images, labels = next(image_gen)
            idx = (image_gen.batch_index - 1) * image_gen.batch_size
            batch_metadata = df[metadata_cols].iloc[idx: idx + len(images)].values
            yield {"image_input": images, "metadata_input": batch_metadata}, labels
    return multi_generator(train_img_gen, train_df), multi_generator(val_img_gen,val_df), train_img_gen.samples, val_img_gen.samples
def plot_metrics(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
def evaluate_model(model, val_df, batch_size=32, image_size=(128, 128)):
    print("\n--- Evaluating Model ---")
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_img_gen = datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filename",
        y_col="class",
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    y_pred = []
    y_true = []
    metadata_cols = ['fever', 'lymph_nodes']
    print(f"Generating predictions for all {val_img_gen.samples} validation images...")
    for i in range(len(val_img_gen)):
        images, labels = val_img_gen[i]
        idx = i * batch_size
        batch_metadata = val_df[metadata_cols].iloc[idx: idx + len(images)].values
        preds = model.predict({"image_input": images, "metadata_input": batch_metadata}, verbose=0)
        y_pred.extend(np.round(preds).astype(int).flatten())
        y_true.extend(labels)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Monkeypox', 'Others'],
                yticklabels=['Monkeypox', 'Others'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Monkeypox', 'Others'], zero_division=1))
def main():
    print("1. Loading CSV Metadata...")
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
    train_csv_path = os.path.join(PROJECT_ROOT, "train_metadata.csv")
    val_csv_path = os.path.join(PROJECT_ROOT, "val_metadata.csv")
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    train_df["filename"] = train_df["filename"].apply(
        lambda x: os.path.join(PROJECT_ROOT, x.replace("..\\", "").replace("../", ""))
    )
    val_df["filename"] = val_df["filename"].apply(
        lambda x: os.path.join(PROJECT_ROOT, x.replace("..\\", "").replace("../", ""))
    )
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("2. Initializing Multi-Input Generators...")
    batch_size = 32
    train_gen, val_gen, train_steps, val_steps = create_multi_input_generators(train_df, val_df, batch_size=batch_size)
    print("3. Building Xception Multi-Modal Network...")
    model = build_xception_model_with_metadata(image_size=(128, 128, 3))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5, verbose=1)
    ]
    print("4. Starting Model Training...")
    epochs = 20
    history = model.fit(
        train_gen,
        steps_per_epoch=max(1,train_steps // batch_size),
        validation_data=val_gen,
        validation_steps=max(1,val_steps // batch_size),
        epochs=epochs,
        callbacks=callbacks
    )
    print("\n5. Training Complete! Generating Visualizations...")
    plot_metrics(history)
    evaluate_model(model, val_df, batch_size=batch_size)
    print("\n6. Saving Model...")
    save_dir = os.path.join(BASE_DIR, "../saved_model")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_save_path = os.path.join(save_dir, "model.h5")
    model.save(model_save_path)
    print(f"Model successfully saved to: {model_save_path}")
if __name__ == "__main__":
    main()