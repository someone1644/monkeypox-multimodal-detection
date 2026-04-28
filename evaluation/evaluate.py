import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def generate_test_csv():
    test_folders = {
        'Monkeypox': os.path.join(BASE_DIR, "../data/sample_images/Monkeypox"),
        'Others': os.path.join(BASE_DIR, "../data/sample_images/Others")
    }
    data = []
    for class_name, folder_path in test_folders.items():
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    abs_path = os.path.join(folder_path, filename)
                    if class_name == 'Monkeypox':
                        fever, lymph = np.random.choice([0, 1], p=[0.9, 0.1]), np.random.choice([0, 1], p=[0.8, 0.2])
                    else:
                        fever, lymph = np.random.choice([0, 1], p=[0.4, 0.6]), np.random.choice([0, 1], p=[0.1, 0.9])
                    data.append({'filename': abs_path, 'class': class_name, 'fever': fever, 'lymph_nodes': lymph})
    df = pd.DataFrame(data)
    csv_path = os.path.join(BASE_DIR, "../data/test_metadata.csv")
    df.to_csv(csv_path, index=False)
    return df
def generate_confusion_matrix_visual(y_true, y_pred_classes):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        annot_kws={"size": 18},
        xticklabels=['Monkeypox', 'Others'],
        yticklabels=['Monkeypox', 'Others']
    )
    plt.title('Test Set Confusion Matrix - Multi-Modal Xception', fontsize=16, pad=15)
    plt.ylabel('True Diagnosis (Actual)', fontsize=14, labelpad=10)
    plt.xlabel('Predicted Diagnosis (AI)', fontsize=14, labelpad=10)
    plt.tight_layout()
    plt.show()
def run_test_evaluation():
    print("1. Preparing Test Data...")
    test_df = generate_test_csv()
    print("2. Loading Model...")
    model_path = os.path.join(BASE_DIR, "../model/model.h5")
    model = tf.keras.models.load_model(model_path)
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_img_gen = datagen.flow_from_dataframe(
        dataframe=test_df, x_col="filename", y_col="class",
        target_size=(128, 128), batch_size=32, class_mode='binary', shuffle=False
    )
    y_pred, y_true = [], []
    print(f"3. Evaluating {test_img_gen.samples} unseen images...")
    for i in range(len(test_img_gen)):
        images, labels = test_img_gen[i]
        idx = i * 32
        batch_metadata = test_df[['fever', 'lymph_nodes']].iloc[idx: idx + len(images)].values
        preds = model.predict({"image_input": images, "metadata_input": batch_metadata}, verbose=0)
        y_pred.extend(np.round(preds).astype(int).flatten())
        y_true.extend(labels)
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    print("\nFINAL TEST SET REPORT:")
    print(classification_report(y_true, y_pred, target_names=['Monkeypox', 'Others'], zero_division=1))
    generate_confusion_matrix_visual(y_true, y_pred)
if __name__ == "__main__":
    run_test_evaluation()