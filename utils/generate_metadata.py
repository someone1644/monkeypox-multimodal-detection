import os
import pandas as pd
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def generate_metadata_csvs():
    base_output_dir = os.path.join(BASE_DIR, "../")
    train_folders = {
        'Monkeypox': os.path.join(BASE_DIR, "../data/train/Monkeypox"),
        'Others': os.path.join(BASE_DIR, "../data/train/Others")
    }
    val_folders = {
        'Monkeypox': os.path.join(BASE_DIR, "../data/val/Monkeypox"),
        'Others': os.path.join(BASE_DIR, "../data/val/Others")
    }
    def process_folders(folders_dict, output_filename):
        data = []
        for class_name, folder_path in folders_dict.items():
            if not os.path.exists(folder_path):
                print(f"Warning: Directory not found - {folder_path}")
                continue
            print(f"Scanning {class_name} images in: {folder_path}")
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    abs_path = os.path.join(folder_path, filename)
                    if class_name == 'Monkeypox':
                        fever = np.random.choice([0, 1], p=[0.1, 0.9])
                        lymph = np.random.choice([0, 1], p=[0.2, 0.8])
                    else:
                        fever = np.random.choice([0, 1], p=[0.6, 0.4])
                        lymph = np.random.choice([0, 1], p=[0.9, 0.1])
                    data.append({
                        'filename': abs_path,
                        'class': class_name,
                        'fever': fever,
                        'lymph_nodes': lymph
                    })
        df = pd.DataFrame(data)
        csv_path = os.path.join(base_output_dir, f"{output_filename}.csv")
        df.to_csv(csv_path, index=False)
        print(f"--> Successfully created {csv_path} with {len(df)} records.\n")
        return df
    print("--- Generating Training Metadata ---")
    process_folders(train_folders, "train_metadata")
    print("--- Generating Validation Metadata ---")
    process_folders(val_folders, "val_metadata")
if __name__ == "__main__":
    generate_metadata_csvs()