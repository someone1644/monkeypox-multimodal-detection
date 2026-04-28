import pandas as pd
import os

# CHANGE THIS if needed
DATASET_TYPE = "val"   # use: train / val / test

# Input CSV
input_csv = f"data/{DATASET_TYPE}_metadata.csv"

# Output CSV (overwrite same file)
output_csv = f"data/{DATASET_TYPE}_metadata.csv"

df = pd.read_csv(input_csv)

def fix_path(row):
    original_path = row['filename']

    # Extract only the image name
    filename = os.path.basename(original_path)

    # Get class (Monkeypox / Others)
    class_name = row['class']

    # Build new relative path
    new_path = f"data/{DATASET_TYPE}/{class_name}/{filename}"

    return new_path

# Apply fix
df['filename'] = df.apply(fix_path, axis=1)

# Save updated CSV
df.to_csv(output_csv, index=False)

print(f"✅ Fixed paths in {output_csv}")