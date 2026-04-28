import os as os
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "../saved_model/model.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Please check the path.")
model = tf.keras.models.load_model(model_path)
def predict_monkeypox(img, has_fever, has_lymph_nodes):
    if img is None:
        return {"Error: Please upload an image": 1.0}, "Awaiting image..."
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    fever_val = 1 if has_fever else 0
    lymph_val = 1 if has_lymph_nodes else 0
    metadata_array = np.array([[fever_val, lymph_val]])
    try:
        prediction_prob = model.predict(
            {"image_input": img_array, "metadata_input": metadata_array},
            verbose=0
        )[0][0]
    except Exception as e:
        return {"Error": 1.0}, f"Model error: {e}"
    if prediction_prob >= 0.5:
        conf = float(prediction_prob)
        results = {"Others (Chickenpox/Measles)": conf, "Monkeypox": 1 - conf}
        alert = "### 🟢 Low Risk for Monkeypox\nClinical presentation aligns more closely with common viral rashes."
    else:
        conf = float(1 - prediction_prob)
        results = {"Monkeypox": conf, "Others (Chickenpox/Measles)": 1 - conf}
        alert = "### 🔴 High Risk for Monkeypox\nImmediate isolation and official laboratory PCR testing is recommended."
    return results, alert
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate")) as dashboard:
    gr.Markdown(
        """
        # Monkeypox Detection Using Deep Learning: An Xception-Based Image Classification Approach
        Upload a high-resolution image of the skin lesion and input the patient's vitals to receive an AI-assisted diagnostic evaluation.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Patient Data Input")
            input_image = gr.Image(type="pil", label="Dermatological Image")
            with gr.Group():
                gr.Markdown("**Clinical Symptoms**")
                fever_check = gr.Checkbox(label="🤒 Patient presents with Fever (>38°C / 100.4°F)")
                lymph_check = gr.Checkbox(label="🩺 Patient presents with Lymphadenopathy (Swollen Lymph Nodes)")
            analyze_btn = gr.Button("Run Diagnostic Scan", variant="primary")
            clear_btn = gr.ClearButton(components=[input_image, fever_check, lymph_check])
        with gr.Column(scale=1):
            gr.Markdown("### AI Analysis Results")
            output_labels = gr.Label(num_top_classes=2, label="Confidence Distribution")
            clinical_alert = gr.Markdown("Awaiting input data...")
    gr.Markdown("---")
    gr.Markdown(
        "*Disclaimer: This AI tool is for academic demonstration purposes only and should not replace professional medical diagnosis.*")
    analyze_btn.click(
        fn=predict_monkeypox,
        inputs=[input_image, fever_check, lymph_check],
        outputs=[output_labels, clinical_alert]
    )
if __name__ == "__main__":
    dashboard.launch()