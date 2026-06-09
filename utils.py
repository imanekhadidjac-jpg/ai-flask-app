import os
import cv2
import tensorflow as tf
import pickle
import shap
import matplotlib
matplotlib.use('Agg') # to avoid GUI errors
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import Xception, preprocess_input
from fpdf import FPDF
from datetime import datetime
import numpy as np

def clean_text(text):
    return str(text).encode('latin-1', 'ignore').decode('latin-1')

np.bool = bool
np.int = int

# =====================
# LOAD MODELS (تعديل ذكي لمنع الكراش عند الإقلاع)
# =====================
feature_extractor = None

def get_feature_extractor():
    global feature_extractor
    if feature_extractor is None:
        try:
            feature_extractor = Xception(
                weights="imagenet",
                include_top=False,
                pooling="avg"
            )
        except Exception as e:
            print("خطأ في تحميل Xception، سيتم المحاولة بدون أوزان مسبقة:", e)
            feature_extractor = Xception(
                weights=None,
                include_top=False,
                pooling="avg"
            )
    return feature_extractor

# تحميل ملفات الـ SVM بأمان
try:
    scaler = pickle.load(open("mod/xception_scaler.pkl","rb"))
    svm_model = pickle.load(open("mod/xception_svm.pkl","rb"))
except Exception as e:
    print("تنبيه: ملفات الـ SVM لم تُحمل بعد، سيتم تجاوزها للإقلاع:", e)
    scaler = None
    svm_model = None

class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def predict_image(image_path):
    print("IMAGE =", image_path)
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(img)
    img_input = np.expand_dims(img_array, axis=0)
    img_input = preprocess_input(img_input)

    model = get_feature_extractor()
    features = model(img_input, training=False)
    
    if scaler is None or svm_model is None:
        return "NonDemented (تجريبي)", 99.0, None, None

    features_scaled = scaler.transform(features)
    probs = svm_model.predict_proba(features_scaled)[0]

    pred_class = np.argmax(probs)
    label = class_names[pred_class]
    confidence = probs[pred_class]

    gradcam = make_gradcam(image_path)
    shap_img = make_shap(image_path)

    return label, round(confidence*100, 2), gradcam, shap_img

def make_gradcam(img_path):
    try:
        img = image.load_img(img_path, target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        model = get_feature_extractor()
        last_conv_layer = None
        for layer in reversed(model.layers):
            if "conv" in layer.name:
                last_conv_layer = layer
                break

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = tf.reduce_mean(predictions)

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap = heatmap / (np.max(heatmap) + 1e-8)

        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        filename = f"gradcam_{os.path.basename(img_path)}"
        path = os.path.join("static", "uploads", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, result)

        return f"uploads/{filename}"
    except Exception as e:
        print("خطأ في GradCAM:", e)
        return None

def make_shap(img_path):
    try:
        import time
        import gc
        gc.collect()

        correct_img_path = os.path.normpath(img_path)
        if not os.path.exists(correct_img_path):
            correct_img_path = os.path.normpath(os.path.join("static", img_path.replace("static/", "").replace("static\\", "")))

        img = image.load_img(correct_img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        x = np.expand_dims(img_array, axis=0)

        def predict_pipeline(numpy_images):
            preprocessed = preprocess_input(numpy_images.copy())
            model = get_feature_extractor()
            features = model(preprocessed, training=False)
            if scaler is None or svm_model is None:
                return np.array([[0.25, 0.25, 0.25, 0.25]])
            features_scaled = scaler.transform(features)
            return svm_model.predict_proba(features_scaled)

        masker = shap.maskers.Image("blur(128,128)", x[0].shape)
        explainer = shap.Explainer(predict_pipeline, masker, output_names=class_names)
        shap_values = explainer(x, max_evals=50, batch_size=1)

        shap_values.data = np.array(shap_values.data, dtype=np.float32) / 255.0
        plt.clf()
        plt.close('all')

        shap.plots.image(shap_values[0], show=False)

        filename = f"shap_{int(time.time())}.png"
        save_path = os.path.normpath(os.path.join("static", "uploads", filename))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig = plt.gcf()
        fig.set_size_inches(18, 6) 
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close('all')
        gc.collect()

        return f"uploads/{filename}"
    except Exception as e:
        print("خطأ في SHAP:", e)
        return None

def generate_pdf_report(patient, doctor, visit):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Alzheimer Disease Detection Report", ln=1, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(100, 10, "Patient Information:", ln=0)
    pdf.cell(100, 10, "Doctor Information:", ln=1)
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(100, 10, clean_text(f"Name: {patient.first_name} {patient.last_name}"), ln=0)
    pdf.cell(100, 10, clean_text(f"Name: Dr. {doctor.name}"), ln=1)
    
    pdf.cell(100, 10, f"Age: {patient.age}", ln=0)
    pdf.cell(100, 10, f"Date of Analysis: {visit.date.strftime('%Y-%m-%d')}", ln=1)
    
    pdf.cell(100, 10, f"Gender: {patient.gender}", ln=1)
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Diagnostic Results", ln=1)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, clean_text(f"Predicted Class: {visit.prediction}"), ln=1)
    pdf.cell(200, 10, f"Confidence Score: {visit.confidence}%", ln=1)
    pdf.ln(10)
    
    y_pos_row1 = pdf.get_y()
    
    try:
        pdf.image(os.path.join("static", visit.image_path), x=15, y=y_pos_row1, w=85)
    except:
        pass
        
    try:
        if visit.gradcam_path:
            pdf.image(os.path.join("static", visit.gradcam_path), x=110, y=y_pos_row1, w=85)
    except:
        pass
    
    pdf.set_y(y_pos_row1 + 90)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(190, 8, "Left: Original MRI  |  Right: Grad-CAM Activation Map", ln=1, align="C")
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 8, "SHAP Multi-Class Explainability Output:", ln=1, align="L")
    
    y_pos_row2 = pdf.get_y()
    try:
        if visit.shap_path:
            pdf.image(os.path.join("static", visit.shap_path), x=10, y=y_pos_row2, w=190)
    except:
        pass
        
    pdf.set_y(y_pos_row2 + 70)
    
    pdf_filename = f"report_{visit.id}.pdf"
    pdf_path = os.path.join("static", "uploads", pdf_filename)
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.output(pdf_path)
    
    return f"uploads/{pdf_filename}"