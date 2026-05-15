import os
import cv2
#import tensorflow as tf
import pickle
#import shap
import matplotlib
matplotlib.use('Agg') # to avoid GUI errors
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from fpdf import FPDF
from datetime import datetime
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

np.bool = bool
np.int = int

# =====================
# LOAD MODELS
# =====================
feature_extractor = load_model("mod/feature_extractor_NEW.keras")
scaler = pickle.load(open("mod/scaler_new.pkl", "rb"))
svm_model = pickle.load(open("mod/svm_new.pkl", "rb"))

class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(img)
    img_input = np.expand_dims(img_array, axis=0)
    img_input = efficientnet_preprocess(img_input)

    features = feature_extractor(img_input, training=False)
    features_scaled = scaler.transform(features)

    probs = svm_model.predict_proba(features_scaled)[0]
    pred_class = np.argmax(probs)

    label = class_names[pred_class]
    confidence = probs[pred_class]

    #return label, round(confidence*100, 2), None, None
    #gradcam = make_gradcam(image_path)
    #shap_img = make_shap(image_path)

    return label, round(confidence*100, 2), gradcam, shap_img
    
def make_gradcam(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = efficientnet_preprocess(img_array)

    # نلقاو آخر conv layer
    last_conv_layer = None
    for layer in reversed(feature_extractor.layers):
        if "conv" in layer.name:
            last_conv_layer = layer
            break

    grad_model = tf.keras.models.Model(
        [feature_extractor.inputs],
        [last_conv_layer.output, feature_extractor.output]
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
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # ✔ صححت هنا
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    filename = f"gradcam_{os.path.basename(img_path)}"
    path = os.path.join("static", "uploads", filename)
    cv2.imwrite(path, result)

    return f"uploads/{filename}"

def make_shap(img_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import os
    from PIL import Image
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    img_input = np.expand_dims(img_array, axis=0)
    img_input_preprocessed = efficientnet_preprocess(np.copy(img_input))
    
    # 1. Original Prediction
    features = feature_extractor(img_input_preprocessed, training=False)
    features_scaled = scaler.transform(features)
    probs = svm_model.predict_proba(features_scaled)[0]
    pred_class = np.argmax(probs)
    base_prob = probs[pred_class]
    
    predicted_label = class_names[pred_class]
    
    # 2. Occlusion Sensitivity Map (Real SHAP alternative)
    patch_size = 32
    stride = 32
    
    batch_imgs = []
    coords = []
    
    for y in range(0, 224, stride):
        for x in range(0, 224, stride):
            occluded_img = img_array.copy()
            # mask patch with black
            occluded_img[y:y+patch_size, x:x+patch_size, :] = 0 
            batch_imgs.append(occluded_img)
            coords.append((y, x))
            
    batch_imgs = np.array(batch_imgs)
    batch_input = efficientnet_preprocess(batch_imgs)
    
    batch_features = feature_extractor(batch_input, training=False)
    batch_features_scaled = scaler.transform(batch_features)
    batch_probs = svm_model.predict_proba(batch_features_scaled)
    
    # Drops for the predicted class
    drops = base_prob - batch_probs[:, pred_class]
    
    shap_map = np.zeros((224, 224, 3), dtype=np.float32)
    max_drop = np.max(drops)
    min_drop = np.min(drops)
    
    for i, (y, x) in enumerate(coords):
        drop = drops[i]
        
        # Positive drop means masking HURT the prediction -> Patch is important -> Red
        if drop > 0 and max_drop > 0:
            intensity = drop / max_drop
            if intensity > 0.2: # Keep localized
                shap_map[y:y+patch_size, x:x+patch_size] = [255 * intensity, 0, 0] 
                
        # Negative drop means masking HELPED the prediction -> Patch contradicts -> Blue
        elif drop < 0 and min_drop < 0:
            intensity = drop / min_drop
            if intensity > 0.2:
                shap_map[y:y+patch_size, x:x+patch_size] = [0, 0, 255 * intensity] 
                
    shap_map = shap_map.astype(np.uint8)
    
    mask = cv2.cvtColor(shap_map, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    overlay = img_array.copy()
    blended = cv2.addWeighted(img_array, 0.4, shap_map, 0.8, 0)
    overlay[mask > 0] = blended[mask > 0]
    
    # 3. Create Figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.set_title(f"SHAP Explanation for {predicted_label}", fontsize=14, pad=15)
    ax.axis('off')
    
    plt.tight_layout()
    
    filename = f"shap_{os.path.basename(img_path)}"
    save_path = os.path.join("static", "uploads", filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close(fig)
    
    return f"uploads/{filename}"
    
def generate_pdf_report(patient, doctor, visit):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Alzheimer's Disease Detection Report", ln=1, align="C")
    pdf.ln(10)
    
    # Patient & Doctor Info
    pdf.set_font("Arial", "B", 12)
    pdf.cell(100, 10, "Patient Information:", ln=0)
    pdf.cell(100, 10, "Doctor Information:", ln=1)
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(100, 10, f"Name: {patient.first_name} {patient.last_name}", ln=0)
    pdf.cell(100, 10, f"Name: Dr. {doctor.name}", ln=1)
    
    pdf.cell(100, 10, f"Age: {patient.age}", ln=0)
    pdf.cell(100, 10, f"Date of Analysis: {visit.date.strftime('%Y-%m-%d')}", ln=1)
    
    pdf.cell(100, 10, f"Gender: {patient.gender}", ln=1)
    pdf.ln(10)
    
    # Result
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Diagnostic Results", ln=1)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"Predicted Class: {visit.prediction}", ln=1)
    pdf.cell(200, 10, f"Confidence Score: {visit.confidence}%", ln=1)
    pdf.ln(10)
    
    # Images (XAI)
    # Check if images exist and add them
    y_pos = pdf.get_y()
    
    # Original Image
    try:
        pdf.image(os.path.join("static", visit.image_path), x=10, y=y_pos, w=60)
        pdf.cell(60, 70, "", ln=0) # space for image
    except:
        pass
        
    # GradCAM
    try:
        if visit.gradcam_path:
            pdf.image(os.path.join("static", visit.gradcam_path), x=75, y=y_pos, w=60)
    except:
        pass

    # SHAP
    try:
        if visit.shap_path:
            pdf.image(os.path.join("static", visit.shap_path), x=140, y=y_pos, w=60)
    except:
        pass
        
    pdf.ln(75)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(200, 10, "(Left: Original MRI, Middle: Grad-CAM, Right: SHAP Explainability)", ln=1, align="C")
    
    # Save PDF
    pdf_filename = f"report_{visit.id}.pdf"
    pdf_path = os.path.join("static", "uploads", pdf_filename)
    pdf.output(pdf_path)
    
    return f"uploads/{pdf_filename}"
