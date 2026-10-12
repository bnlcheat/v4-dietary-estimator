import os
import numpy as np
from PIL import Image

# Suppress verbose TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="V4-Pro Dietary Estimator",
    page_icon="🥗",
    layout="wide"
)

st.title("🥗 V4-Pro Dietary Intake Estimator")
st.markdown("Dual-Branch **EfficientNetB0** predicting total mass (log-space) and fractional macronutrient density (softmax).")

# ---------------------------------------------------------
# Architecture Builder & Weights Loader
# ---------------------------------------------------------
@st.cache_resource
def load_system_model():
    # Automatically detect the weights file regardless of what you named it
    if os.path.exists("v4_weights.keras"):
        weights_path = "v4_weights.keras"
    elif os.path.exists("v4_weights.h5"):
        weights_path = "v4_weights.h5"
    elif os.path.exists("model_v4_weights.h5"):
        weights_path = "model_v4_weights.h5"
    else:
        st.error(f"Missing weights file. Please upload your 17.5 MB weights file to GitHub.")
        st.stop()
    
    # Reconstruct the dual-branch network topology
    base_v4 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), dtype=tf.float32, name="plate_input")
    
    data_aug_v4 = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"), 
        tf.keras.layers.RandomRotation(0.08), 
        tf.keras.layers.RandomZoom(0.08),
    ], name="v4_aug")
    x = data_aug_v4(inputs)

    features = base_v4(x, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.BatchNormalization()(features)

    mass_branch = tf.keras.layers.Dense(128, activation='swish')(features)
    mass_branch = tf.keras.layers.Dropout(0.2)(mass_branch)
    pred_log_mass = tf.keras.layers.Dense(1, activation='relu', dtype='float32', name='mass_output')(mass_branch)

    comp_branch = tf.keras.layers.Dense(128, activation='swish')(features)
    comp_branch = tf.keras.layers.Dropout(0.2)(comp_branch)
    pred_fractions = tf.keras.layers.Dense(4, activation='softmax', dtype='float32', name='fractions_output')(comp_branch)

    v4_pro_output = tf.keras.layers.Concatenate(axis=-1, name='final_output')([pred_log_mass, pred_fractions])
    model = tf.keras.models.Model(inputs=inputs, outputs=v4_pro_output)
    
    # Load weights
    model.load_weights(weights_path)
    return model

# ---------------------------------------------------------
# Image Upload & V4 Inference Pipeline
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a top-down meal photo (JPEG, PNG)...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1.2])

    with col1:
        try:
            st.image(img, caption="Input Meal Photo", use_container_width=True)
        except TypeError:
            st.image(img, caption="Input Meal Photo", use_column_width=True)

    with col2:
        with st.spinner("Decoding dual-branch network predictions..."):
            img_arr = np.array(img.resize((224, 224)), dtype=np.float32)
            img_input = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_arr, axis=0))

            raw_pred = model.predict(img_input, verbose=0)
            pred = np.array(raw_pred).squeeze()
            if pred.ndim > 1:
                pred = pred[0]
            
            # Unpack Branch A (Mass)
            pred_mass = float(np.expm1(pred[0]))
            pred_mass = max(pred_mass, 1.0)

            # Unpack Branch B (Softmax Fractions)
            f_fat = float(pred[1])
            f_carb = float(pred[2])
            f_prot = float(pred[3])
            f_water = float(pred[4])
            
            pred_fat = pred_mass * f_fat
            pred_carb = pred_mass * f_carb
            pred_prot = pred_mass * f_prot
            
            # Thermodynamic Energy Coupling
            pred_cals = (9.0 * pred_fat) + (4.0 * pred_carb) + (4.0 * pred_prot)
            moisture = pred_mass * f_water

        st.subheader("Nutritional Estimation")
        m1, m2 = st.columns(2)
        m1.metric("Total Plate Mass", f"{pred_mass:.1f} g")
        m2.metric("Total Energy", f"{pred_cals:.1f} kcal")

        st.markdown("---")
        st.markdown("### Macronutrient Profile")

        st.write(f"**Protein:** {pred_prot:.1f} g ({f_prot*100:.1f}%)")
        st.progress(float(np.clip(f_prot, 0.0, 1.0)))

        st.write(f"**Carbohydrates:** {pred_carb:.1f} g ({f_carb*100:.1f}%)")
        st.progress(float(np.clip(f_carb, 0.0, 1.0)))

        st.write(f"**Fat:** {pred_fat:.1f} g ({f_fat*100:.1f}%)")
        st.progress(float(np.clip(f_fat, 0.0, 1.0)))

        with st.expander("Physical Domain Audit & Diagnostics", expanded=True):
            st.write(f"• **Network Logits:** `[log_mass: {pred[0]:.3f}, f_fat: {f_fat:.3f}, f_carb: {f_carb:.3f}, f_prot: {f_prot:.3f}, f_water: {f_water:.3f}]`")
            st.write(f"• **Softmax Integrity Check:** Sum of fractions = `{(f_fat + f_carb + f_prot + f_water):.3f}`")
            st.write(f"• **Estimated Moisture & Fiber:** `{moisture:.1f} g`")
            st.info("✔ Verified: Predictions strictly satisfy mathematical fractional bounds.")
else:
    st.info("Upload a top-down food image using the file uploader above to view dietary metrics.")
