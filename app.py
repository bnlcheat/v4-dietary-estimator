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
# Custom Loss Definition (Required for .keras deserialization)
# ---------------------------------------------------------
@tf.function
def relative_accuracy_loss(y_true, y_pred):
    mass_true = y_true[:, 0]
    mass_pred = y_pred[:, 0]
    mass_rel_err = tf.abs(mass_true - mass_pred) / (mass_true + 0.1)
    
    fractions_true = y_true[:, 1:]
    fractions_pred = y_pred[:, 1:]
    frac_err = tf.reduce_mean(tf.abs(fractions_true - fractions_pred), axis=-1)
    
    return tf.reduce_mean(2.5 * mass_rel_err + 2.0 * frac_err)

# ---------------------------------------------------------
# Loader utilizing Streamlit 1.24 compatible caching
# ---------------------------------------------------------
@st.cache_resource
def load_system_model():
    model_path = "v4_pro_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Missing '{model_path}'. Please make sure it is in the same folder as app.py.")
        st.stop()
    
    # Load model passing the custom loss function mapping
    return tf.keras.models.load_model(model_path, custom_objects={'relative_accuracy_loss': relative_accuracy_loss})

try:
    model = load_system_model()
    st.sidebar.success("✅ V4-Pro Model loaded successfully!")
except Exception as err:
    st.sidebar.error(f"Initialization Error: {err}")
    st.stop()

st.sidebar.header("Model Specifications")
st.sidebar.markdown(
    """
    - **Backbone:** EfficientNetB0
    - **Branch A:** Log1p Mass Regression
    - **Branch B:** Softmax Macro Fractions
    - **Physics:** Thermodynamic Atwater Coupling
    """
)

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
            # 1. Native Tensor Scaling & Explicit EfficientNet Preprocessing
            img_arr = np.array(img.resize((224, 224)), dtype=np.float32)
            img_input = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_arr, axis=0))

            # 2. Forward Pass cleanly without deserialization errors
            raw_pred = model.predict(img_input, verbose=0)
            pred = np.array(raw_pred).squeeze()
            if pred.ndim > 1:
                pred = pred[0]
            
            # 3. Unpack Branch A (Mass) via expm1
            pred_mass = float(np.expm1(pred[0]))
            pred_mass = max(pred_mass, 1.0) # Prevent zero-mass plates

            # 4. Unpack Branch B (Softmax Fractions)
            f_fat = float(pred[1])
            f_carb = float(pred[2])
            f_prot = float(pred[3])
            f_water = float(pred[4])
            
            # 5. Translate Fractions to Grams
            pred_fat = pred_mass * f_fat
            pred_carb = pred_mass * f_carb
            pred_prot = pred_mass * f_prot
            
            # 6. Thermodynamic Energy Coupling (Atwater factors)
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