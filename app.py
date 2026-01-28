import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "brain_tumour_model.keras"
CLASS_NAMES_PATH = "class_names.txt"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Brain Tumour MRI Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Lightweight, mobile-friendly styling ----
st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; max-width: 980px; }
.stButton button { width: 100%; padding: 0.8rem 1rem; border-radius: 14px; }
.card {
  padding: 1rem 1.2rem;
  border-radius: 16px;
  border: 1px solid rgba(120,120,120,0.25);
  background: rgba(250,250,250,0.65);
}
.small { opacity: 0.8; font-size: 0.95rem; }
@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  h1 { font-size: 1.6rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model_and_classes():
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    return model, class_names

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return x

# ---- Header ----
st.title("Brain Tumour MRI Classifier")
st.write("Upload an MRI image to get a prediction and confidence.")

# ---- Options (better than sidebar on mobile) ----
with st.expander("Options", expanded=False):
    show_probs = st.checkbox("Show class probabilities", value=True)
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.60, 0.01)
    show_topk = st.checkbox("Show top 2 predictions", value=True)

model, class_names = load_model_and_classes()

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1, 1], vertical_alignment="top")

with col1:
    if uploaded is None:
        st.markdown(
            """
            <div class="card">
              <b>How to use</b>
              <p class="small">1) Upload a JPG/PNG MRI image<br/>
              2) The model predicts: glioma, meningioma, notumor, pituitary</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

with col2:
    if uploaded is not None:
        pil_img = Image.open(uploaded)
        x = preprocess_image(pil_img)

        probs = model.predict(x, verbose=0)[0]
        probs = probs.astype(float)

        top_idx = int(np.argmax(probs))
        top_label = class_names[top_idx] if top_idx < len(class_names) else f"class_{top_idx}"
        top_conf = float(probs[top_idx])

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Result")

        if top_conf >= threshold:
            st.success(f"{top_label} (confidence: {top_conf:.3f})")
        else:
            st.warning(f"Uncertain: best guess is {top_label} (confidence: {top_conf:.3f})")

        if show_topk:
            top2 = np.argsort(probs)[::-1][:2]
            lines = []
            for i in top2:
                label = class_names[i] if i < len(class_names) else f"class_{i}"
                lines.append(f"- {label}: {probs[i]:.3f}")
            st.markdown("**Top predictions**\n" + "\n".join(lines))

        st.markdown(
            '<p class="small">If confidence is low, try a clearer slice or a different image.</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if show_probs:
            st.subheader("Class probabilities")
            # Sort for readability on mobile
            order = np.argsort(probs)[::-1]
            prob_table = {class_names[i]: float(probs[i]) for i in order if i < len(class_names)}
            st.bar_chart(prob_table, horizontal=True)
