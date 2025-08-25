
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

st.title("🌿 سیستم آنلاین تشخیص سلامت برگ")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("leaf_model.h5")
    return model

model = load_model()

class_labels = ["apple_healthy", "apple_sick", "pear_healthy"]

def predict_image(img):
    img = load_img(img, target_size=(128,128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    return class_labels[class_idx]

uploaded_file = st.file_uploader("📷 یک تصویر برگ آپلود کنید:", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="تصویر آپلود شده", use_column_width=True)
    with st.spinner("⏳ در حال پردازش..."):
        label = predict_image(uploaded_file)
    st.success(f"✅ نتیجه: برگ تشخیص داده شد به عنوان **{label}**")
