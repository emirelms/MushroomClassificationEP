import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# 📌 Model Yolu
model_path = "/Users/emirelmas/Desktop/yolotesto/best.pt"

# 🚀 Cihazı belirle (GPU varsa kullan)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path).to(device)

# 📌 Streamlit Arayüzü Başlığı
st.title("🍄 Mantar Sınıflandırma Modeli")
st.write("📸 Bir fotoğraf yükleyin ve **Analiz Et** butonuna basarak sonucu görün!")

# 📂 Fotoğraf Yükleme Alanı
uploaded_file = st.file_uploader("📂 Bir mantar fotoğrafı yükleyin", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file:
    # 📷 Yüklenen fotoğrafı göster
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Fotoğraf", use_column_width=True)

    # 🔍 Analiz Et Butonu
    if st.button("🔍 Analiz Et"):
        # 📷 OpenCV formatına çevir
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 🔍 Model ile tahmin yap
        results = model(image_cv)
        result = results[0]

        if result.probs is not None:
            predicted_class_index = result.probs.top1
            predicted_class = model.names[predicted_class_index]
            confidence_score = result.probs.top1conf

            # 📊 Tahmin Sonuçları
            st.write(f"✅ **Tahmin:** {predicted_class}")
            st.write(f"🎯 **Güven Skoru:** {confidence_score:.2f}")

            # 📊 Pie Chart Göster
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(
                [confidence_score, 1 - confidence_score],
                labels=[predicted_class, "Diğer"],
                autopct="%1.1f%%",
                colors=["lightgreen", "lightcoral"]
            )
            ax.set_title("Model Tahmin Güveni")
            st.pyplot(fig)
        else:
            st.write("⚠ Model tahmin yapamadı!")
