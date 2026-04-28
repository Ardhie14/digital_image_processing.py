import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# ========================
# KONFIGURASI HALAMAN
# ========================
st.set_page_config(
    page_title="PCD Studio - Pengolahan Citra Digital",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# CUSTOM CSS - TEMA BIRU PUTIH (DIPERBAIKI)
# ========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #e8f0fe 0%, #d4e4fc 50%, #e8f0fe 100%);
    }
    
    .css-1r6slb0, .css-1v0mbdj, .stMarkdown, .element-container {
        background-color: white;
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    h1 {
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 50%, #1a5276 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        font-weight: bold !important;
        text-align: center;
        margin-bottom: 30px !important;
    }
    
    h2 {
        color: #1a5276 !important;
        border-left: 4px solid #2980b9;
        padding-left: 15px;
        margin-top: 20px !important;
    }
    
    h3 {
        color: #2471a3 !important;
        border-left: 3px solid #85c1e9;
        padding-left: 12px;
        margin-top: 15px !important;
    }
    
    p, li, .stMarkdown, span, div {
        color: #2c3e50 !important;
    }
    
    /* PERBAIKAN WARNING BOX */
    .stAlert {
        border-radius: 12px;
        padding: 16px !important;
        margin: 10px 0 !important;
    }
    
    .stAlert.stWarning {
        background-color: #fef3c7 !important;
        border-left: 4px solid #f59e0b !important;
    }
    
    .stAlert.stWarning p, 
    .stAlert.stWarning div,
    .stAlert.stWarning .stMarkdown {
        color: #92400e !important;
        font-weight: 600 !important;
    }
    
    .stAlert.stSuccess {
        background-color: #d1fae5 !important;
        border-left: 4px solid #10b981 !important;
    }
    
    .stAlert.stSuccess p,
    .stAlert.stSuccess div {
        color: #065f46 !important;
    }
    
    .stAlert.stInfo {
        background-color: #dbeafe !important;
        border-left: 4px solid #3b82f6 !important;
    }
    
    .stAlert.stInfo p,
    .stAlert.stInfo div {
        color: #1e40af !important;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7, .stSidebar {
        background: linear-gradient(180deg, #1a5276 0%, #2471a3 100%);
    }
    
    .stSidebar .stMarkdown, 
    .stSidebar p, 
    .stSidebar label,
    .stSidebar span,
    .stSidebar div {
        color: white !important;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #2980b9 0%, #1a5276 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: white;
        color: #2c3e50 !important;
        border-radius: 10px;
        border: 1px solid #85c1e9;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #2980b9;
    }
    
    /* Image */
    .stImage {
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        border: 1px solid #d4e4fc;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #ebf5fb;
        border-radius: 12px;
        padding: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        color: #1a5276 !important;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2980b9 0%, #1a5276 100%);
        color: white !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: white !important;
        background: linear-gradient(135deg, #1a5276 0%, #2471a3 100%);
        border-radius: 16px;
        margin-top: 30px;
    }
    
    .footer p {
        color: white !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2980b9 0%, #1a5276 100%);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    .metric-card h1, .metric-card h4, .metric-card p {
        color: white !important;
    }
    
    /* Code block */
    .stCodeBlock {
        background-color: #f8f9fa !important;
        border: 1px solid #d4e4fc;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# SIDEBAR NAVIGASI
# ========================
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h2 style="color: white; border-left: none; text-align: center;">💙 PCD STUDIO</h2>
    <p style="color: #d4e4fc;">Pengolahan Citra Digital</p>
    <hr style="border-color: #d4e4fc;">
</div>
""", unsafe_allow_html=True)

menu_icons = {
    "Pendahuluan": "📖",
    "Teori Singkat": "📚",
    "Load & Tampilkan Gambar": "📂",
    "Transformasi Geometri": "🔄",
    "Enhancement": "✨",
    "Histogram": "📊",
    "Noise & Filtering": "🎛️",
    "FFT (Analisis Frekuensi)": "🌊",
    "Implementasi": "💻",
    "Hasil & Pembahasan": "📝",
    "Kesimpulan": "📌"
}

menu = st.sidebar.radio(
    "📌 NAVIGASI",
    list(menu_icons.keys()),
    format_func=lambda x: f"{menu_icons[x]} {x}"
)

# Session state
if 'gambar_asli' not in st.session_state:
    st.session_state.gambar_asli = None
if 'gambar_proses' not in st.session_state:
    st.session_state.gambar_proses = None

def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def reset_to_original():
    if st.session_state.gambar_asli is not None:
        st.session_state.gambar_proses = st.session_state.gambar_asli.copy()
        st.success("✅ Gambar berhasil di-reset ke original!")

# Status di sidebar
st.sidebar.markdown("---")
if st.session_state.gambar_asli is not None:
    st.sidebar.success("✅ STATUS: Gambar tersedia")
    h, w = st.session_state.gambar_asli.shape[:2]
    st.sidebar.info(f"📐 Dimensi: {w} x {h} px")
    if st.sidebar.button("🔄 Reset ke Gambar Asli", use_container_width=True):
        reset_to_original()
else:
    st.sidebar.warning("⚠️ STATUS: Belum ada gambar")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 12px;">
    <p style="color: #d4e4fc;">Dibangun dengan Streamlit</p>
    <p style="color: #d4e4fc;">Tugas Pengolahan Citra Digital</p>
    <p style="color: white;">© 2026</p>
</div>
""", unsafe_allow_html=True)

# ========================
# 1. PENDAHULUAN
# ========================
if menu == "Pendahuluan":
    st.markdown("# 💙 PENDAHULUAN")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Selamat Datang di PCD Studio")
        st.markdown("Aplikasi ini dikembangkan untuk memenuhi tugas kuliah **Pengolahan Citra Digital**.")
        st.markdown("")
        st.markdown("#### 🎯 Tujuan Aplikasi")
        st.markdown("- ✅ Memahami konsep dasar pengolahan citra digital")
        st.markdown("- ✅ Mengimplementasikan berbagai teknik pengolahan citra")
        st.markdown("- ✅ Menganalisis efek setiap operasi pada citra")
        st.markdown("- ✅ Menyediakan alat interaktif untuk eksperimen")
        st.markdown("")
        st.markdown("#### ✨ Fitur Lengkap")
        st.markdown("- 🔄 **Transformasi Geometri** (Rotasi, Translasi, Skala, Flip)")
        st.markdown("- ✨ **Enhancement** (Brightness, Contrast, Sharpening)")
        st.markdown("- 📊 **Histogram** & Equalization")
        st.markdown("- 🎛️ **Noise & Filtering** (Gaussian, Median, Low-pass, High-pass)")
        st.markdown("- 🌊 **FFT** (Analisis Frekuensi)")
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2980b9 0%, #1a5276 100%); border-radius: 16px; padding: 20px; text-align: center;">
            <h3 style="color: white;">📋 INFORMASI</h3>
            <hr style="border-color: #85c1e9;">
            <p style="color: white;"><strong>Mata Kuliah</strong><br>Pengolahan Citra Digital</p>
            <p style="color: white;"><strong>Semester</strong><br>Genap 2025/2026</p>
            <p style="color: white;"><strong>Platform</strong><br>Streamlit Cloud</p>
        </div>
        """, unsafe_allow_html=True)

# ========================
# 2. TEORI SINGKAT
# ========================
elif menu == "Teori Singkat":
    st.markdown("# 📚 TEORI SINGKAT")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔄 Transformasi Geometri", "✨ Enhancement", "📊 Histogram", "🎛️ Noise & Filtering", "🌊 FFT"])
    
    with tab1:
        st.markdown("### Transformasi Geometri")
        st.markdown("Transformasi geometri adalah operasi yang mengubah posisi piksel dalam citra.")
        st.markdown("")
        st.markdown("| Operasi | Rumus | Keterangan |")
        st.markdown("|---------|-------|-------------|")
        st.markdown("| **Rotasi** | x' = x cos θ - y sin θ | Memutar gambar |")
        st.markdown("| **Translasi** | x' = x + dx, y' = y + dy | Memindahkan posisi |")
        st.markdown("| **Skala** | x' = sx·x, y' = sy·y | Mengubah ukuran |")
        st.markdown("| **Flip** | x' = -x atau y' = -y | Membalik gambar |")
    
    with tab2:
        st.markdown("### Image Enhancement")
        st.markdown("Enhancement bertujuan meningkatkan kualitas visual citra.")
        st.markdown("- **Brightness**: I'(x,y) = I(x,y) + c")
        st.markdown("- **Contrast**: I'(x,y) = α·I(x,y) + β")
        st.markdown("- **Sharpening**: Menggunakan kernel high-pass filter")
    
    with tab3:
        st.markdown("### Histogram Citra")
        st.markdown("Histogram adalah grafik yang menunjukkan distribusi intensitas piksel.")
        st.markdown("**Histogram Equalization:** Menyebarkan intensitas secara merata untuk meningkatkan kontras.")
    
    with tab4:
        st.markdown("### Noise & Filtering")
        st.markdown("**Jenis Noise:** Gaussian dan Salt & Pepper")
        st.markdown("**Filter:** Gaussian, Median, Low-pass, High-pass")
    
    with tab5:
        st.markdown("### Fast Fourier Transform (FFT)")
        st.markdown("FFT mengubah citra dari domain spasial ke domain frekuensi.")

# ========================
# 3. LOAD GAMBAR
# ========================
elif menu == "Load & Tampilkan Gambar":
    st.markdown("# 📂 LOAD & TAMPILKAN GAMBAR")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📤 Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state.gambar_asli = load_image(uploaded_file)
            st.session_state.gambar_proses = st.session_state.gambar_asli.copy()
            st.balloons()
            st.success("✅ Gambar berhasil di-load!")
            h, w = st.session_state.gambar_asli.shape[:2]
            st.info(f"📐 Dimensi: {w} x {h} px | 🎨 Mode: RGB")
    
    with col2:
        if st.session_state.gambar_asli is not None:
            st.markdown("### 🖼️ Preview Gambar")
            st.image(st.session_state.gambar_asli, caption="Gambar Asli", use_container_width=True)
        else:
            st.info("💡 Belum ada gambar. Silakan upload gambar terlebih dahulu.")
    
    if st.session_state.gambar_asli is not None:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🟢 Gambar Asli")
            st.image(st.session_state.gambar_asli, use_container_width=True)
        with col2:
            st.markdown("#### 🔵 Gambar Proses")
            st.image(st.session_state.gambar_proses, use_container_width=True)

# ========================
# 4. TRANSFORMASI GEOMETRI
# ========================
elif menu == "Transformasi Geometri":
    st.markdown("# 🔄 TRANSFORMASI GEOMETRI")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            transform = st.selectbox("🎯 Pilih Transformasi", ["Rotasi", "Translasi", "Skala", "Flip"])
            
            if transform == "Rotasi":
                angle = st.slider("🔄 Sudut Rotasi (derajat):", -180, 180, 0)
                if st.button("🚀 Terapkan Rotasi", use_container_width=True):
                    h, w = img.shape[:2]
                    center = (w//2, h//2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    result = cv2.warpAffine(img, M, (w, h))
                    st.session_state.gambar_proses = result
                    st.rerun()
                    
            elif transform == "Translasi":
                tx = st.slider("⬅️ ➡️ Translasi X:", -100, 100, 0)
                ty = st.slider("⬆️ ⬇️ Translasi Y:", -100, 100, 0)
                if st.button("🚀 Terapkan Translasi", use_container_width=True):
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    result = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    st.session_state.gambar_proses = result
                    st.rerun()
                    
            elif transform == "Skala":
                scale = st.slider("🔍 Faktor Skala:", 0.1, 3.0, 1.0)
                if st.button("🚀 Terapkan Skala", use_container_width=True):
                    new_w = int(img.shape[1] * scale)
                    new_h = int(img.shape[0] * scale)
                    result = cv2.resize(img, (new_w, new_h))
                    st.session_state.gambar_proses = result
                    st.rerun()
                    
            else:
                flip_code = st.selectbox("🪞 Arah Flip:", ["Horizontal", "Vertikal"])
                if st.button("🚀 Terapkan Flip", use_container_width=True):
                    if flip_code == "Horizontal":
                        result = cv2.flip(img, 1)
                    else:
                        result = cv2.flip(img, 0)
                    st.session_state.gambar_proses = result
                    st.rerun()
        
        with col2:
            st.markdown("### 📊 Hasil Transformasi")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Sebelum**")
                st.image(img, use_container_width=True)
            with col_b:
                st.markdown("**Sesudah**")
                st.image(st.session_state.gambar_proses, use_container_width=True)
    else:
        st.warning("⚠️ Belum ada gambar! Silakan load gambar terlebih dahulu di menu 'Load & Tampilkan Gambar'.")

# ========================
# 5. ENHANCEMENT
# ========================
elif menu == "Enhancement":
    st.markdown("# ✨ ENHANCEMENT")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        pil_img = Image.fromarray(img)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            enhancement_type = st.selectbox("🎨 Pilih Enhancement", ["Brightness", "Contrast", "Sharpening"])
            
            if enhancement_type == "Brightness":
                factor = st.slider("💡 Faktor Kecerahan:", 0.5, 2.0, 1.0)
                if st.button("✨ Terapkan Brightness", use_container_width=True):
                    enhancer = ImageEnhance.Brightness(pil_img)
                    result = np.array(enhancer.enhance(factor))
                    st.session_state.gambar_proses = result
                    st.rerun()
                    
            elif enhancement_type == "Contrast":
                factor = st.slider("🎨 Faktor Kontras:", 0.5, 2.0, 1.0)
                if st.button("✨ Terapkan Contrast", use_container_width=True):
                    enhancer = ImageEnhance.Contrast(pil_img)
                    result = np.array(enhancer.enhance(factor))
                    st.session_state.gambar_proses = result
                    st.rerun()
                    
            else:
                if st.button("🔪 Terapkan Sharpening", use_container_width=True):
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    result = cv2.filter2D(img, -1, kernel)
                    st.session_state.gambar_proses = result
                    st.rerun()
        
        with col2:
            st.markdown("### 📊 Hasil Enhancement")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Sebelum**")
                st.image(img, use_container_width=True)
            with col_b:
                st.markdown("**Sesudah**")
                st.image(st.session_state.gambar_proses, use_container_width=True)
    else:
        st.warning("⚠️ Belum ada gambar! Silakan load gambar terlebih dahulu.")

# ========================
# 6. HISTOGRAM
# ========================
elif menu == "Histogram":
    st.markdown("# 📊 HISTOGRAM & EQUALIZATION")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        fig = plt.figure(figsize=(14, 6))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Citra Grayscale", fontsize=12, fontweight='bold', color='#1a5276')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.hist(gray.ravel(), bins=256, color='#2980b9', alpha=0.7, edgecolor='#1a5276')
        plt.title("Histogram Asli", fontsize=12, fontweight='bold', color='#1a5276')
        plt.xlabel("Intensitas Piksel", color='#2c3e50')
        plt.ylabel("Frekuensi", color='#2c3e50')
        plt.grid(True, alpha=0.3)
        plt.gca().set_facecolor('#ebf5fb')
        
        equalized = cv2.equalizeHist(gray)
        plt.subplot(1, 3, 3)
        plt.hist(equalized.ravel(), bins=256, color='#3498db', alpha=0.7, edgecolor='#1a5276')
        plt.title("Histogram Setelah Equalization", fontsize=12, fontweight='bold', color='#1a5276')
        plt.xlabel("Intensitas Piksel", color='#2c3e50')
        plt.ylabel("Frekuensi", color='#2c3e50')
        plt.grid(True, alpha=0.3)
        plt.gca().set_facecolor('#ebf5fb')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🟢 Sebelum Equalization**")
            st.image(gray, use_container_width=True)
        with col2:
            st.markdown("**🔵 Setelah Equalization**")
            st.image(equalized, use_container_width=True)
        
        if st.button("✅ Terapkan Equalization ke Gambar Proses", use_container_width=True):
            st.session_state.gambar_proses = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            st.success("✅ Equalization berhasil diterapkan!")
            st.rerun()
    else:
        st.warning("⚠️ Belum ada gambar! Silakan load gambar terlebih dahulu.")

# ========================
# 7. NOISE & FILTERING
# ========================
elif menu == "Noise & Filtering":
    st.markdown("# 🎛️ NOISE & FILTERING")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            operation = st.selectbox("🔧 Pilih Operasi", ["Tambah Noise", "Terapkan Filter"])
            
            if operation == "Tambah Noise":
                noise_type = st.selectbox("📡 Jenis Noise", ["Gaussian", "Salt & Pepper"])
                if st.button("⚡ Tambahkan Noise", use_container_width=True):
                    if noise_type == "Gaussian":
                        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
                        result = cv2.add(img, noise)
                    else:
                        s_vs_p = 0.5
                        amount = 0.05
                        result = img.copy()
                        num_salt = np.ceil(amount * img.size * s_vs_p)
                        coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape]
                        result[coords[0], coords[1], :] = [255, 255, 255]
                        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
                        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img.shape]
                        result[coords[0], coords[1], :] = [0, 0, 0]
                    st.session_state.gambar_proses = result
                    st.rerun()
                    
            else:
                filter_type = st.selectbox("🔍 Pilih Filter", ["Gaussian", "Median", "Low-pass", "High-pass"])
                if st.button("🎯 Terapkan Filter", use_container_width=True):
                    if filter_type == "Gaussian":
                        result = cv2.GaussianBlur(img, (5, 5), 0)
                    elif filter_type == "Median":
                        result = cv2.medianBlur(img, 5)
                    elif filter_type == "Low-pass":
                        kernel = np.ones((5, 5), np.float32) / 25
                        result = cv2.filter2D(img, -1, kernel)
                    else:
                        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                        result = cv2.filter2D(img, -1, kernel)
                    st.session_state.gambar_proses = result
                    st.rerun()
        
        with col2:
            st.markdown("### 📊 Hasil Operasi")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Sebelum**")
                st.image(img, use_container_width=True)
            with col_b:
                st.markdown("**Sesudah**")
                st.image(st.session_state.gambar_proses, use_container_width=True)
    else:
        st.warning("⚠️ Belum ada gambar! Silakan load gambar terlebih dahulu.")

# ========================
# 8. FFT
# ========================
elif menu == "FFT (Analisis Frekuensi)":
    st.markdown("# 🌊 FFT - ANALISIS FREKUENSI")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        
        fig = plt.figure(figsize=(15, 5))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Citra Grayscale", fontsize=12, fontweight='bold', color='#1a5276')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title("Magnitude Spectrum (FFT)", fontsize=12, fontweight='bold', color='#1a5276')
        plt.axis('off')
        
        f_ishift = ifftshift(fshift)
        img_back = ifft2(f_ishift)
        img_back = np.abs(img_back)
        plt.subplot(1, 3, 3)
        plt.imshow(img_back, cmap='gray')
        plt.title("Hasil Inverse FFT", fontsize=12, fontweight='bold', color='#1a5276')
        plt.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        with st.expander("📖 Penjelasan FFT"):
            st.markdown("- **Magnitude Spectrum**: Menunjukkan distribusi energi frekuensi dalam citra")
            st.markdown("- **Titik terang di tengah**: Komponen frekuensi rendah (informasi utama)")
            st.markdown("- **Semakin jauh dari pusat**: Frekuensi tinggi (detail tepi, noise)")
    else:
        st.warning("⚠️ Belum ada gambar! Silakan load gambar terlebih dahulu.")

# ========================
# 9. IMPLEMENTASI
# ========================
elif menu == "Implementasi":
    st.markdown("# 💻 IMPLEMENTASI")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🛠️ Tools & Libraries")
        st.markdown("| Library | Fungsi |")
        st.markdown("|---------|--------|")
        st.markdown("| **Streamlit** | Framework web interaktif |")
        st.markdown("| **OpenCV** | Operasi citra |")
        st.markdown("| **Pillow** | Enhancement dasar |")
        st.markdown("| **NumPy** | Manipulasi array |")
        st.markdown("| **Matplotlib** | Visualisasi |")
        st.markdown("| **SciPy** | FFT |")
    
    with col2:
        st.markdown("### 📁 Struktur Project")
        st.code("""
PCD-STUDIO/
├── streamlit_app.py
├── requirements.txt
└── README.md
        """)
    
    st.markdown("---")
    st.markdown("### 🔧 Kode Utama")
    st.code("""
import cv2
import numpy as np
from scipy.fft import fft2, fftshift

img = cv2.imread('gambar.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
f = fft2(gray)
fshift = fftshift(f)
    """, language='python')

# ========================
# 10. HASIL & PEMBAHASAN
# ========================
elif menu == "Hasil & Pembahasan":
    st.markdown("# 📝 HASIL & PEMBAHASAN")
    
    st.markdown("### 📊 Observasi & Analisis")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔄 Transformasi Geometri")
        st.markdown("**Hasil:** Rotasi mempertahankan informasi visual namun terjadi pemotongan di tepi.")
        st.markdown("**Analisis:** Transformasi geometri efektif untuk koreksi posisi dan orientasi citra.")
        st.markdown("")
        st.markdown("#### ✨ Enhancement")
        st.markdown("**Hasil:** Brightness meningkatkan kecerahan global. Sharpening mempertegas tepi objek.")
        st.markdown("**Analisis:** Enhancement berguna untuk memperbaiki kualitas visual citra.")
    
    with col2:
        st.markdown("#### 🎛️ Noise & Filtering")
        st.markdown("**Hasil:** Gaussian noise diatasi dengan Gaussian filter.")
        st.markdown("**Analisis:** Pemilihan filter harus disesuaikan dengan jenis noise.")
        st.markdown("")
        st.markdown("#### 🌊 FFT")
        st.markdown("**Hasil:** Spektrum frekuensi menunjukkan komponen frekuensi rendah di pusat.")
        st.markdown("**Analisis:** FFT berguna untuk analisis dan filtering di domain frekuensi.")
    
    st.markdown("---")
    st.info("💡 Screenshot hasil pengujian setiap fitur dapat ditambahkan di sini untuk dokumentasi tugas.")

# ========================
# 11. KESIMPULAN
# ========================
elif menu == "Kesimpulan":
    st.markdown("# 📌 KESIMPULAN")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h1 style="color: white; font-size: 2rem;">✅</h1>
            <h4 style="color: white;">TERIMPLEMENTASI</h4>
            <p style="color: white;">Semua fitur berhasil</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h1 style="color: white; font-size: 2rem;">🎯</h1>
            <h4 style="color: white;">TUJUAN TERCAPAI</h4>
            <p style="color: white;">Aplikasi interaktif</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h1 style="color: white; font-size: 2rem;">🚀</h1>
            <h4 style="color: white;">SIAP DEPLOY</h4>
            <p style="color: white;">Akses kapan saja</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Ringkasan Kesimpulan")
    st.markdown("1. ✅ Aplikasi berhasil mengimplementasikan 5 teknik utama pengolahan citra digital")
    st.markdown("2. 🎨 Antarmuka interaktif memudahkan eksperimen dan pembelajaran")
    st.markdown("3. ⚡ Semua fitur berfungsi dengan baik sesuai teori")
    st.markdown("4. 🌐 Aplikasi dapat diakses secara online melalui Streamlit Cloud")

# ========================
# FOOTER
# ========================
st.markdown("""
<div class="footer">
    <p><strong>PCD STUDIO</strong> | Pengolahan Citra Digital | Tugas Akhir Semester | 2026</p>
</div>
""", unsafe_allow_html=True)
