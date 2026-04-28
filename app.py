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
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# CUSTOM CSS - TEMA MERAH, HITAM, BIRU
# ========================
st.markdown("""
<style>
    /* Main background - Hitam Gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0f0f0f 100%);
    }
    
    /* Card style untuk konten */
    .css-1r6slb0, .css-1v0mbdj, .stMarkdown, .element-container, .stAlert {
        background-color: rgba(20, 20, 30, 0.95);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border-left: 4px solid #e74c3c;
    }
    
    /* Judul utama */
    h1 {
        background: linear-gradient(135deg, #e74c3c 0%, #2980b9 50%, #e74c3c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: bold !important;
        text-align: center;
        margin-bottom: 30px !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Subheader */
    h2 {
        color: #e74c3c !important;
        border-left: 4px solid #2980b9;
        padding-left: 15px;
        margin-top: 20px !important;
    }
    
    h3 {
        color: #3498db !important;
        border-left: 3px solid #e74c3c;
        padding-left: 12px;
        margin-top: 15px !important;
    }
    
    h4 {
        color: #ecf0f1 !important;
    }
    
    /* Text color */
    p, li, .stMarkdown {
        color: #ecf0f1 !important;
    }
    
    /* Sidebar styling - Hitam profesional */
    .css-1d391kg, .css-12oz5g7, .stSidebar {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 100%);
        border-right: 1px solid #e74c3c;
    }
    
    /* Sidebar text */
    .stSidebar .stMarkdown, .stSidebar p, .stSidebar label {
        color: #ecf0f1 !important;
    }
    
    /* Button styling - Merah & Biru */
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #2980b9 0%, #1a6da0 100%);
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        border: 1px solid #e74c3c;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #e74c3c;
    }
    
    /* Info box styling */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #e74c3c;
        background-color: #1a1a2e;
    }
    
    .stAlert > div {
        color: #ecf0f1;
    }
    
    /* Success box */
    .stSuccess {
        background-color: rgba(46, 204, 113, 0.2);
        border-left-color: #2ecc71;
    }
    
    /* Warning box */
    .stWarning {
        background-color: rgba(241, 196, 15, 0.2);
        border-left-color: #f1c40f;
    }
    
    /* Image container */
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid #e74c3c;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2c3e50;
        border-radius: 6px;
        color: #ecf0f1;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #2c3e50 0%, #1a1a2e 100%);
        border-radius: 8px;
        color: #e74c3c;
        border: 1px solid #e74c3c;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #ecf0f1;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        border-radius: 10px;
        margin-top: 30px;
        border-top: 2px solid #e74c3c;
        border-bottom: 2px solid #2980b9;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #0a0a0a 100%);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        border: 1px solid #e74c3c;
    }
    
    /* Code block */
    .stCodeBlock {
        background-color: #0a0a0a !important;
        border: 1px solid #2980b9;
        border-radius: 8px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    ::-webkit-scrollbar-thumb {
        background: #e74c3c;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# SIDEBAR NAVIGASI
# ========================
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h2 style="color: #e74c3c; border-left: none; text-align: center;">🎯 PCD STUDIO</h2>
    <p style="color: #3498db;">Pengolahan Citra Digital</p>
    <hr style="border-color: #e74c3c;">
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

# Fungsi baca gambar
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
    <p style="color: #7f8c8d;">Dibangun dengan Streamlit</p>
    <p style="color: #7f8c8d;">Tugas Pengolahan Citra Digital</p>
    <p style="color: #e74c3c;">© 2026</p>
</div>
""", unsafe_allow_html=True)

# ========================
# 1. PENDAHULUAN
# ========================
if menu == "Pendahuluan":
    st.markdown("# 📖 PENDAHULUAN")
    
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
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #0a0a0a 100%); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #e74c3c;">
            <h3 style="color: #e74c3c;">📋 INFORMASI</h3>
            <hr style="border-color: #2980b9;">
            <p><strong style="color: #e74c3c;">Mata Kuliah</strong><br>Pengolahan Citra Digital</p>
            <p><strong style="color: #e74c3c;">Semester</strong><br>Genap 2025/2026</p>
            <p><strong style="color: #e74c3c;">Platform</strong><br>Streamlit Cloud</p>
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
        st.markdown("")
        st.markdown("- **Brightness**: I'(x,y) = I(x,y) + c")
        st.markdown("- **Contrast**: I'(x,y) = α·I(x,y) + β")
        st.markdown("- **Sharpening**: Menggunakan kernel high-pass filter")
    
    with tab3:
        st.markdown("### Histogram Citra")
        st.markdown("Histogram adalah grafik yang menunjukkan distribusi intensitas piksel.")
        st.markdown("")
        st.markdown("**Histogram Equalization:**")
        st.markdown("- Menyebarkan intensitas secara merata")
        st.markdown("- Meningkatkan kontras gambar")
        st.markdown("- Rumus: s_k = T(r_k) = Σ(n_j/n)")
    
    with tab4:
        st.markdown("### Noise & Filtering")
        st.markdown("**Jenis Noise:**")
        st.markdown("- **Gaussian**: Noise dengan distribusi normal")
        st.markdown("- **Salt & Pepper**: Titik acak hitam/putih")
        st.markdown("")
        st.markdown("**Filter:**")
        st.markdown("- **Median Filter**: Baik untuk salt & pepper noise")
        st.markdown("- **Gaussian Filter**: Baik untuk Gaussian noise")
        st.markdown("- **Low-pass Filter**: Menghaluskan gambar")
        st.markdown("- **High-pass Filter**: Mendeteksi tepi")
    
    with tab5:
        st.markdown("### Fast Fourier Transform (FFT)")
        st.markdown("FFT mengubah citra dari domain spasial ke domain frekuensi.")
        st.markdown("")
        st.markdown("- **Frekuensi Rendah**: Informasi utama, perubahan gradual")
        st.markdown("- **Frekuensi Tinggi**: Detail tepi, noise")
        st.markdown("- **Magnitude Spectrum**: Visualisasi energi frekuensi")

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
        st.warning("⚠️ Belum ada gambar! Silakan load gambar terlebih dahulu.")

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
        fig.patch.set_facecolor('#1a1a2e')
        
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Citra Grayscale", fontsize=12, fontweight='bold', color='#e74c3c')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.hist(gray.ravel(), bins=256, color='#e74c3c', alpha=0.7, edgecolor='white')
        plt.title("Histogram Asli", fontsize=12, fontweight='bold', color='#e74c3c')
        plt.xlabel("Intensitas Piksel", color='#ecf0f1')
        plt.ylabel("Frekuensi", color='#ecf0f1')
        plt.grid(True, alpha=0.3)
        plt.gca().set_facecolor('#2c3e50')
        
        equalized = cv2.equalizeHist(gray)
        plt.subplot(1, 3, 3)
        plt.hist(equalized.ravel(), bins=256, color='#2980b9', alpha=0.7, edgecolor='white')
        plt.title("Histogram Setelah Equalization", fontsize=12, fontweight='bold', color='#2980b9')
        plt.xlabel("Intensitas Piksel", color='#ecf0f1')
        plt.ylabel("Frekuensi", color='#ecf0f1')
        plt.grid(True, alpha=0.3)
        plt.gca().set_facecolor('#2c3e50')
        
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
        fig.patch.set_facecolor('#1a1a2e')
        
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Citra Grayscale", fontsize=12, fontweight='bold', color='#e74c3c')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title("Magnitude Spectrum (FFT)", fontsize=12, fontweight='bold', color='#e74c3c')
        plt.axis('off')
        
        f_ishift = ifftshift(fshift)
        img_back = ifft2(f_ishift)
        img_back = np.abs(img_back)
        plt.subplot(1, 3, 3)
        plt.imshow(img_back, cmap='gray')
        plt.title("Hasil Inverse FFT", fontsize=12, fontweight='bold', color='#2980b9')
        plt.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        with st.expander("📖 Penjelasan FFT"):
            st.markdown("- **Magnitude Spectrum**: Menunjukkan distribusi energi frekuensi dalam citra")
            st.markdown("- **Titik terang di tengah**: Komponen frekuensi rendah (informasi utama)")
            st.markdown("- **Semakin jauh dari pusat**: Frekuensi tinggi (detail tepi, noise)")
            st.markdown("- **Inverse FFT**: Mengembalikan citra ke domain spasial")
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
        st.markdown("")
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
    st.markdown("### 🔧 Kode Utama (Core Implementation)")
    st.code("""
import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift

# Load gambar
img = cv2.imread('gambar.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Rotasi
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1)
rotated = cv2.warpAffine(img, M, (w, h))

# Histogram Equalization
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
equalized = cv2.equalizeHist(gray)

# FFT
f = fft2(gray)
fshift = fftshift(f)
magnitude = np.log(np.abs(fshift) + 1)
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
        st.markdown("")
        
        st.markdown("#### 📊 Histogram")
        st.markdown("**Hasil:** Citra gelap memiliki histogram terkonsentrasi di kiri.")
        st.markdown("**Analisis:** Equalization efektif untuk meningkatkan kontras citra secara otomatis.")
    
    with col2:
        st.markdown("#### 🎛️ Noise & Filtering")
        st.markdown("**Hasil:** Gaussian noise diatasi dengan Gaussian filter.")
        st.markdown("**Analisis:** Pemilihan filter harus disesuaikan dengan jenis noise.")
        st.markdown("")
        
        st.markdown("#### 🌊 FFT")
        st.markdown("**Hasil:** Spektrum frekuensi menunjukkan komponen frekuensi rendah di pusat.")
        st.markdown("**Analisis:** FFT berguna untuk analisis dan filtering di domain frekuensi.")
        st.markdown("")
        
        st.markdown("#### ✅ Kesimpulan Analisis")
        st.markdown("Semua teknik pengolahan citra berhasil diimplementasikan dan berfungsi dengan baik.")
    
    st.markdown("---")
    st.markdown("### 📸 Dokumentasi Hasil Uji Coba")
    st.info("💡 Screenshot hasil pengujian setiap fitur dapat ditambahkan di sini untuk dokumentasi tugas.")

# ========================
# 11. KESIMPULAN
# ========================
elif menu == "Kesimpulan":
    st.markdown("# 📌 KESIMPULAN")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%); border-radius: 10px; padding: 15px; text-align: center;">
            <h1 style="color: white; font-size: 2rem;">✅</h1>
            <h4 style="color: white;">TERIMPLEMENTASI</h4>
            <p style="color: white;">Semua fitur berhasil diimplementasikan</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a6da0 0%, #2980b9 100%); border-radius: 10px; padding: 15px; text-align: center;">
            <h1 style="color: white; font-size: 2rem;">🎯</h1>
            <h4 style="color: white;">TUJUAN TERCAPAI</h4>
            <p style="color: white;">Aplikasi interaktif untuk pembelajaran</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2c3e50 0%, #1a1a2e 100%); border-radius: 10px; padding: 15px; text-align: center; border: 1px solid #e74c3c;">
            <h1 style="color: #e74c3c; font-size: 2rem;">🚀</h1>
            <h4 style="color: #e74c3c;">SIAP DEPLOY</h4>
            <p style="color: white;">Aplikasi siap digunakan kapan saja</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Ringkasan Kesimpulan")
    
    st.markdown("1. **✅ Aplikasi berhasil mengimplementasikan 5 teknik utama** pengolahan citra digital")
    st.markdown("2. **🎨 Antarmuka interaktif** memudahkan eksperimen dan pembelajaran")
    st.markdown("3. **⚡ Semua fitur berfungsi** dengan baik sesuai teori pengolahan citra")
    st.markdown("4. **🌐 Aplikasi dapat diakses** secara online melalui Streamlit Cloud")
    st.markdown("5. **📈 Hasil pengujian** menunjukkan setiap operasi berjalan sesuai parameter")
    
    st.markdown("")
    st.markdown("### 🔮 Saran Pengembangan Lebih Lanjut")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- Implementasi deteksi tepi (Canny, Sobel, Prewitt)")
        st.markdown("- Segmentasi citra dengan thresholding adaptif")
    with col2:
        st.markdown("- Ekstraksi fitur untuk klasifikasi gambar")
        st.markdown("- Integrasi model deep learning (CNN)")

# ========================
# FOOTER
# ========================
st.markdown("""
<div class="footer">
    <p><strong style="color: #e74c3c;">PCD STUDIO</strong> | Pengolahan Citra Digital | Tugas Akhir Semester | 2026</p>
    <p style="font-size: 12px;">Dibangun dengan Streamlit, OpenCV, dan Python</p>
</div>
""", unsafe_allow_html=True)
