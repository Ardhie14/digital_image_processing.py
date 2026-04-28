import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import time

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
# CUSTOM CSS - TEMA PROFESIONAL (NAVY + EMERALD)
# ========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
    }
    
    .css-1r6slb0, .css-1v0mbdj, .stMarkdown, .element-container {
        background-color: white;
        border-radius: 20px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    h1 {
        background: linear-gradient(135deg, #0f2b3d 0%, #1a5276 50%, #0f2b3d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 32px !important;
    }
    
    h2 {
        color: #0f2b3d !important;
        border-left: 5px solid #2ecc71;
        padding-left: 16px;
        margin-top: 24px !important;
        font-weight: 600 !important;
    }
    
    h3 {
        color: #1a5276 !important;
        border-left: 3px solid #2ecc71;
        padding-left: 12px;
        margin-top: 20px !important;
        font-weight: 600 !important;
    }
    
    p, li, .stMarkdown {
        color: #334155 !important;
        line-height: 1.6 !important;
    }
    
    .css-1d391kg, .css-12oz5g7, .stSidebar {
        background: linear-gradient(180deg, #0f2b3d 0%, #1a3a4f 100%);
        border-right: none;
    }
    
    .stSidebar .stMarkdown, 
    .stSidebar p, 
    .stSidebar label,
    .stSidebar span,
    .stSidebar div {
        color: #ffffff !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #27ae60 0%, #219a52 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
    }
    
    .stSlider > div > div > div {
        background-color: #2ecc71;
    }
    
    .stAlert.stWarning {
        background-color: #fef3c7 !important;
        border-left: 5px solid #f59e0b !important;
    }
    
    .stAlert.stWarning p {
        color: #92400e !important;
        font-weight: 600 !important;
    }
    
    .stAlert.stSuccess {
        background-color: #d1fae5 !important;
        border-left: 5px solid #10b981 !important;
    }
    
    .stAlert.stSuccess p {
        color: #065f46 !important;
    }
    
    .stImage {
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        border-radius: 14px;
        padding: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px;
        color: #1a5276 !important;
        padding: 8px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white !important;
    }
    
    .footer {
        text-align: center;
        padding: 24px;
        background: linear-gradient(135deg, #0f2b3d 0%, #1a3a4f 100%);
        border-radius: 20px;
        margin-top: 40px;
    }
    
    .footer p {
        color: white !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #0f2b3d 0%, #1a3a4f 100%);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    
    .metric-card h1, .metric-card h4, .metric-card p {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# SIDEBAR
# ========================
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <div style="width: 60px; height: 60px; background: #2ecc71; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto;">
        <span style="font-size: 30px;">🎯</span>
    </div>
    <h2 style="color: white; border-left: none; margin-top: 16px;">PCD STUDIO</h2>
    <p style="color: #a0c4e8;">Pengolahan Citra Digital</p>
    <hr>
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
    "MENU NAVIGASI",
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
        st.success("Gambar berhasil di-reset ke original!")

# Status sidebar
st.sidebar.markdown("---")
if st.session_state.gambar_asli is not None:
    st.sidebar.success("STATUS: Gambar tersedia")
    h, w = st.session_state.gambar_asli.shape[:2]
    st.sidebar.info(f"Dimensi: {w} x {h} px")
    if st.sidebar.button("Reset ke Gambar Asli", use_container_width=True):
        reset_to_original()
        st.rerun()
else:
    st.sidebar.warning("STATUS: Belum ada gambar")

st.sidebar.markdown("---")
st.sidebar.markdown("Streamlit Cloud | Tugas PCD | 2026")

# ========================
# 1. PENDAHULUAN
# ========================
if menu == "Pendahuluan":
    st.markdown("# PENDAHULUAN")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Selamat Datang di PCD Studio")
        st.markdown("Aplikasi ini dikembangkan untuk memenuhi tugas kuliah Pengolahan Citra Digital.")
        st.markdown("")
        st.markdown("#### Tujuan Aplikasi")
        st.markdown("- Memahami konsep dasar pengolahan citra digital")
        st.markdown("- Mengimplementasikan berbagai teknik pengolahan citra")
        st.markdown("- Menganalisis efek setiap operasi pada citra")
        st.markdown("- Menyediakan alat interaktif untuk eksperimen")
        st.markdown("")
        st.markdown("#### Fitur Lengkap")
        st.markdown("- Transformasi Geometri (Rotasi, Translasi, Skala, Flip)")
        st.markdown("- Enhancement (Brightness, Contrast, Sharpening)")
        st.markdown("- Histogram & Equalization")
        st.markdown("- Noise & Filtering (Gaussian, Median, Low-pass, High-pass)")
        st.markdown("- FFT (Analisis Frekuensi)")
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0f2b3d 0%, #1a3a4f 100%); border-radius: 16px; padding: 20px; text-align: center;">
            <h3 style="color: white;">Informasi Tugas</h3>
            <hr>
            <p style="color: white;"><strong>Mata Kuliah</strong><br>Pengolahan Citra Digital</p>
            <p style="color: white;"><strong>Semester</strong><br>Genap 2025/2026</p>
            <p style="color: white;"><strong>Platform</strong><br>Streamlit Cloud</p>
        </div>
        """, unsafe_allow_html=True)

# ========================
# 2. TEORI SINGKAT
# ========================
elif menu == "Teori Singkat":
    st.markdown("# TEORI SINGKAT")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Transformasi Geometri", 
        "Enhancement", 
        "Histogram", 
        "Noise & Filtering", 
        "FFT"
    ])
    
    with tab1:
        st.markdown("### Transformasi Geometri")
        st.markdown("Transformasi geometri adalah operasi yang mengubah posisi piksel dalam citra.")
        st.markdown("- **Rotasi**: Memutar gambar dengan sudut tertentu")
        st.markdown("- **Translasi**: Memindahkan posisi gambar")
        st.markdown("- **Skala**: Memperbesar/memperkecil gambar")
        st.markdown("- **Flip**: Membalik gambar (horizontal/vertikal)")
    
    with tab2:
        st.markdown("### Image Enhancement")
        st.markdown("Enhancement bertujuan meningkatkan kualitas visual citra.")
        st.markdown("- **Brightness**: Mengatur kecerahan gambar")
        st.markdown("- **Contrast**: Mengatur perbedaan warna")
        st.markdown("- **Sharpening**: Mempertegas tepi objek")
    
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
        st.markdown("- **Frekuensi Rendah**: Informasi utama, perubahan gradual")
        st.markdown("- **Frekuensi Tinggi**: Detail tepi, noise")

# ========================
# 3. LOAD GAMBAR
# ========================
elif menu == "Load & Tampilkan Gambar":
    st.markdown("# LOAD & TAMPILKAN GAMBAR")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state.gambar_asli = load_image(uploaded_file)
            st.session_state.gambar_proses = st.session_state.gambar_asli.copy()
            st.balloons()
            st.success("Gambar berhasil di-load!")
            h, w = st.session_state.gambar_asli.shape[:2]
            st.info(f"Dimensi: {w} x {h} px | Mode: RGB")
    
    with col2:
        if st.session_state.gambar_asli is not None:
            st.markdown("### Preview Gambar")
            st.image(st.session_state.gambar_asli, caption="Gambar Asli", use_container_width=True)
        else:
            st.info("Belum ada gambar. Silakan upload gambar terlebih dahulu.")
    
    if st.session_state.gambar_asli is not None:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Gambar Asli")
            st.image(st.session_state.gambar_asli, use_container_width=True)
        with col2:
            st.markdown("#### Gambar Proses")
            st.image(st.session_state.gambar_proses, use_container_width=True)

# ========================
# 4. TRANSFORMASI GEOMETRI
# ========================
elif menu == "Transformasi Geometri":
    st.markdown("# TRANSFORMASI GEOMETRI")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            transform = st.selectbox("Pilih Transformasi", ["Rotasi", "Translasi", "Skala", "Flip"])
            
            if transform == "Rotasi":
                angle = st.slider("Sudut Rotasi (derajat):", -180, 180, 0)
                if st.button("Terapkan Rotasi", use_container_width=True):
                    h, w = img.shape[:2]
                    center = (w//2, h//2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    result = cv2.warpAffine(img, M, (w, h))
                    st.session_state.gambar_proses = result
                    st.rerun()
                    
            elif transform == "Translasi":
                tx = st.slider("Translasi X:", -100, 100, 0)
                ty = st.slider("Translasi Y:", -100, 100, 0)
                if st.button("Terapkan Translasi", use_container_width=True):
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    result = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    st.session_state.gambar_proses = result
                    st.rerun()
                    
            elif transform == "Skala":
                scale = st.slider("Faktor Skala:", 0.1, 3.0, 1.0)
                if st.button("Terapkan Skala", use_container_width=True):
                    new_w = int(img.shape[1] * scale)
                    new_h = int(img.shape[0] * scale)
                    result = cv2.resize(img, (new_w, new_h))
                    st.session_state.gambar_proses = result
                    st.rerun()
                    
            else:
                flip_code = st.selectbox("Arah Flip:", ["Horizontal", "Vertikal"])
                if st.button("Terapkan Flip", use_container_width=True):
                    if flip_code == "Horizontal":
                        result = cv2.flip(img, 1)
                    else:
                        result = cv2.flip(img, 0)
                    st.session_state.gambar_proses = result
                    st.rerun()
        
        with col2:
            st.markdown("### Hasil Transformasi")
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(img, caption="Sebelum", use_container_width=True)
            with col_b:
                st.image(st.session_state.gambar_proses, caption="Sesudah", use_container_width=True)
    else:
        st.warning("Belum ada gambar! Silakan load gambar terlebih dahulu.")

# ========================
# 5. ENHANCEMENT
# ========================
elif menu == "Enhancement":
    st.markdown("# ENHANCEMENT")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        pil_img = Image.fromarray(img)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            enhancement_type = st.selectbox("Pilih Enhancement", ["Brightness", "Contrast", "Sharpening"])
            
            if enhancement_type == "Brightness":
                factor = st.slider("Faktor Kecerahan:", 0.5, 2.0, 1.0)
                if st.button("Terapkan Brightness", use_container_width=True):
                    enhancer = ImageEnhance.Brightness(pil_img)
                    result = np.array(enhancer.enhance(factor))
                    st.session_state.gambar_proses = result
                    st.rerun()
                    
            elif enhancement_type == "Contrast":
                factor = st.slider("Faktor Kontras:", 0.5, 2.0, 1.0)
                if st.button("Terapkan Contrast", use_container_width=True):
                    enhancer = ImageEnhance.Contrast(pil_img)
                    result = np.array(enhancer.enhance(factor))
                    st.session_state.gambar_proses = result
                    st.rerun()
                    
            else:
                if st.button("Terapkan Sharpening", use_container_width=True):
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    result = cv2.filter2D(img, -1, kernel)
                    st.session_state.gambar_proses = result
                    st.rerun()
        
        with col2:
            st.markdown("### Hasil Enhancement")
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(img, caption="Sebelum", use_container_width=True)
            with col_b:
                st.image(st.session_state.gambar_proses, caption="Sesudah", use_container_width=True)
    else:
        st.warning("Belum ada gambar! Silakan load gambar terlebih dahulu.")

# ========================
# 6. HISTOGRAM
# ========================
elif menu == "Histogram":
    st.markdown("# HISTOGRAM & EQUALIZATION")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        fig = plt.figure(figsize=(14, 6))
        fig.patch.set_facecolor('#f8fafc')
        
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Citra Grayscale", fontsize=12, fontweight='bold', color='#0f2b3d')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.hist(gray.ravel(), bins=256, color='#2ecc71', alpha=0.7)
        plt.title("Histogram Asli", fontsize=12, fontweight='bold', color='#0f2b3d')
        plt.xlabel("Intensitas Piksel")
        plt.ylabel("Frekuensi")
        plt.grid(True, alpha=0.3)
        
        equalized = cv2.equalizeHist(gray)
        plt.subplot(1, 3, 3)
        plt.hist(equalized.ravel(), bins=256, color='#1a5276', alpha=0.7)
        plt.title("Histogram Setelah Equalization", fontsize=12, fontweight='bold', color='#0f2b3d')
        plt.xlabel("Intensitas Piksel")
        plt.ylabel("Frekuensi")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray, caption="Sebelum Equalization", use_container_width=True)
        with col2:
            st.image(equalized, caption="Setelah Equalization", use_container_width=True)
        
        if st.button("Terapkan Equalization ke Gambar Proses", use_container_width=True):
            st.session_state.gambar_proses = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            st.success("Equalization berhasil diterapkan!")
            st.rerun()
    else:
        st.warning("Belum ada gambar! Silakan load gambar terlebih dahulu.")

# ========================
# 7. NOISE & FILTERING
# ========================
elif menu == "Noise & Filtering":
    st.markdown("# NOISE & FILTERING")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            operation = st.selectbox("Pilih Operasi", ["Tambah Noise", "Terapkan Filter"])
            
            if operation == "Tambah Noise":
                noise_type = st.selectbox("Jenis Noise", ["Gaussian", "Salt & Pepper"])
                if st.button("Tambahkan Noise", use_container_width=True):
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
                filter_type = st.selectbox("Pilih Filter", ["Gaussian", "Median", "Low-pass", "High-pass"])
                if st.button("Terapkan Filter", use_container_width=True):
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
            st.markdown("### Hasil Operasi")
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(img, caption="Sebelum", use_container_width=True)
            with col_b:
                st.image(st.session_state.gambar_proses, caption="Sesudah", use_container_width=True)
    else:
        st.warning("Belum ada gambar! Silakan load gambar terlebih dahulu.")

# ========================
# 8. FFT (ANALISIS FREKUENSI)
# ========================
elif menu == "FFT (Analisis Frekuensi)":
    st.markdown("# FFT - ANALISIS FREKUENSI")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        
        fig = plt.figure(figsize=(15, 5))
        fig.patch.set_facecolor('#f8fafc')
        
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Citra Grayscale", fontsize=12, fontweight='bold', color='#0f2b3d')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title("Magnitude Spectrum (FFT)", fontsize=12, fontweight='bold', color='#0f2b3d')
        plt.axis('off')
        
        f_ishift = ifftshift(fshift)
        img_back = ifft2(f_ishift)
        img_back = np.abs(img_back)
        plt.subplot(1, 3, 3)
        plt.imshow(img_back, cmap='gray')
        plt.title("Hasil Inverse FFT", fontsize=12, fontweight='bold', color='#0f2b3d')
        plt.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        with st.expander("Penjelasan FFT"):
            st.markdown("- **Magnitude Spectrum**: Menunjukkan distribusi energi frekuensi dalam citra")
            st.markdown("- **Titik terang di tengah**: Komponen frekuensi rendah (informasi utama)")
            st.markdown("- **Semakin jauh dari pusat**: Frekuensi tinggi (detail tepi, noise)")
            st.markdown("- **Inverse FFT**: Mengembalikan citra ke domain spasial")
    else:
        st.warning("Belum ada gambar! Silakan load gambar terlebih dahulu.")

# ========================
# 9. IMPLEMENTASI
# ========================
elif menu == "Implementasi":
    st.markdown("# IMPLEMENTASI")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Tools & Libraries")
        st.markdown("- **Streamlit**: Framework web interaktif")
        st.markdown("- **OpenCV**: Operasi citra (transformasi, filter, noise)")
        st.markdown("- **PIL/Pillow**: Enhancement dasar")
        st.markdown("- **NumPy**: Manipulasi array & FFT")
        st.markdown("- **Matplotlib**: Visualisasi histogram & spektrum FFT")
        st.markdown("- **SciPy**: FFT lanjutan")
    
    with col2:
        st.markdown("### Struktur Project")
        st.code("""
PCD-STUDIO/
├── streamlit_app.py
├── requirements.txt
└── README.md
        """)
    
    st.markdown("---")
    st.markdown("### Kode Utama")
    st.code("""
import cv2
import numpy as np
from scipy.fft import fft2, fftshift

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
    st.markdown("# HASIL & PEMBAHASAN")
    
    st.markdown("### Observasi & Analisis")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1. Transformasi Geometri")
        st.markdown("**Hasil:** Rotasi mempertahankan informasi visual namun terjadi pemotongan di tepi.")
        st.markdown("**Analisis:** Transformasi geometri efektif untuk koreksi posisi dan orientasi citra.")
        st.markdown("")
        
        st.markdown("#### 2. Enhancement")
        st.markdown("**Hasil:** Brightness meningkatkan kecerahan global. Sharpening mempertegas tepi objek.")
        st.markdown("**Analisis:** Enhancement berguna untuk memperbaiki kualitas visual citra.")
        st.markdown("")
        
        st.markdown("#### 3. Histogram")
        st.markdown("**Hasil:** Citra gelap memiliki histogram terkonsentrasi di kiri.")
        st.markdown("**Analisis:** Equalization efektif untuk meningkatkan kontras citra secara otomatis.")
    
    with col2:
        st.markdown("#### 4. Noise & Filtering")
        st.markdown("**Hasil:** Gaussian noise diatasi dengan Gaussian filter.")
        st.markdown("**Analisis:** Pemilihan filter harus disesuaikan dengan jenis noise.")
        st.markdown("")
        
        st.markdown("#### 5. FFT")
        st.markdown("**Hasil:** Spektrum frekuensi menunjukkan komponen frekuensi rendah di pusat.")
        st.markdown("**Analisis:** FFT berguna untuk analisis dan filtering di domain frekuensi.")
        st.markdown("")
        
        st.markdown("#### Kesimpulan Analisis")
        st.markdown("Semua teknik pengolahan citra berhasil diimplementasikan dan berfungsi dengan baik.")
    
    st.markdown("---")
    st.info("Screenshot hasil pengujian setiap fitur dapat ditambahkan di sini untuk dokumentasi tugas.")

# ========================
# 11. KESIMPULAN
# ========================
elif menu == "Kesimpulan":
    st.markdown("# KESIMPULAN")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h1 style="font-size: 2rem;">✅</h1>
            <h4>TERIMPLEMENTASI</h4>
            <p>Semua fitur berhasil diimplementasikan</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h1 style="font-size: 2rem;">🎯</h1>
            <h4>TUJUAN TERCAPAI</h4>
            <p>Aplikasi interaktif untuk pembelajaran</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h1 style="font-size: 2rem;">🚀</h1>
            <h4>SIAP DEPLOY</h4>
            <p>Aplikasi siap digunakan kapan saja</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Ringkasan Kesimpulan")
    st.markdown("1. Aplikasi berhasil mengimplementasikan 5 teknik utama pengolahan citra digital")
    st.markdown("2. Antarmuka interaktif memudahkan eksperimen dan pembelajaran")
    st.markdown("3. Semua fitur berfungsi dengan baik sesuai teori pengolahan citra")
    st.markdown("4. Aplikasi dapat diakses secara online melalui Streamlit Cloud")
    st.markdown("5. Hasil pengujian menunjukkan setiap operasi berjalan sesuai parameter")
    
    st.markdown("")
    st.markdown("### Saran Pengembangan Lebih Lanjut")
    st.markdown("- Implementasi deteksi tepi (Canny, Sobel, Prewitt)")
    st.markdown("- Segmentasi citra dengan thresholding adaptif")
    st.markdown("- Ekstraksi fitur untuk klasifikasi gambar")
    st.markdown("- Integrasi model deep learning (CNN)")

# ========================
# FOOTER
# ========================
st.markdown("""
<div class="footer">
    <p><strong>PCD STUDIO</strong> | Pengolahan Citra Digital | Tugas Akhir Semester | 2026</p>
    <p style="font-size: 12px;">Dibangun dengan Streamlit, OpenCV, dan Python</p>
</div>
""", unsafe_allow_html=True)
