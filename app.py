import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from io import BytesIO
import base64

# ========================
# KONFIGURASI HALAMAN
# ========================
st.set_page_config(
    page_title="PCD Studio - Pengolahan Citra Digital",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# CUSTOM CSS
# ========================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card style untuk konten */
    .css-1r6slb0, .css-1v0mbdj, .stMarkdown {
        background-color: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Judul utama */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: bold !important;
        text-align: center;
        margin-bottom: 30px !important;
    }
    
    /* Subheader */
    h2, h3 {
        color: #4a5568 !important;
        border-left: 4px solid #667eea;
        padding-left: 15px;
        margin-top: 20px !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Info box */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Image container */
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metric card */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: white;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# SIDEBAR NAVIGASI DENGAN ICON
# ========================
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h2 style="color: white; border-left: none;">🎨 PCD Studio</h2>
    <p style="color: #a0aec0;">Pengolahan Citra Digital</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Menu dengan icon yang lebih menarik
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
    "📌 **Pilih Menu**",
    list(menu_icons.keys()),
    format_func=lambda x: f"{menu_icons[x]} {x}"
)

# Session state
if 'gambar_asli' not in st.session_state:
    st.session_state.gambar_asli = None
if 'gambar_proses' not in st.session_state:
    st.session_state.gambar_proses = None
if 'history' not in st.session_state:
    st.session_state.history = []

# Fungsi baca gambar
def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Fungsi untuk reset gambar
def reset_to_original():
    if st.session_state.gambar_asli is not None:
        st.session_state.gambar_proses = st.session_state.gambar_asli.copy()
        st.success("✅ Gambar berhasil di-reset ke original!")

# Status di sidebar
st.sidebar.markdown("---")
if st.session_state.gambar_asli is not None:
    st.sidebar.success(f"✅ **Status:** Gambar tersedia")
    h, w = st.session_state.gambar_asli.shape[:2]
    st.sidebar.info(f"📐 **Dimensi:** {w} x {h} px")
    
    if st.sidebar.button("🔄 Reset ke Gambar Asli", use_container_width=True):
        reset_to_original()
else:
    st.sidebar.warning("⚠️ **Status:** Belum ada gambar")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: #a0aec0; font-size: 12px;">
    <p>Dibangun dengan ❤️ menggunakan Streamlit</p>
    <p>Tugas Pengolahan Citra Digital</p>
    <p>© 2026</p>
</div>
""", unsafe_allow_html=True)

# ========================
# 1. PENDAHULUAN
# ========================
if menu == "Pendahuluan":
    st.markdown("# 📖 Pendahuluan")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Selamat Datang di PCD Studio!
        
        Aplikasi ini dikembangkan untuk memenuhi tugas kuliah **Pengolahan Citra Digital**.
        
        #### 🎯 Tujuan Aplikasi
        - Memahami konsep dasar pengolahan citra digital
        - Mengimplementasikan berbagai teknik pengolahan citra
        - Menganalisis efek setiap operasi pada citra
        - Menyediakan alat interaktif untuk eksperimen
        
        #### ✨ Fitur Lengkap
        - 🔄 **Transformasi Geometri** (Rotasi, Translasi, Skala, Flip)
        - ✨ **Enhancement** (Brightness, Contrast, Sharpening)
        - 📊 **Histogram** & Equalization
        - 🎛️ **Noise & Filtering** (Gaussian, Median, Low-pass, High-pass)
        - 🌊 **FFT** (Analisis Frekuensi)
        """)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 20px; text-align: center; color: white;">
            <h3 style="color: white;">🎓</h3>
            <h4 style="color: white;">Informasi Tugas</h4>
            <hr>
            <p><strong>Mata Kuliah</strong><br>Pengolahan Citra Digital</p>
            <p><strong>Semester</strong><br>Genap 2025/2026</p>
        </div>
        """, unsafe_allow_html=True)

# ========================
# 2. TEORI SINGKAT
# ========================
elif menu == "Teori Singkat":
    st.markdown("# 📚 Teori Singkat")
    
    tabs = st.tabs(["🎯 Transformasi Geometri", "✨ Enhancement", "📊 Histogram", "🎛️ Noise & Filtering", "🌊 FFT"])
    
    with tabs[0]:
        st.markdown("""
        ### Transformasi Geometri
        
        Transformasi geometri adalah operasi yang mengubah posisi piksel dalam citra.
        
        | Operasi | Rumus | Keterangan |
        |---------|-------|-------------|
        | **Rotasi** | x' = x cos θ - y sin θ | Memutar gambar |
        | **Translasi** | x' = x + dx, y' = y + dy | Memindahkan posisi |
        | **Skala** | x' = sx · x, y' = sy · y | Mengubah ukuran |
        | **Flip** | x' = -x atau y' = -y | Membalik gambar |
        """)
    
    with tabs[1]:
        st.markdown("""
        ### Image Enhancement
        
        Enhancement bertujuan meningkatkan kualitas visual citra.
        
        - **Brightness**: I'(x,y) = I(x,y) + c
        - **Contrast**: I'(x,y) = α·I(x,y) + β
        - **Sharpening**: Menggunakan kernel high-pass filter
        """)
    
    with tabs[2]:
        st.markdown("""
        ### Histogram Citra
        
        Histogram adalah grafik yang menunjukkan distribusi intensitas piksel.
        
        **Histogram Equalization**:
        - Menyebarkan intensitas secara merata
        - Meningkatkan kontras gambar
        - Rumus: s_k = T(r_k) = Σ(n_j/n)
        """)
    
    with tabs[3]:
        st.markdown("""
        ### Noise & Filtering
        
        **Jenis Noise:**
        - Gaussian: Noise dengan distribusi normal
        - Salt & Pepper: Titik acak hitam/putih
        
        **Filter:**
        - Median Filter: Baik untuk salt & pepper noise
        - Gaussian Filter: Baik untuk Gaussian noise
        """)
    
    with tabs[4]:
        st.markdown("""
        ### Fast Fourier Transform (FFT)
        
        FFT mengubah citra dari domain spasial ke domain frekuensi.
        
        - **Frekuensi Rendah**: Informasi utama, perubahan gradual
        - **Frekuensi Tinggi**: Detail tepi, noise
        - **Magnitude Spectrum**: Visualisasi energi frekuensi
        """)

# ========================
# 3. LOAD GAMBAR
# ========================
elif menu == "Load & Tampilkan Gambar":
    st.markdown("# 📂 Load & Tampilkan Gambar")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📤 **Upload gambar** (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload gambar yang akan diproses"
        )
        
        if uploaded_file is not None:
            st.session_state.gambar_asli = load_image(uploaded_file)
            st.session_state.gambar_proses = st.session_state.gambar_asli.copy()
            st.balloons()
            st.success("✅ **Gambar berhasil di-load!**")
            
            # Tampilkan info gambar
            h, w = st.session_state.gambar_asli.shape[:2]
            st.info(f"📐 **Dimensi:** {w} x {h} px | 🎨 **Mode:** RGB")
    
    with col2:
        if st.session_state.gambar_asli is not None:
            st.markdown("### 🖼️ Preview Gambar")
            st.image(st.session_state.gambar_asli, caption="Gambar Asli", use_container_width=True)
        else:
            st.markdown("""
            <div style="background: #f0f0f0; border-radius: 10px; padding: 50px; text-align: center;">
                <h3>📷</h3>
                <p>Belum ada gambar yang di-load</p>
                <p style="font-size: 12px;">Silakan upload gambar terlebih dahulu</p>
            </div>
            """, unsafe_allow_html=True)
    
    if st.session_state.gambar_asli is not None:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.gambar_asli, caption="🟢 Gambar Asli", use_container_width=True)
        with col2:
            st.image(st.session_state.gambar_proses, caption="🟡 Gambar Proses", use_container_width=True)

# ========================
# 4. TRANSFORMASI GEOMETRI
# ========================
elif menu == "Transformasi Geometri":
    st.markdown("# 🔄 Transformasi Geometri")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            transform = st.selectbox(
                "🎯 **Pilih Transformasi**",
                ["Rotasi", "Translasi", "Skala", "Flip"]
            )
            
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
            st.markdown("### Hasil Transformasi")
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(img, caption="Sebelum", use_container_width=True)
            with col_b:
                st.image(st.session_state.gambar_proses, caption="Sesudah", use_container_width=True)
    else:
        st.warning("⚠️ **Belum ada gambar!** Silakan load gambar terlebih dahulu di menu 'Load & Tampilkan Gambar'.")

# ========================
# 5. ENHANCEMENT
# ========================
elif menu == "Enhancement":
    st.markdown("# ✨ Enhancement")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        pil_img = Image.fromarray(img)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            enhancement_type = st.selectbox(
                "🎨 **Pilih Enhancement**",
                ["Brightness", "Contrast", "Sharpening"]
            )
            
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
            st.markdown("### Hasil Enhancement")
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(img, caption="Sebelum", use_container_width=True)
            with col_b:
                st.image(st.session_state.gambar_proses, caption="Sesudah", use_container_width=True)
    else:
        st.warning("⚠️ **Belum ada gambar!** Silakan load gambar terlebih dahulu.")

# ========================
# 6. HISTOGRAM
# ========================
elif menu == "Histogram":
    st.markdown("# 📊 Histogram & Equalization")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        fig = plt.figure(figsize=(14, 6))
        
        # Plot 1: Histogram Asli
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Citra Grayscale", fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Plot 2: Histogram
        plt.subplot(1, 3, 2)
        plt.hist(gray.ravel(), bins=256, color='#667eea', alpha=0.7, edgecolor='black')
        plt.title("Histogram Asli", fontsize=12, fontweight='bold')
        plt.xlabel("Intensitas Piksel")
        plt.ylabel("Frekuensi")
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Equalized
        equalized = cv2.equalizeHist(gray)
        plt.subplot(1, 3, 3)
        plt.hist(equalized.ravel(), bins=256, color='#764ba2', alpha=0.7, edgecolor='black')
        plt.title("Histogram Setelah Equalization", fontsize=12, fontweight='bold')
        plt.xlabel("Intensitas Piksel")
        plt.ylabel("Frekuensi")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray, caption="Citra Grayscale Sebelum", use_container_width=True)
        with col2:
            st.image(equalized, caption="Citra Setelah Equalization", use_container_width=True)
        
        if st.button("✅ Terapkan Equalization ke Gambar Proses", use_container_width=True):
            st.session_state.gambar_proses = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            st.success("Equalization berhasil diterapkan!")
            st.rerun()
    else:
        st.warning("⚠️ **Belum ada gambar!** Silakan load gambar terlebih dahulu.")

# ========================
# 7. NOISE & FILTERING
# ========================
elif menu == "Noise & Filtering":
    st.markdown("# 🎛️ Noise & Filtering")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            operation = st.selectbox("🔧 **Pilih Operasi**", ["Tambah Noise", "Terapkan Filter"])
            
            if operation == "Tambah Noise":
                noise_type = st.selectbox("📡 **Jenis Noise**", ["Gaussian", "Salt & Pepper"])
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
                filter_type = st.selectbox("🔍 **Pilih Filter**", ["Gaussian", "Median", "Low-pass", "High-pass"])
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
            st.markdown("### Hasil Operasi")
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(img, caption="Sebelum", use_container_width=True)
            with col_b:
                st.image(st.session_state.gambar_proses, caption="Sesudah", use_container_width=True)
    else:
        st.warning("⚠️ **Belum ada gambar!** Silakan load gambar terlebih dahulu.")

# ========================
# 8. FFT
# ========================
elif menu == "FFT (Analisis Frekuensi)":
    st.markdown("# 🌊 FFT - Analisis Frekuensi")
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        
        fig = plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Citra Grayscale", fontsize=12, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title("Magnitude Spectrum (FFT)", fontsize=12, fontweight='bold')
        plt.axis('off')
        
        f_ishift = ifftshift(fshift)
        img_back = ifft2(f_ishift)
        img_back = np.abs(img_back)
        plt.subplot(1, 3, 3)
        plt.imshow(img_back, cmap='gray')
        plt.title("Hasil Inverse FFT", fontsize=12, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        with st.expander("📖 **Penjelasan FFT**"):
            st.markdown("""
            - **Magnitude Spectrum**: Menunjukkan distribusi energi frekuensi dalam citra
            - **Titik terang di tengah**: Komponen frekuensi rendah (informasi utama)
            - **Semakin jauh dari pusat**: Frekuensi tinggi (detail tepi, noise)
            - **Inverse FFT**: Mengembalikan citra ke domain spasial
            """)
    else:
        st.warning("⚠️ **Belum ada gambar!** Silakan load gambar terlebih dahulu.")

# ========================
# 9. IMPLEMENTASI
# ========================
elif menu == "Implementasi":
    st.markdown("# 💻 Implementasi")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🛠️ Tools & Libraries
        
        | Library | Fungsi |
        |---------|--------|
        | **Streamlit** | Framework web interaktif |
        | **OpenCV** | Operasi citra |
        | **Pillow** | Enhancement dasar |
        | **NumPy** | Manipulasi array |
        | **Matplotlib** | Visualisasi |
        | **SciPy** | FFT |
        """)
    
    with col2:
        st.markdown("""
        ### 📁 Struktur Kode
        
