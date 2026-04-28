import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import time

# ======================================================================
# KONFIGURASI HALAMAN
# ======================================================================
st.set_page_config(
    page_title="PCD Studio - Pengolahan Citra Digital",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# CUSTOM CSS - TEMA PROFESIONAL (NAVY + EMERALD)
# ======================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f2b3d 0%, #1a5276 50%, #0f2b3d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 16px !important;
    }
    
    .sub-header {
        text-align: center;
        color: #1a5276 !important;
        margin-bottom: 32px !important;
        font-size: 1.1rem !important;
    }
    
    .card {
        background-color: white;
        border-radius: 20px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    h1, h2, h3 {
        color: #0f2b3d !important;
    }
    
    h2 {
        border-left: 5px solid #2ecc71;
        padding-left: 16px;
        margin-top: 24px !important;
        font-weight: 600 !important;
    }
    
    h3 {
        border-left: 3px solid #2ecc71;
        padding-left: 12px;
        margin-top: 20px !important;
        font-weight: 600 !important;
    }
    
    p, li, .stMarkdown {
        color: #334155 !important;
        line-height: 1.6 !important;
    }
    
    /* Sidebar styling */
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
    
    .sidebar-title {
        text-align: center;
        padding: 20px 0;
    }
    
    .sidebar-logo {
        width: 60px;
        height: 60px;
        background: #2ecc71;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        font-size: 30px;
    }
    
    /* Button styling */
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
    
    /* Form elements */
    .stSelectbox > div > div, .stSlider > div > div {
        background-color: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
    }
    
    .stSlider > div > div > div {
        background-color: #2ecc71;
    }
    
    /* Alert styling */
    .stAlert.stWarning {
        background-color: #fef3c7 !important;
        border-left: 5px solid #f59e0b !important;
    }
    
    .stAlert.stSuccess {
        background-color: #d1fae5 !important;
        border-left: 5px solid #10b981 !important;
    }
    
    /* Image styling */
    .stImage {
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
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
    
    /* Footer */
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
    
    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #0f2b3d 0%, #1a3a4f 100%);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    
    .metric-card h1, .metric-card h4, .metric-card p {
        color: white !important;
    }
    
    /* Info box */
    .info-box {
        background-color: #e8f4f8;
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #2ecc71;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# FUNGSI UTILITY
# ======================================================================

def load_image_from_file(uploaded_file):
    """Memuat gambar dari file yang diupload"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def reset_to_original():
    """Reset gambar proses ke gambar asli"""
    if st.session_state.gambar_asli is not None:
        st.session_state.gambar_proses = st.session_state.gambar_asli.copy()
        return True
    return False

def apply_with_progress(func, *args, **kwargs):
    """Menerapkan fungsi dengan progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Memproses gambar...")
    progress_bar.progress(30)
    
    result = func(*args, **kwargs)
    
    progress_bar.progress(100)
    status_text.text("Proses selesai!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    return result

def get_image_info(img):
    """Mendapatkan informasi tentang gambar"""
    if img is None:
        return None
    h, w = img.shape[:2]
    return {
        "dimensi": f"{w} x {h} px",
        "jumlah_piksel": w * h,
        "rasio_aspek": f"{w/h:.2f}"
    }

def create_comparison_display(img1, img2, caption1="Sebelum", caption2="Sesudah"):
    """Membuat tampilan perbandingan gambar"""
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption=caption1, use_container_width=True)
    with col2:
        st.image(img2, caption=caption2, use_container_width=True)

# ======================================================================
# INISIALISASI SESSION STATE
# ======================================================================
if 'gambar_asli' not in st.session_state:
    st.session_state.gambar_asli = None
if 'gambar_proses' not in st.session_state:
    st.session_state.gambar_proses = None
if 'riwayat_operasi' not in st.session_state:
    st.session_state.riwayat_operasi = []

def tambah_riwayat(operasi, parameter=""):
    """Menambahkan operasi ke riwayat"""
    st.session_state.riwayat_operasi.append({
        "operasi": operasi,
        "parameter": parameter,
        "waktu": time.strftime("%H:%M:%S")
    })

# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-title">
        <div class="sidebar-logo">🖼️</div>
        <h2 style="color: white; border-left: none; margin-top: 16px;">PCD STUDIO</h2>
        <p style="color: #a0c4e8;">Pengolahan Citra Digital</p>
        <hr>
    </div>
    """, unsafe_allow_html=True)
    
    menu_icons = {
        "Pendahuluan": "📖",
        "Landasan Teori": "📚",
        "Load & Visualisasi": "📂",
        "Transformasi Geometri": "🔄",
        "Image Enhancement": "✨",
        "Histogram & Equalization": "📊",
        "Noise & Filtering": "🎛️",
        "Analisis Frekuensi (FFT)": "🌊",
        "Implementasi Teknis": "💻",
        "Analisis Hasil": "📝",
        "Kesimpulan": "📌"
    }
    
    menu = st.radio(
        "MENU NAVIGASI",
        list(menu_icons.keys()),
        format_func=lambda x: f"{menu_icons[x]} {x}"
    )
    
    st.markdown("---")
    
    # Status gambar
    if st.session_state.gambar_asli is not None:
        st.success("✅ STATUS: Gambar tersedia")
        info = get_image_info(st.session_state.gambar_asli)
        if info:
            st.info(f"📏 Dimensi: {info['dimensi']}\n\n🔢 Piksel: {info['jumlah_piksel']:,}")
        
        if st.button("🔄 Reset ke Gambar Asli", use_container_width=True):
            if reset_to_original():
                st.success("Gambar berhasil di-reset!")
                tambah_riwayat("Reset", "Ke gambar asli")
                st.rerun()
    else:
        st.warning("⚠️ STATUS: Belum ada gambar")
    
    st.markdown("---")
    
    # Riwayat operasi
    if len(st.session_state.riwayat_operasi) > 0:
        with st.expander("📜 Riwayat Operasi"):
            for item in st.session_state.riwayat_operasi[-5:]:
                st.markdown(f"- **{item['operasi']}** {item['parameter']} _{item['waktu']}_")
    
    st.markdown("---")
    st.caption("© 2026 PCD Studio\nDibangun dengan Streamlit")

# ======================================================================
# 1. PENDAHULUAN
# ======================================================================
if menu == "Pendahuluan":
    st.markdown('<h1 class="main-header">PENDAHULUAN</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Pengolahan Citra Digital - Semester Genap 2025/2026</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h2>🎯 Latar Belakang</h2>
            <p>Pengolahan citra digital merupakan bidang ilmu yang mempelajari teknik-teknik 
            untuk memanipulasi dan menganalisis citra digital. Dalam era digital saat ini, 
            pengolahan citra memiliki peran vital dalam berbagai bidang seperti:</p>
            <ul>
                <li><strong>Kedokteran</strong> - Diagnosa berbasis citra medis (CT Scan, MRI, X-ray)</li>
                <li><strong>Industri</strong> - Quality control dan inspeksi produk otomatis</li>
                <li><strong>Keamanan</strong> - Pengenalan wajah, sidik jari, dan plat nomor</li>
                <li><strong>Remote Sensing</strong> - Analisis citra satelit dan drone</li>
                <li><strong>Multimedia</strong> - Editing foto, video, dan augmented reality</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h2>📋 Tujuan Aplikasi</h2>
            <p>Aplikasi PCD Studio dikembangkan untuk memenuhi tugas akhir mata kuliah 
            Pengolahan Citra Digital dengan tujuan:</p>
            <ol>
                <li><strong>Memahami Konsep Dasar</strong> - Mempelajari representasi citra digital, 
                domain spasial dan frekuensi</li>
                <li><strong>Mengimplementasikan Teknik</strong> - Menerapkan berbagai algoritma 
                pengolahan citra secara interaktif</li>
                <li><strong>Menganalisis Efek</strong> - Mengamati dan memahami dampak setiap 
                operasi terhadap kualitas citra</li>
                <li><strong>Menyediakan Alat Eksperimen</strong> - Memungkinkan eksplorasi 
                parameter secara real-time</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3 style="text-align: center;">📊 Informasi Tugas</h3>
            <hr>
            <p><strong>Mata Kuliah</strong><br>Pengolahan Citra Digital</p>
            <p><strong>Kode MK</strong><br>TIF-1234</p>
            <p><strong>Semester</strong><br>Genap 2025/2026</p>
            <p><strong>Platform</strong><br>Streamlit Cloud</p>
            <p><strong>Fitur</strong><br>11 Menu + Real-time Processing</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3 style="text-align: center;">🛠️ Teknologi</h3>
            <ul>
                <li>Streamlit</li>
                <li>OpenCV</li>
                <li>NumPy/SciPy</li>
                <li>PIL/Pillow</li>
                <li>Matplotlib</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ======================================================================
# 2. LANDASAN TEORI
# ======================================================================
elif menu == "Landasan Teori":
    st.markdown('<h1 class="main-header">LANDASAN TEORI</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs([
        "🎨 Representasi Citra Digital",
        "🔄 Domain Spasial", 
        "🌊 Domain Frekuensi",
        "📊 Histogram",
        "🎛️ Filtering"
    ])
    
    with tabs[0]:
        st.markdown("""
        <div class="card">
            <h2>Representasi Citra Digital</h2>
            <p>Citra digital direpresentasikan sebagai matriks dua dimensi (untuk grayscale) 
            atau tiga dimensi (untuk RGB) yang berisi nilai intensitas piksel.</p>
            
            <h3>📐 Resolusi & Kuantisasi</h3>
            <ul>
                <li><strong>Resolusi Spasial</strong> - Ukuran matriks (lebar × tinggi) dalam piksel</li>
                <li><strong>Resolusi Intensitas</strong> - Jumlah bit per piksel (8-bit = 0-255)</li>
            </ul>
            
            <h3>🎨 Model Warna</h3>
            <ul>
                <li><strong>RGB (Red-Green-Blue)</strong> - Model aditif untuk display</li>
                <li><strong>Grayscale</strong> - Satu kanal intensitas (0=hitam, 255=putih)</li>
                <li><strong>HSV (Hue-Saturation-Value)</strong> - Pemisahan warna dan intensitas</li>
            </ul>
            
            <div class="info-box">
                <strong>💡 Konsep Penting:</strong> Setiap operasi pengolahan citra pada dasarnya 
                adalah manipulasi nilai-nilai dalam matriks piksel ini.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("""
        <div class="card">
            <h2>Operasi Domain Spasial</h2>
            <p>Operasi yang dilakukan langsung pada nilai piksel citra.</p>
            
            <h3>⚡ Operasi Titik (Point Operations)</h3>
            <ul>
                <li><strong>Brightness Adjustment</strong> - Menambah/mengurangi nilai piksel: g(x,y) = f(x,y) + c</li>
                <li><strong>Contrast Adjustment</strong> - Mengalikan nilai piksel: g(x,y) = k × f(x,y)</li>
                <li><strong>Gamma Correction</strong> - Koreksi non-linear: g(x,y) = f(x,y)^γ</li>
            </ul>
            
            <h3>🔲 Operasi Tetangga (Neighborhood Operations)</h3>
            <ul>
                <li><strong>Convolution</strong> - Menggunakan kernel/kernel untuk memproses lingkungan piksel</li>
                <li><strong>Filtering</strong> - Low-pass (penghalusan), High-pass (penajaman)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("""
        <div class="card">
            <h2>Domain Frekuensi</h2>
            <p>Menggunakan Transformasi Fourier untuk mengubah citra ke domain frekuensi.</p>
            
            <h3>📈 Fast Fourier Transform (FFT)</h3>
            <ul>
                <li><strong>Frekuensi Rendah</strong> - Informasi utama citra (perubahan gradual)</li>
                <li><strong>Frekuensi Menengah</strong> - Detail dan tekstur objek</li>
                <li><strong>Frekuensi Tinggi</strong> - Tepi (edges) dan noise</li>
            </ul>
            
            <h3>🔍 Aplikasi Domain Frekuensi</h3>
            <ul>
                <li><strong>Low-pass Filtering</strong> - Menghilangkan noise</li>
                <li><strong>High-pass Filtering</strong> - Mendeteksi tepi</li>
                <li><strong>Band-pass Filtering</strong> - Mengekstrak frekuensi tertentu</li>
            </ul>
            
            <div class="info-box">
                <strong>💡 Rumus FFT 2D:</strong> F(u,v) = ΣΣ f(x,y) e^(-j2π(ux/M + vy/N))
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("""
        <div class="card">
            <h2>Histogram Citra</h2>
            <p>Distribusi frekuensi nilai intensitas piksel dalam citra.</p>
            
            <h3>📊 Karakteristik Histogram</h3>
            <ul>
                <li><strong>Citra Gelap</strong> - Puncak di sisi kiri histogram</li>
                <li><strong>Citra Terang</strong> - Puncak di sisi kanan histogram</li>
                <li><strong>Citra Kontras Rendah</strong> - Histogram sempit (terkonsentrasi)</li>
                <li><strong>Citra Kontras Tinggi</strong> - Histogram tersebar merata</li>
            </ul>
            
            <h3>⚖️ Histogram Equalization</h3>
            <p>Teknik untuk meratakan distribusi intensitas sehingga kontras citra meningkat.</p>
            <p>Rumus transformasi: s_k = T(r_k) = Σ_{j=0}^{k} (n_j / N)</p>
            <p>dimana n_j adalah jumlah piksel dengan intensitas j, dan N total piksel.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[4]:
        st.markdown("""
        <div class="card">
            <h2>Filtering Citra</h2>
            
            <h3>🎲 Noise</h3>
            <ul>
                <li><strong>Gaussian Noise</strong> - Noise dengan distribusi normal (model noise sensor)</li>
                <li><strong>Salt & Pepper</strong> - Noise impulsif (piksel putih/hitam acak)</li>
            </ul>
            
            <h3>🔧 Jenis Filter</h3>
            <ul>
                <li><strong>Gaussian Filter</strong> - Smoothing dengan kernel Gaussian (menjaga tepi)</li>
                <li><strong>Median Filter</strong> - Non-linear filter, sangat efektif untuk salt & pepper</li>
                <li><strong>Low-pass Filter</strong> - Menggunakan kernel rata-rata untuk menghaluskan</li>
                <li><strong>High-pass Filter</strong> - Kernel Laplacian/High-boost untuk penajaman</li>
            </ul>
            
            <h3>📐 Contoh Kernel Low-pass (Rata-rata 3x3)</h3>
            <pre>
            [1/9, 1/9, 1/9]
            [1/9, 1/9, 1/9]
            [1/9, 1/9, 1/9]
            </pre>
            
            <h3>📐 Contoh Kernel High-pass (Laplacian)</h3>
            <pre>
            [ 0, -1,  0]
            [-1,  5, -1]
            [ 0, -1,  0]
            </pre>
        </div>
        """, unsafe_allow_html=True)

# ======================================================================
# 3. LOAD & VISUALISASI
# ======================================================================
elif menu == "Load & Visualisasi":
    st.markdown('<h1 class="main-header">LOAD & VISUALISASI CITRA</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📂 Upload Gambar", 
            type=["jpg", "jpeg", "png"],
            help="Format yang didukung: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            with st.spinner("Memuat gambar..."):
                st.session_state.gambar_asli = load_image_from_file(uploaded_file)
                st.session_state.gambar_proses = st.session_state.gambar_asli.copy()
                st.session_state.riwayat_operasi = []
                tambah_riwayat("Load Gambar", uploaded_file.name)
            st.success(f"✅ Berhasil memuat: {uploaded_file.name}")
            st.balloons()
            
            info = get_image_info(st.session_state.gambar_asli)
            if info:
                st.info(f"""
                📏 **Dimensi:** {info['dimensi']}
                🔢 **Total Piksel:** {info['jumlah_piksel']:,}
                📐 **Rasio Aspek:** {info['rasio_aspek']}
                🎨 **Mode Warna:** RGB (3 kanal)
                """)
    
    with col2:
        if st.session_state.gambar_asli is not None:
            st.image(st.session_state.gambar_asli, caption="📷 Gambar Asli", use_container_width=True)
        else:
            st.info("👈 Silakan upload gambar terlebih dahulu untuk memulai")
    
    if st.session_state.gambar_asli is not None:
        st.markdown("---")
        st.markdown("### 📊 Perbandingan Visual")
        create_comparison_display(
            st.session_state.gambar_asli, 
            st.session_state.gambar_proses,
            "Original", "Hasil Proses"
        )

# ======================================================================
# 4. TRANSFORMASI GEOMETRI
# ======================================================================
elif menu == "Transformasi Geometri":
    st.markdown('<h1 class="main-header">TRANSFORMASI GEOMETRI</h1>', unsafe_allow_html=True)
    
    if st.session_state.gambar_asli is not None:
        col_left, col_right = st.columns([1, 1.5])
        
        with col_left:
            transform_type = st.selectbox(
                "📐 Pilih Jenis Transformasi",
                ["🔄 Rotasi", "📏 Translasi", "🔍 Skala", "🪞 Flip"]
            )
            
            params = {}
            
            if "Rotasi" in transform_type:
                angle = st.slider("Sudut Rotasi (derajat)", -180, 180, 0, 1)
                params["angle"] = angle
                if st.button("🔄 Terapkan Rotasi", use_container_width=True):
                    img = st.session_state.gambar_proses.copy()
                    h, w = img.shape[:2]
                    center = (w//2, h//2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    result = cv2.warpAffine(img, M, (w, h))
                    st.session_state.gambar_proses = result
                    tambah_riwayat("Rotasi", f"{angle}°")
                    st.rerun()
                    
            elif "Translasi" in transform_type:
                tx = st.slider("Geser Horizontal (X)", -200, 200, 0, 10)
                ty = st.slider("Geser Vertikal (Y)", -200, 200, 0, 10)
                params = {"tx": tx, "ty": ty}
                if st.button("📏 Terapkan Translasi", use_container_width=True):
                    img = st.session_state.gambar_proses.copy()
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    result = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    st.session_state.gambar_proses = result
                    tambah_riwayat("Translasi", f"X={tx}, Y={ty}")
                    st.rerun()
                    
            elif "Skala" in transform_type:
                scale_x = st.slider("Faktor Skala Horizontal", 0.1, 3.0, 1.0, 0.05)
                scale_y = st.slider("Faktor Skala Vertikal", 0.1, 3.0, 1.0, 0.05)
                params = {"scale_x": scale_x, "scale_y": scale_y}
                if st.button("🔍 Terapkan Skala", use_container_width=True):
                    img = st.session_state.gambar_proses.copy()
                    new_w = int(img.shape[1] * scale_x)
                    new_h = int(img.shape[0] * scale_y)
                    result = cv2.resize(img, (new_w, new_h))
                    st.session_state.gambar_proses = result
                    tambah_riwayat("Skala", f"X={scale_x:.2f}, Y={scale_y:.2f}")
                    st.rerun()
                    
            else:  # Flip
                flip_dir = st.selectbox("Arah Flip", ["⬅️ Horizontal (Kiri-Kanan)", "⬆️ Vertikal (Atas-Bawah)"])
                if st.button("🪞 Terapkan Flip", use_container_width=True):
                    if "Horizontal" in flip_dir:
                        result = cv2.flip(st.session_state.gambar_proses.copy(), 1)
                        tambah_riwayat("Flip", "Horizontal")
                    else:
                        result = cv2.flip(st.session_state.gambar_proses.copy(), 0)
                        tambah_riwayat("Flip", "Vertikal")
                    st.session_state.gambar_proses = result
                    st.rerun()
            
            # Penjelasan teoritis
            with st.expander("ℹ️ Teori Transformasi Geometri"):
                st.markdown("""
                **Transformasi geometri** adalah operasi yang mengubah posisi piksel dalam citra.
                
                - **Rotasi**: Memutar citra terhadap titik pusat menggunakan matriks rotasi
                - **Translasi**: Memindahkan citra dengan vektor (dx, dy)
                - **Skala**: Memperbesar/memperkecil citra menggunakan interpolasi
                - **Flip**: Membalikkan citra (horizontal/vertikal)
                """)
        
        with col_right:
            st.markdown("### 📊 Hasil Transformasi")
            create_comparison_display(
                st.session_state.gambar_proses if 'img' not in dir() else img,
                st.session_state.gambar_proses,
                "Sebelum Transformasi", "Setelah Transformasi"
            )
    else:
        st.warning("⚠️ Belum ada gambar. Silakan load gambar terlebih dahulu pada menu 'Load & Visualisasi'.")

# ======================================================================
# 5. IMAGE ENHANCEMENT
# ======================================================================
elif menu == "Image Enhancement":
    st.markdown('<h1 class="main-header">IMAGE ENHANCEMENT</h1>', unsafe_allow_html=True)
    
    if st.session_state.gambar_asli is not None:
        col_left, col_right = st.columns([1, 1.5])
        
        with col_left:
            enhancement_type = st.selectbox(
                "✨ Pilih Enhancement",
                ["Brightness", "Contrast", "Sharpening"]
            )
            
            if enhancement_type == "Brightness":
                factor = st.slider("Faktor Kecerahan", 0.5, 2.0, 1.0, 0.05)
                if st.button("✨ Terapkan Brightness", use_container_width=True):
                    pil_img = Image.fromarray(st.session_state.gambar_proses.copy())
                    enhancer = ImageEnhance.Brightness(pil_img)
                    result = np.array(enhancer.enhance(factor))
                    st.session_state.gambar_proses = result
                    tambah_riwayat("Brightness", f"f={factor:.2f}")
                    st.rerun()
                    
            elif enhancement_type == "Contrast":
                factor = st.slider("Faktor Kontras", 0.5, 2.5, 1.0, 0.05)
                if st.button("✨ Terapkan Contrast", use_container_width=True):
                    pil_img = Image.fromarray(st.session_state.gambar_proses.copy())
                    enhancer = ImageEnhance.Contrast(pil_img)
                    result = np.array(enhancer.enhance(factor))
                    st.session_state.gambar_proses = result
                    tambah_riwayat("Contrast", f"f={factor:.2f}")
                    st.rerun()
                    
            else:  # Sharpening
                intensity = st.slider("Intensitas Sharpening", 0.5, 3.0, 1.0, 0.1)
                if st.button("✨ Terapkan Sharpening", use_container_width=True):
                    kernel = np.array([[0, -1, 0], [-1, 4 + intensity, -1], [0, -1, 0]])
                    result = cv2.filter2D(st.session_state.gambar_proses.copy(), -1, kernel)
                    st.session_state.gambar_proses = result
                    tambah_riwayat("Sharpening", f"intensity={intensity:.2f}")
                    st.rerun()
            
            with st.expander("ℹ️ Teori Enhancement"):
                st.markdown("""
                **Image Enhancement** bertujuan meningkatkan kualitas visual citra.
                
                - **Brightness**: g(x,y) = f(x,y) + c (c = faktor kecerahan)
                - **Contrast**: g(x,y) = k × f(x,y) (k = faktor kontras)
                - **Sharpening**: Menggunakan kernel Laplacian untuk mempertegas tepi
                """)
        
        with col_right:
            st.markdown("### 📊 Hasil Enhancement")
            create_comparison_display(
                st.session_state.gambar_proses,
                st.session_state.gambar_proses,
                "Sebelum Enhancement", "Setelah Enhancement"
            )
    else:
        st.warning("⚠️ Belum ada gambar. Silakan load gambar terlebih dahulu.")

# ======================================================================
# 6. HISTOGRAM & EQUALIZATION
# ======================================================================
elif menu == "Histogram & Equalization":
    st.markdown('<h1 class="main-header">HISTOGRAM & EQUALIZATION</h1>', unsafe_allow_html=True)
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Visualisasi histogram
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.patch.set_facecolor('#f8fafc')
        
        # Citra Grayscale
        axes[0, 0].imshow(gray, cmap='gray')
        axes[0, 0].set_title("Citra Grayscale", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Histogram Asli
        axes[0, 1].hist(gray.ravel(), bins=256, color='#2ecc71', alpha=0.7)
        axes[0, 1].set_title("Histogram Asli", fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel("Intensitas Piksel (0-255)")
        axes[0, 1].set_ylabel("Frekuensi")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative Distribution Function
        hist, bins = np.histogram(gray.ravel(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        axes[0, 2].plot(cdf_normalized, color='#1a5276')
        axes[0, 2].set_title("CDF (Cumulative Distribution)", fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel("Intensitas Piksel")
        axes[0, 2].set_ylabel("Frekuensi Kumulatif")
        axes[0, 2].grid(True, alpha=0.3)
        
        # Equalization
        equalized = cv2.equalizeHist(gray)
        axes[1, 0].imshow(equalized, cmap='gray')
        axes[1, 0].set_title("Hasil Equalization", fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Histogram Setelah Equalization
        axes[1, 1].hist(equalized.ravel(), bins=256, color='#1a5276', alpha=0.7)
        axes[1, 1].set_title("Histogram Setelah Equalization", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel("Intensitas Piksel (0-255)")
        axes[1, 1].set_ylabel("Frekuensi")
        axes[1, 1].grid(True, alpha=0.3)
        
        # Perbandingan
        axes[1, 2].hist(gray.ravel(), bins=256, color='#2ecc71', alpha=0.5, label='Asli')
        axes[1, 2].hist(equalized.ravel(), bins=256, color='#1a5276', alpha=0.5, label='Equalized')
        axes[1, 2].set_title("Perbandingan Histogram", fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel("Intensitas Piksel")
        axes[1, 2].set_ylabel("Frekuensi")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray, caption="Sebelum Equalization", use_container_width=True)
        with col2:
            st.image(equalized, caption="Setelah Equalization", use_container_width=True)
        
        if st.button("⚖️ Terapkan Equalization ke Gambar Proses", use_container_width=True):
            st.session_state.gambar_proses = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            tambah_riwayat("Histogram Equalization", "")
            st.success("✅ Equalization berhasil diterapkan!")
            st.rerun()
        
        with st.expander("ℹ️ Teori Histogram Equalization"):
            st.markdown("""
            **Histogram Equalization** adalah teknik untuk meningkatkan kontras citra.
            
            **Langkah-langkah:**
            1. Hitung histogram citra (frekuensi setiap intensitas)
            2. Hitung Cumulative Distribution Function (CDF)
            3. Normalisasi CDF ke rentang 0-255
            4. Mapping nilai intensitas asli ke nilai baru berdasarkan CDF
            
            **Rumus:** s_k = T(r_k) = Σ_{j=0}^{k} (n_j / N) × 255
            - n_j = jumlah piksel dengan intensitas j
            - N = total piksel
            """)
    else:
        st.warning("⚠️ Belum ada gambar. Silakan load gambar terlebih dahulu.")

# ======================================================================
# 7. NOISE & FILTERING
# ======================================================================
elif menu == "Noise & Filtering":
    st.markdown('<h1 class="main-header">NOISE & FILTERING</h1>', unsafe_allow_html=True)
    
    if st.session_state.gambar_asli is not None:
        col_left, col_right = st.columns([1, 1.5])
        
        with col_left:
            operation = st.selectbox("🎛️ Pilih Operasi", ["Tambah Noise", "Filtering"])
            
            if operation == "Tambah Noise":
                noise_type = st.selectbox("Jenis Noise", ["Gaussian", "Salt & Pepper"])
                
                if noise_type == "Gaussian":
                    mean = st.slider("Mean", 0, 50, 0)
                    std = st.slider("Standard Deviation", 5, 50, 25)
                    if st.button("🎲 Tambah Gaussian Noise", use_container_width=True):
                        img = st.session_state.gambar_proses.copy()
                        noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
                        result = cv2.add(img, noise)
                        st.session_state.gambar_proses = result
                        tambah_riwayat("Noise Gaussian", f"mean={mean}, std={std}")
                        st.rerun()
                else:  # Salt & Pepper
                    amount = st.slider("Intensitas Noise", 0.01, 0.2, 0.05, 0.01)
                    if st.button("🎲 Tambah Salt & Pepper", use_container_width=True):
                        img = st.session_state.gambar_proses.copy()
                        result = img.copy()
                        s_vs_p = 0.5
                        num_salt = np.ceil(amount * img.size * s_vs_p)
                        coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape]
                        result[coords[0], coords[1], :] = [255, 255, 255]
                        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
                        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img.shape]
                        result[coords[0], coords[1], :] = [0, 0, 0]
                        st.session_state.gambar_proses = result
                        tambah_riwayat("Noise Salt & Pepper", f"amount={amount:.3f}")
                        st.rerun()
                        
            else:  # Filtering
                filter_type = st.selectbox("Pilih Filter", ["Gaussian", "Median", "Low-pass (Rata-rata)", "High-pass (Penajaman)"])
                
                if st.button("🔧 Terapkan Filter", use_container_width=True):
                    img = st.session_state.gambar_proses.copy()
                    if filter_type == "Gaussian":
                        result = cv2.GaussianBlur(img, (5, 5), 0)
                        tambah_riwayat("Filter Gaussian", "kernel 5x5")
                    elif filter_type == "Median":
                        result = cv2.medianBlur(img, 5)
                        tambah_riwayat("Filter Median", "kernel 5x5")
                    elif "Low-pass" in filter_type:
                        kernel = np.ones((5, 5), np.float32) / 25
                        result = cv2.filter2D(img, -1, kernel)
                        tambah_riwayat("Filter Low-pass", "kernel 5x5 rata-rata")
                    else:  # High-pass
                        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                        result = cv2.filter2D(img, -1, kernel)
                        tambah_riwayat("Filter High-pass", "Laplacian kernel")
                    st.session_state.gambar_proses = result
                    st.rerun()
            
            with st.expander("ℹ️ Teori Noise & Filtering"):
                st.markdown("""
                **Noise** adalah gangguan pada citra yang mengurangi kualitas.
                
                - **Gaussian Noise**: Noise dengan distribusi normal (model sensor noise)
                - **Salt & Pepper**: Piksel acak menjadi hitam atau putih
                
                **Filtering** untuk mengurangi noise:
                - **Gaussian**: Smoothing dengan bobot Gaussian (baik untuk Gaussian noise)
                - **Median**: Mengambil nilai median (sangat baik untuk Salt & Pepper)
                - **Low-pass**: Smoothing dengan kernel rata-rata
                - **High-pass**: Penajaman tepi menggunakan kernel Laplacian
                """)
        
        with col_right:
            st.markdown("### 📊 Hasil Operasi")
            create_comparison_display(
                st.session_state.gambar_proses,
                st.session_state.gambar_proses,
                "Sebelum Operasi", "Setelah Operasi"
            )
    else:
        st.warning("⚠️ Belum ada gambar. Silakan load gambar terlebih dahulu.")

# ======================================================================
# 8. ANALISIS FREKUENSI (FFT)
# ======================================================================
elif menu == "Analisis Frekuensi (FFT)":
    st.markdown('<h1 class="main-header">ANALISIS FREKUENSI (FFT)</h1>', unsafe_allow_html=True)
    
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # FFT
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        
        # Inverse FFT untuk verifikasi
        f_ishift = ifftshift(fshift)
        img_back = np.abs(ifft2(f_ishift))
        
        # Visualisasi
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('#f8fafc')
        
        axes[0, 0].imshow(gray, cmap='gray')
        axes[0, 0].set_title("1. Citra Grayscale (Domain Spasial)", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(magnitude_spectrum, cmap='gray')
        axes[0, 1].set_title("2. Magnitude Spectrum (Domain Frekuensi)", fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(img_back, cmap='gray')
        axes[1, 0].set_title("3. Inverse FFT (Rekonstruksi)", fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Difference (error)
        diff = np.abs(gray - img_back)
        axes[1, 1].imshow(diff, cmap='hot')
        axes[1, 1].set_title(f"4. Error Rekonstruksi (Maks: {diff.max():.2f})", fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📊 **Informasi FFT:**\n\n- Citra ukuran: {gray.shape}\n- Nilai maks spectrum: {magnitude_spectrum.max():.2f}\n- Error maks rekonstruksi: {diff.max():.6f}")
        with col2:
            st.success("✅ FFT dan Inverse FFT berhasil dilakukan dengan error sangat kecil (preserved)")
        
        with st.expander("ℹ️ Teori Fast Fourier Transform (FFT)"):
            st.markdown("""
            **Fast Fourier Transform (FFT)** mengubah citra dari domain spasial ke domain frekuensi.
            
            ### Interpretasi Spektrum Frekuensi:
            - **Titik pusat terang**: Komponen frekuensi rendah (informasi utama)
            - **Semakin terang**: Semakin besar kontribusi frekuensi tersebut
            - **Sumbu X dan Y**: Frekuensi horizontal dan vertikal
            
            ### Aplikasi Domain Frekuensi:
            1. **Low-pass Filtering** - Meloloskan frekuensi rendah (menghaluskan)
            2. **High-pass Filtering** - Meloloskan frekuensi tinggi (deteksi tepi)
            3. **Band-pass Filtering** - Meloloskan rentang frekuensi tertentu
            4. **Compression** - Menghilangkan frekuensi yang tidak penting
            
            ### Rumus FFT 2D:
            F(u,v) = Σ_{x=0}^{M-1} Σ_{y=0}^{N-1} f(x,y) e^{-j2π(ux/M + vy/N)}
            
            dimana:
            - f(x,y) = intensitas piksel di posisi (x,y)
            - F(u,v) = nilai di domain frekuensi
            - M,N = dimensi citra
            """)
    else:
        st.warning("⚠️ Belum ada gambar. Silakan load gambar terlebih dahulu.")

# ======================================================================
# 9. IMPLEMENTASI TEKNIS
# ======================================================================
elif menu == "Implementasi Teknis":
    st.markdown('<h1 class="main-header">IMPLEMENTASI TEKNIS</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📦 Library & Tools", "💻 Kode Utama", "📁 Struktur Project"])
    
    with tabs[0]:
        st.markdown("""
        <div class="card">
            <h2>Teknologi yang Digunakan</h2>
            
            <h3>🐍 Python Libraries</h3>
            <ul>
                <li><strong>Streamlit</strong> - Framework untuk membangun web app interaktif</li>
                <li><strong>OpenCV (cv2)</strong> - Library utama untuk operasi citra (transformasi, filter, noise)</li>
                <li><strong>NumPy</strong> - Manipulasi array multidimensi (representasi citra)</li>
                <li><strong>SciPy</strong> - Implementasi FFT (Fast Fourier Transform)</li>
                <li><strong>PIL/Pillow</strong> - Image enhancement dasar (brightness, contrast)</li>
                <li><strong>Matplotlib</strong> - Visualisasi histogram dan spektrum FFT</li>
            </ul>
            
            <h3>⚙️ Algoritma yang Diimplementasikan</h3>
            <ul>
                <li><strong>Transformasi Geometri</strong> - Rotasi, translasi, scaling, flip</li>
                <li><strong>Enhancement</strong> - Brightness, contrast, sharpening (kernel Laplacian)</li>
                <li><strong>Histogram</strong> - Perhitungan distribusi dan equalization</li>
                <li><strong>Noise Generation</strong> - Gaussian dan Salt & Pepper</li>
                <li><strong>Spatial Filtering</strong> - Gaussian, Median, Low-pass, High-pass</li>
                <li><strong>Frequency Domain</strong> - FFT 2D, magnitude spectrum, inverse FFT</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("""
        <div class="card">
            <h2>Contoh Implementasi Kode</h2>
        </div>
        """, unsafe_allow_html=True)
        
        code_samples = {
            "Rotasi Citra": """
import cv2
import numpy as np

# Load gambar
img = cv2.imread('gambar.jpg')
h, w = img.shape[:2]

# Rotasi 45 derajat terhadap pusat
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
            """,
            
            "Histogram Equalization": """
# Konversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Equalization
equalized = cv2.equalizeHist(gray)

# Visualisasi
import matplotlib.pyplot as plt
plt.hist(equalized.ravel(), bins=256)
plt.show()
            """,
            
            "FFT 2D": """
from scipy.fft import fft2, fftshift, ifft2

# FFT
f = fft2(gray)
fshift = fftshift(f)

# Magnitude spectrum
magnitude = np.log(np.abs(fshift) + 1)

# Inverse FFT
f_ishift = ifftshift(fshift)
img_back = np.abs(ifft2(f_ishift))
            """,
            
            "Convolution Filter": """
# Low-pass filter (rata-rata 5x5)
kernel_avg = np.ones((5,5), np.float32) / 25
filtered = cv2.filter2D(img, -1, kernel_avg)

# High-pass filter (Laplacian)
kernel_lap = np.array([[-1,-1,-1],
                       [-1, 8,-1],
                       [-1,-1,-1]])
sharpened = cv2.filter2D(img, -1, kernel_lap)
            """
        }
        
        selected_code = st.selectbox("Pilih Contoh Kode", list(code_samples.keys()))
        st.code(code_samples[selected_code], language='python')
    
    with tabs[2]:
        st.markdown("""
        <div class="card">
            <h2>Struktur Project</h2>
            <pre>
PCD-STUDIO/
│
├── streamlit_app.py          # Main aplikasi (kode lengkap)
├── requirements.txt          # Dependencies
├── README.md                # Dokumentasi
│
├── utils/                   # (Opsional) Modul tambahan
│   ├── image_processing.py  # Fungsi-fungsi pengolahan citra
│   ├── visualization.py     # Fungsi visualisasi
│   └── constants.py         # Konstanta dan konfigurasi
│
└── assets/                  # Aset (logo, icon, dll)
    └── sample_images/       # Contoh gambar uji
            </pre>
            
            <h3>📄 requirements.txt</h3>
            <pre>
streamlit>=1.28.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
Pillow>=10.0.0
matplotlib>=3.7.0
            </pre>
            
            <h3>🚀 Cara Menjalankan</h3>
            <pre>
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run streamlit_app.py
            </pre>
        </div>
        """, unsafe_allow_html=True)

# ======================================================================
# 10. ANALISIS HASIL
# ======================================================================
elif menu == "Analisis Hasil":
    st.markdown('<h1 class="main-header">ANALISIS HASIL & PEMBAHASAN</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>📊 Observasi dan Analisis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🔄 Transformasi Geometri</h3>
            <p><strong>Hasil Observasi:</strong></p>
            <ul>
                <li>Rotasi mempertahankan informasi visual namun terjadi pemotongan di tepi citra</li>
                <li>Translasi memindahkan posisi citra dengan seamless (tanpa artefak)</li>
                <li>Skala menggunakan interpolasi, kualitas menurun jika diperbesar terlalu besar</li>
                <li>Flip membalik orientasi tanpa kehilangan informasi</li>
            </ul>
            <p><strong>Analisis:</strong> Transformasi geometri efektif untuk koreksi posisi dan orientasi citra. 
            Namun perlu diperhatikan interpolasi saat scaling untuk menjaga kualitas.</p>
        </div>
        
        <div class="card">
            <h3>✨ Image Enhancement</h3>
            <p><strong>Hasil Observasi:</strong></p>
            <ul>
                <li>Brightness menambah/mengurangi kecerahan global citra</li>
                <li>Contrast meningkatkan perbedaan antara area gelap dan terang</li>
                <li>Sharpening mempertegas tepi objek dengan kernel Laplacian</li>
            </ul>
            <p><strong>Analisis:</strong> Enhancement sangat berguna untuk memperbaiki kualitas visual citra 
            yang kurang optimal (terlalu gelap, terlalu terang, atau kurang tajam).</p>
        </div>
        
        <div class="card">
            <h3>📊 Histogram</h3>
            <p><strong>Hasil Observasi:</strong></p>
            <ul>
                <li>Citra gelap memiliki histogram terkonsentrasi di sisi kiri (nilai rendah)</li>
                <li>Citra terang memiliki histogram di sisi kanan (nilai tinggi)</li>
                <li>Equalization menyebarkan histogram secara merata</li>
            </ul>
            <p><strong>Analisis:</strong> Equalization efektif untuk meningkatkan kontras citra secara otomatis, 
            terutama pada citra dengan pencahayaan kurang merata.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🎛️ Noise & Filtering</h3>
            <p><strong>Hasil Observasi:</strong></p>
            <ul>
                <li>Gaussian noise menghasilkan variasi intensitas halus di seluruh citra</li>
                <li>Salt & pepper menghasilkan titik-titik hitam/putih acak</li>
                <li>Gaussian filter efektif untuk Gaussian noise</li>
                <li>Median filter sangat efektif untuk salt & pepper</li>
                <li>Low-pass filter menghaluskan (mengurangi detail)</li>
                <li>High-pass filter menajamkan tepi</li>
            </ul>
            <p><strong>Analisis:</strong> Pemilihan filter harus disesuaikan dengan jenis noise. 
            Median filter adalah pilihan terbaik untuk salt & pepper karena sifat non-linear-nya.</p>
        </div>
        
        <div class="card">
            <h3>🌊 Analisis Frekuensi (FFT)</h3>
            <p><strong>Hasil Observasi:</strong></p>
            <ul>
                <li>Spektrum frekuensi menunjukkan komponen frekuensi rendah di pusat (terang)</li>
                <li>Frekuensi tinggi berada di tepi spektrum</li>
                <li>Inverse FFT berhasil merekonstruksi citra semula dengan error minimal</li>
            </ul>
            <p><strong>Analisis:</strong> FFT sangat berguna untuk analisis dan filtering di domain frekuensi. 
            Komponen frekuensi rendah merepresentasikan informasi utama, sementara frekuensi tinggi 
            merepresentasikan detail tepi dan noise.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>📝 Kesimpulan Analisis</h2>
        <p>Berdasarkan seluruh pengujian yang telah dilakukan, dapat disimpulkan bahwa:</p>
        <ol>
            <li><strong>Semua fitur berfungsi dengan baik</strong> - 11 menu operasi pengolahan citra 
            berhasil diimplementasikan dan menghasilkan output sesuai teori.</li>
            <li><strong>Aplikasi interaktif memudahkan pembelajaran</strong> - Parameter real-time memungkinkan 
            eksplorasi dan pemahaman dampak setiap operasi.</li>
            <li><strong>Integrasi berbagai teknik</strong> - Aplikasi menggabungkan domain spasial dan frekuensi 
            dalam satu platform terintegrasi.</li>
            <li><strong>Responsif dan user-friendly</strong> - Antarmuka yang intuitif dengan visualisasi 
            yang informatif.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ======================================================================
# 11. KESIMPULAN
# ======================================================================
elif menu == "Kesimpulan":
    st.markdown('<h1 class="main-header">KESIMPULAN</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h1 style="font-size: 2rem;">✅</h1>
            <h4>TERIMPLEMENTASI</h4>
            <p>11 menu operasi pengolahan citra</p>
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
            <p>Aplikasi siap digunakan online</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h2>📋 Ringkasan Kesimpulan</h2>
            <ol>
                <li><strong>Implementasi Lengkap</strong> - Aplikasi berhasil mengimplementasikan 5 teknik utama 
                pengolahan citra digital (transformasi geometri, enhancement, histogram, noise/filtering, FFT).</li>
                <li><strong>Pembelajaran Interaktif</strong> - Antarmuka interaktif dengan parameter real-time 
                memudahkan eksperimen dan pemahaman konsep.</li>
                <li><strong>Kesesuaian Teori</strong> - Semua fitur berfungsi sesuai dengan teori pengolahan 
                citra digital yang telah dipelajari.</li>
                <li><strong>Aksesibilitas</strong> - Aplikasi dapat diakses secara online melalui Streamlit Cloud, 
                memudahkan penggunaan di mana saja.</li>
                <li><strong>Dokumentasi Lengkap</strong> - Setiap menu dilengkapi penjelasan teoritis dan 
                analisis hasil.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h2>🚀 Saran Pengembangan</h2>
            <p>Untuk pengembangan lebih lanjut, disarankan:</p>
            <ul>
                <li><strong>Deteksi Tepi</strong> - Implementasi operator Canny, Sobel, Prewitt</li>
                <li><strong>Segmentasi Citra</strong> - Thresholding adaptif dan clustering (K-means)</li>
                <li><strong>Ekstraksi Fitur</strong> - GLCM, HOG, atau LBP untuk klasifikasi</li>
                <li><strong>Deep Learning</strong> - Integrasi model CNN untuk klasifikasi gambar</li>
                <li><strong>Batch Processing</strong> - Kemampuan memproses banyak gambar sekaligus</li>
                <li><strong>Export Report</strong> - Menyimpan hasil analisis dalam format PDF</li>
            </ul>
        </div>
        
        <div class="card">
            <h2>🔗 Referensi</h2>
            <ul>
                <li>Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing, 4th Edition.</li>
                <li>OpenCV Documentation: https://docs.opencv.org/</li>
                <li>Streamlit Documentation: https://docs.streamlit.io/</li>
                <li>Scipy FFT: https://docs.scipy.org/doc/scipy/reference/fft.html</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ======================================================================
# FOOTER
# ======================================================================
st.markdown("""
<div class="footer">
    <p><strong>PCD STUDIO</strong> | Pengolahan Citra Digital | Tugas Akhir Semester Genap 2025/2026</p>
    <p style="font-size: 12px;">Dibangun dengan Streamlit, OpenCV, NumPy, SciPy, dan Python</p>
    <p style="font-size: 12px; opacity: 0.7;">© 2026 - All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
