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
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
    }
    
    /* Card style */
    .css-1r6slb0, .css-1v0mbdj, .stMarkdown, .element-container {
        background-color: white;
        border-radius: 20px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .css-1r6slb0:hover, .css-1v0mbdj:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #0f2b3d 0%, #1a5276 50%, #0f2b3d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 32px !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #0f2b3d !important;
        border-left: 5px solid #2ecc71;
        padding-left: 16px;
        margin-top: 24px !important;
        margin-bottom: 16px !important;
        font-weight: 600 !important;
    }
    
    h3 {
        color: #1a5276 !important;
        border-left: 3px solid #2ecc71;
        padding-left: 12px;
        margin-top: 20px !important;
        font-weight: 600 !important;
    }
    
    h4 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* Text */
    p, li, .stMarkdown {
        color: #334155 !important;
        line-height: 1.6 !important;
    }
    
    /* Sidebar */
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
    
    /* Sidebar Radio */
    .stSidebar .stRadio > div > label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        padding: 8px 12px;
        border-radius: 10px;
        transition: all 0.2s;
    }
    
    .stSidebar .stRadio > div > label:hover {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71 !important;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(46, 204, 113, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #27ae60 0%, #219a52 100%);
        box-shadow: 0 6px 20px rgba(46, 204, 113, 0.4);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        transition: all 0.2s;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #2ecc71;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #2ecc71;
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: 14px;
        padding: 16px !important;
        margin: 16px 0 !important;
        border: none !important;
    }
    
    .stAlert.stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b !important;
    }
    
    .stAlert.stWarning p {
        color: #92400e !important;
        font-weight: 600 !important;
    }
    
    .stAlert.stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10b981 !important;
    }
    
    .stAlert.stSuccess p {
        color: #065f46 !important;
    }
    
    .stAlert.stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 5px solid #3b82f6 !important;
    }
    
    .stAlert.stInfo p {
        color: #1e40af !important;
    }
    
    /* Image */
    .stImage {
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        overflow: hidden;
        transition: transform 0.3s;
    }
    
    .stImage:hover {
        transform: scale(1.02);
    }
    
    /* Tabs */
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
        transition: all 0.2s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 12px;
        color: #0f2b3d !important;
        font-weight: 600;
        border: 1px solid #e2e8f0;
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #0f2b3d 0%, #1a3a4f 100%);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card h1, .metric-card h4, .metric-card p {
        color: white !important;
    }
    
    /* Code */
    .stCodeBlock {
        background-color: #1e293b !important;
        border-radius: 14px;
        border: none;
    }
    
    /* Divider */
    hr {
        margin: 24px 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #2ecc71, #1a5276, #2ecc71, transparent);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 24px;
        background: linear-gradient(135deg, #0f2b3d 0%, #1a3a4f 100%);
        border-radius: 20px;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    
    .footer p {
        color: white !important;
        margin: 0;
    }
    
    /* Upload Box */
    .upload-box {
        border: 2px dashed #2ecc71;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        background-color: #f8fafc;
        transition: all 0.3s;
    }
    
    .upload-box:hover {
        background-color: #f0fdf4;
        border-color: #27ae60;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #e2e8f0;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #2ecc71;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #27ae60;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# ANIMASI SIDEBAR
# ========================
st.sidebar.markdown("""
<div style="text-align: center; padding: 24px 0 16px 0;">
    <div style="width: 70px; height: 70px; background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px auto;">
        <span style="font-size: 36px;">🎯</span>
    </div>
    <h2 style="color: white; border-left: none; text-align: center; margin: 0;">PCD STUDIO</h2>
    <p style="color: #a0c4e8; font-size: 14px; margin-top: 8px;">Pengolahan Citra Digital</p>
    <hr style="background: linear-gradient(90deg, transparent, #2ecc71, transparent); margin: 16px 0;">
</div>
""", unsafe_allow_html=True)

# ========================
# MENU NAVIGASI
# ========================
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
    format_func=lambda x: f"{menu_icons[x]}  {x}"
)

# ========================
# SESSION STATE
# ========================
if 'gambar_asli' not in st.session_state:
    st.session_state.gambar_asli = None
if 'gambar_proses' not in st.session_state:
    st.session_state.gambar_proses = None

# ========================
# FUNCTIONS
# ========================
def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def reset_to_original():
    if st.session_state.gambar_asli is not None:
        st.session_state.gambar_proses = st.session_state.gambar_asli.copy()
        st.success("✅ Gambar berhasil di-reset ke original!", icon="🎉")

# ========================
# SIDEBAR STATUS
# ========================
st.sidebar.markdown("---")

if st.session_state.gambar_asli is not None:
    st.sidebar.success("✅ **STATUS: AKTIF**")
    h, w = st.session_state.gambar_asli.shape[:2]
    st.sidebar.info(f"📐 **Dimensi:** {w} x {h} px")
    
    if st.sidebar.button("🔄 Reset ke Gambar Asli", use_container_width=True):
        reset_to_original()
        st.rerun()
else:
    st.sidebar.warning("⚠️ **STATUS: BELUM ADA GAMBAR**")

st.sidebar.markdown("---")

# Status card
st.sidebar.markdown("""
<div style="background: rgba(255,255,255,0.1); border-radius: 12px; padding: 12px; margin-top: 16px;">
    <p style="font-size: 12px; text-align: center; margin: 0;">
        ⚡ Streamlit Cloud<br>
        🎓 Tugas PCD<br>
        📅 2026
    </p>
</div>
""", unsafe_allow_html=True)

# ========================
# 1. PENDAHULUAN
# ========================
if menu == "Pendahuluan":
    st.markdown("# 📖 PENDAHULUAN")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Selamat Datang di PCD Studio
        
        Aplikasi ini dikembangkan untuk memenuhi tugas kuliah **Pengolahan Citra Digital**.
        
        ---
        
        #### 🎯 Tujuan Aplikasi
        
        <span class="badge">01</span> Memahami konsep dasar pengolahan citra digital<br>
        <span class="badge">02</span> Mengimplementasikan berbagai teknik pengolahan citra<br>
        <span class="badge">03</span> Menganalisis efek setiap operasi pada citra<br>
        <span class="badge">04</span> Menyediakan alat interaktif untuk eksperimen
        
        ---
        
        #### ✨ Fitur Lengkap
        
        | Fitur | Deskripsi |
        |-------|-----------|
        | 🔄 Transformasi Geometri | Rotasi, Translasi, Skala, Flip |
        | ✨ Enhancement | Brightness, Contrast, Sharpening |
        | 📊 Histogram | Visualisasi & Equalization |
        | 🎛️ Noise & Filtering | Gaussian, Median, Low-pass, High-pass |
        | 🌊 FFT | Analisis Frekuensi |
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0f2b3d 0%, #1a3a4f 100%); border-radius: 20px; padding: 24px; text-align: center;">
            <div style="font-size: 48px; margin-bottom: 16px;">📋</div>
            <h3 style="color: white; margin-bottom: 16px;">Informasi Tugas</h3>
            <hr style="background: #2ecc71; margin: 16px 0;">
            <p style="color: #e2e8f0;"><strong>Mata Kuliah</strong><br>Pengolahan Citra Digital</p>
            <p style="color: #e2e8f0;"><strong>Semester</strong><br>Genap 2025/2026</p>
            <p style="color: #e2e8f0;"><strong>Platform</strong><br>Streamlit Cloud</p>
            <div style="background: #2ecc71; border-radius: 30px; padding: 8px; margin-top: 16px;">
                <p style="color: #0f2b3d; font-weight: bold; margin: 0;">Ready to Deploy 🚀</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ========================
# 2. TEORI SINGKAT
# ========================
elif menu == "Teori Singkat":
    st.markdown("# 📚 TEORI SINGKAT")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Transformasi Geometri", 
        "✨ Enhancement", 
        "📊 Histogram", 
        "🎛️ Noise & Filtering", 
        "🌊 FFT"
    ])
    
    with tab1:
        st.markdown("""
        ### Transformasi Geometri
        
        Transformasi geometri adalah operasi yang mengubah posisi piksel dalam citra.
        
        | Operasi | Rumus | Keterangan |
        |---------|-------|-------------|
        | **Rotasi** | x' = x cos θ - y sin θ | Memutar gambar dengan sudut θ |
        | **Translasi** | x' = x + dx, y' = y + dy | Memindahkan posisi gambar |
        | **Skala** | x' = sx·x, y' = sy·y | Mengubah ukuran gambar |
        | **Flip** | x' = -x atau y' = -y | Membalik gambar |
        
        > 💡 **Aplikasi:** Koreksi orientasi gambar, pembuatan efek mirror, resize gambar.
        """)
    
    with tab2:
        st.markdown("""
        ### Image Enhancement
        
        Enhancement bertujuan meningkatkan kualitas visual citra.
        
        #### Brightness Enhancement
