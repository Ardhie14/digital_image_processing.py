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
    page_icon="🖼️",
    layout="wide"
)

# ========================
# SIDEBAR NAVIGASI
# ========================
st.sidebar.title("📌 Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu:",
    ["Pendahuluan", "Teori Singkat", "Load & Tampilkan Gambar",
     "Transformasi Geometri", "Enhancement", "Histogram",
     "Noise & Filtering", "FFT (Analisis Frekuensi)",
     "Implementasi", "Hasil & Pembahasan", "Kesimpulan"]
)

# Session state untuk menyimpan gambar
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

# ========================
# 1. PENDAHULUAN
# ========================
if menu == "Pendahuluan":
    st.title("📖 Pendahuluan")
    st.write("Aplikasi Pengolahan Citra Digital")
    st.write("Aplikasi ini dikembangkan untuk memenuhi tugas kuliah Pengolahan Citra Digital.")
    st.write("Aplikasi dibangun menggunakan Streamlit dan Python dengan berbagai fitur:")
    st.write("- Transformasi Geometri (Rotasi, Translasi, Skala, Flip)")
    st.write("- Enhancement (Brightness, Contrast, Sharpening)")
    st.write("- Histogram & Equalization")
    st.write("- Noise & Filtering (Gaussian, Median, Low-pass, High-pass)")
    st.write("- FFT (Analisis Frekuensi)")
    st.write("Tujuan: Memahami konsep dasar pengolahan citra secara interaktif.")

# ========================
# 2. TEORI SINGKAT
# ========================
elif menu == "Teori Singkat":
    st.title("📚 Teori Singkat")
    with st.expander("Transformasi Geometri"):
        st.write("- Rotasi: Memutar gambar dengan sudut tertentu.")
        st.write("- Translasi: Memindahkan posisi gambar.")
        st.write("- Skala: Memperbesar/memperkecil gambar.")
        st.write("- Flip: Membalik gambar (horizontal/vertikal).")
    with st.expander("Enhancement"):
        st.write("- Brightness: Mengatur kecerahan.")
        st.write("- Contrast: Mengatur perbedaan warna.")
        st.write("- Sharpening: Mempertegas tepi objek.")
    with st.expander("Histogram"):
        st.write("Histogram menggambarkan distribusi intensitas piksel.")
        st.write("Equalization menyebarkan intensitas secara merata.")
    with st.expander("Noise & Filtering"):
        st.write("- Gaussian Noise: Noise terdistribusi normal.")
        st.write("- Salt & Pepper Noise: Titik hitam/putih acak.")
        st.write("- Filter Median: Menghilangkan noise impulsif.")
        st.write("- Filter Gaussian: Menghaluskan gambar.")
    with st.expander("FFT (Fast Fourier Transform)"):
        st.write("FFT mengubah citra dari domain spasial ke domain frekuensi.")
        st.write("Memungkinkan analisis komponen frekuensi rendah/tinggi.")

# ========================
# 3. LOAD GAMBAR
# ========================
elif menu == "Load & Tampilkan Gambar":
    st.title("📂 Load & Tampilkan Gambar")
    uploaded_file = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.session_state.gambar_asli = load_image(uploaded_file)
        st.session_state.gambar_proses = st.session_state.gambar_asli.copy()
        st.success("Gambar berhasil di-load!")
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.gambar_asli, caption="Gambar Asli", use_container_width=True)
        with col2:
            st.image(st.session_state.gambar_proses, caption="Gambar Proses", use_container_width=True)
    else:
        st.info("Silakan upload gambar terlebih dahulu.")

# ========================
# 4. TRANSFORMASI GEOMETRI
# ========================
elif menu == "Transformasi Geometri":
    st.title("🔄 Transformasi Geometri")
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        transform = st.selectbox("Pilih Transformasi:", ["Rotasi", "Translasi", "Skala", "Flip"])
        
        if transform == "Rotasi":
            angle = st.slider("Sudut Rotasi (derajat):", -180, 180, 0)
            h, w = img.shape[:2]
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(img, M, (w, h))
            
        elif transform == "Translasi":
            tx = st.slider("Translasi X:", -100, 100, 0)
            ty = st.slider("Translasi Y:", -100, 100, 0)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            result = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            
        elif transform == "Skala":
            scale = st.slider("Faktor Skala:", 0.1, 3.0, 1.0)
            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            result = cv2.resize(img, (new_w, new_h))
            
        else:
            flip_code = st.selectbox("Arah Flip:", ["Horizontal", "Vertikal"])
            if flip_code == "Horizontal":
                result = cv2.flip(img, 1)
            else:
                result = cv2.flip(img, 0)
        
        st.session_state.gambar_proses = result
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Sebelum", use_container_width=True)
        with col2:
            st.image(result, caption="Sesudah", use_container_width=True)
    else:
        st.warning("Load gambar terlebih dahulu di menu Load & Tampilkan Gambar.")

# ========================
# 5. ENHANCEMENT
# ========================
elif menu == "Enhancement":
    st.title("✨ Enhancement")
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        pil_img = Image.fromarray(img)
        
        enhancement_type = st.selectbox("Pilih Enhancement:", ["Brightness", "Contrast", "Sharpening"])
        
        if enhancement_type == "Brightness":
            factor = st.slider("Faktor Kecerahan:", 0.5, 2.0, 1.0)
            enhancer = ImageEnhance.Brightness(pil_img)
            result = np.array(enhancer.enhance(factor))
            
        elif enhancement_type == "Contrast":
            factor = st.slider("Faktor Kontras:", 0.5, 2.0, 1.0)
            enhancer = ImageEnhance.Contrast(pil_img)
            result = np.array(enhancer.enhance(factor))
            
        else:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            result = cv2.filter2D(img, -1, kernel)
        
        st.session_state.gambar_proses = result
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Sebelum", use_container_width=True)
        with col2:
            st.image(result, caption="Sesudah", use_container_width=True)
    else:
        st.warning("Load gambar terlebih dahulu.")

# ========================
# 6. HISTOGRAM
# ========================
elif menu == "Histogram":
    st.title("📊 Histogram & Equalization")
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].imshow(gray, cmap='gray')
        axes[0, 0].set_title("Citra Grayscale")
        axes[0, 0].axis('off')
        
        axes[0, 1].hist(gray.ravel(), bins=256, color='black')
        axes[0, 1].set_title("Histogram Asli")
        
        equalized = cv2.equalizeHist(gray)
        axes[1, 0].imshow(equalized, cmap='gray')
        axes[1, 0].set_title("Setelah Equalization")
        axes[1, 0].axis('off')
        
        axes[1, 1].hist(equalized.ravel(), bins=256, color='black')
        axes[1, 1].set_title("Histogram Equalized")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        if st.button("Terapkan Equalization ke Gambar Proses"):
            st.session_state.gambar_proses = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            st.success("Equalization diterapkan!")
    else:
        st.warning("Load gambar terlebih dahulu.")

# ========================
# 7. NOISE & FILTERING
# ========================
elif menu == "Noise & Filtering":
    st.title("🎛️ Noise & Filtering")
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        
        operation = st.selectbox("Pilih Operasi:", ["Tambah Noise", "Terapkan Filter"])
        
        if operation == "Tambah Noise":
            noise_type = st.selectbox("Jenis Noise:", ["Gaussian", "Salt & Pepper"])
            if st.button("Tambahkan Noise"):
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
                
        else:
            filter_type = st.selectbox("Pilih Filter:", ["Gaussian", "Median", "Low-pass", "High-pass"])
            if st.button("Terapkan Filter"):
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
        
        if 'result' in locals():
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Sebelum", use_container_width=True)
            with col2:
                st.image(result, caption="Sesudah", use_container_width=True)
    else:
        st.warning("Load gambar terlebih dahulu.")

# ========================
# 8. FFT (ANALISIS FREKUENSI)
# ========================
elif menu == "FFT (Analisis Frekuensi)":
    st.title("🌊 FFT - Analisis Frekuensi")
    if st.session_state.gambar_asli is not None:
        img = st.session_state.gambar_proses.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(gray, cmap='gray')
        axes[0].set_title("Citra Grayscale")
        axes[0].axis('off')
        
        axes[1].imshow(magnitude_spectrum, cmap='gray')
        axes[1].set_title("Magnitude Spectrum (FFT)")
        axes[1].axis('off')
        
        f_ishift = ifftshift(fshift)
        img_back = ifft2(f_ishift)
        img_back = np.abs(img_back)
        axes[2].imshow(img_back, cmap='gray')
        axes[2].set_title("Hasil Inverse FFT")
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("Penjelasan: Magnitude Spectrum menunjukkan distribusi energi frekuensi. Titik terang di tengah = komponen frekuensi rendah. Semakin jauh dari pusat = frekuensi tinggi.")
    else:
        st.warning("Load gambar terlebih dahulu.")

# ========================
# 9. IMPLEMENTASI
# ========================
elif menu == "Implementasi":
    st.title("💻 Implementasi")
    st.write("Tools & Libraries:")
    st.write("- Streamlit: Framework web interaktif")
    st.write("- OpenCV: Operasi citra (transformasi, filter, noise)")
    st.write("- PIL/Pillow: Enhancement dasar")
    st.write("- NumPy: Manipulasi array & FFT")
    st.write("- Matplotlib: Visualisasi histogram & spektrum FFT")
    st.write("- SciPy: FFT lanjutan")
    st.write("")
    st.write("Struktur Kode Utama:")
    st.code("""
# Load gambar
img = cv2.imread(uploaded_file)

# Rotasi
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))

# Histogram equalization
equalized = cv2.equalizeHist(gray)

# FFT
f = fft2(gray)
fshift = fftshift(f)
magnitude = np.log(np.abs(fshift) + 1)
""", language="python")

# ========================
# 10. HASIL & PEMBAHASAN
# ========================
elif menu == "Hasil & Pembahasan":
    st.title("📝 Hasil & Pembahasan")
    st.markdown("### Observasi & Analisis:")
    st.markdown("")
    st.markdown("1. **Transformasi Geometri**")
    st.markdown("   - Rotasi dan skala dapat mengubah informasi spasial.")
    st.markdown("   - Translasi memotong gambar jika melebihi batas.")
    st.markdown("")
    st.markdown("2. **Enhancement**")
    st.markdown("   - Brightness > 1.0 membuat gambar lebih terang.")
    st.markdown("   - Sharpening meningkatkan ketajaman tepi.")
    st.markdown("")
    st.markdown("3. **Histogram**")
    st.markdown("   - Citra gelap -> histogram terkonsentrasi di kiri.")
    st.markdown("   - Equalization menyebarkan intensitas -> kontras lebih baik.")
    st.markdown("")
    st.markdown("4. **Noise & Filtering**")
    st.markdown("   - Gaussian noise -> filter Gaussian efektif.")
    st.markdown("   - Salt & pepper noise -> filter median lebih baik.")
    st.markdown("")
    st.markdown("5. **FFT**")
    st.markdown("   - Frekuensi rendah (pusat spektrum) = informasi utama.")
    st.markdown("   - High-pass filter = deteksi tepi.")
    st.markdown("")
    st.markdown("### Contoh Hasil Percobaan:")
    st.markdown("*(Screenshot hasil pengujian dapat ditambahkan di sini)*")

# ========================
# 11. KESIMPULAN
# ========================
elif menu == "Kesimpulan":
    st.title("📌 Kesimpulan")
    st.markdown("### Kesimpulan:")
    st.markdown("")
    st.markdown("1. Aplikasi berhasil mengimplementasikan berbagai teknik pengolahan citra digital secara interaktif.")
    st.markdown("2. Transformasi geometri memudahkan manipulasi posisi dan ukuran objek dalam citra.")
    st.markdown("3. Enhancement meningkatkan kualitas visual citra sesuai kebutuhan.")
    st.markdown("4. Histogram dan equalization membantu perbaikan kontras secara otomatis.")
    st.markdown("5. Noise & filtering menunjukkan pentingnya memilih filter yang tepat sesuai jenis noise.")
    st.markdown("6. FFT memberikan wawasan tentang komposisi frekuensi dalam citra.")
    st.markdown("")
    st.markdown("### Pengembangan Lebih Lanjut:")
    st.markdown("- Implementasi deteksi tepi (Canny, Sobel)")
    st.markdown("- Segmentasi citra (thresholding, watershed)")
    st.markdown("- Ekstraksi fitur (HOG, LBP)")
    st.markdown("- Model deep learning untuk klasifikasi citra")
    st.markdown("")
    st.markdown("---")
    st.markdown("**Aplikasi ini dikembangkan untuk memenuhi tugas kuliah Pengolahan Citra Digital.**")

# ========================
# FOOTER DI SIDEBAR
# ========================
st.sidebar.markdown("---")
if st.session_state.gambar_asli is not None:
    st.sidebar.success("Status: Gambar tersedia")
else:
    st.sidebar.info("Status: Belum ada gambar")
