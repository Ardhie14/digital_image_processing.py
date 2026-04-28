# 🖼️ PCD STUDIO - Aplikasi Pengolahan Citra Digital Interaktif

<div align="center">

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

**Aplikasi pembelajaran interaktif untuk Pengolahan Citra Digital**
*Tugas Akhir Semester Genap 2025/2026*

[![Demo](https://img.shields.io/badge/Demo-Live_App-brightgreen?style=for-the-badge)](https://pcd-studio.streamlit.app)
[![Report](https://img.shields.io/badge/Report-PDF-blue?style=for-the-badge)](docs/laporan_pcd_studio.pdf)

</div>

---

## 📋 Daftar Isi

- [Tentang Aplikasi](#-tentang-aplikasi)
- [Fitur Lengkap](#-fitur-lengkap)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Struktur Proyek](#-struktur-proyek)
- [Hasil Pengujian](#-hasil-pengujian)
- [Troubleshooting](#-troubleshooting)
- [Lisensi](#-lisensi)
- [Kontak](#-kontak)

---

## 🎯 Tentang Aplikasi

**PCD Studio** adalah aplikasi web interaktif yang dikembangkan untuk memenuhi tugas akhir mata kuliah Pengolahan Citra Digital. Aplikasi ini menyediakan berbagai fitur pengolahan citra digital yang dilengkapi dengan penjelasan teoritis dan visualisasi hasil secara real-time.

### Tujuan Pengembangan

| Tujuan | Deskripsi |
|--------|-----------|
| 📚 **Pembelajaran** | Memahami konsep pengolahan citra melalui eksperimen langsung |
| 🔧 **Implementasi** | Menerapkan berbagai algoritma dalam satu platform terintegrasi |
| 📊 **Analisis** | Mengamati dan membandingkan efek setiap operasi |
| 🎓 **Akademik** | Memenuhi tugas akhir mata kuliah Pengolahan Citra Digital |

---

## ✨ Fitur Lengkap

### 1. 📂 Load & Visualisasi Citra
- Upload gambar (format JPG, JPEG, PNG)
- Informasi detail citra (dimensi, jumlah piksel, rasio aspek)
- Preview gambar asli dan hasil proses

### 2. 🔄 Transformasi Geometri
| Fitur | Parameter | Deskripsi |
|-------|-----------|-----------|
| Rotasi | -180° s/d 180° | Memutar citra terhadap titik pusat |
| Translasi | X: -200 s/d 200, Y: -200 s/d 200 | Menggeser posisi citra |
| Skala | 0.1x s/d 3.0x | Memperbesar/memperkecil citra |
| Flip | Horizontal / Vertikal | Membalik orientasi citra |

### 3. ✨ Image Enhancement
| Fitur | Rentang | Fungsi |
|-------|---------|--------|
| Brightness | 0.5 - 2.0 | Mengatur kecerahan citra |
| Contrast | 0.5 - 2.5 | Mengatur perbedaan warna |
| Sharpening | 0.5 - 3.0 | Mempertegas tepi objek |

### 4. 📊 Histogram & Equalization
- Visualisasi histogram distribusi intensitas
- Cumulative Distribution Function (CDF)
- Histogram equalization untuk peningkatan kontras
- Perbandingan sebelum dan sesudah

### 5. 🎛️ Noise & Filtering
**Jenis Noise:**
- Gaussian Noise (Mean: 0-50, Std: 5-50)
- Salt & Pepper Noise (Intensitas: 1%-20%)

**Jenis Filter:**
- Gaussian Filter (Kernel 5x5)
- Median Filter (Kernel 5x5)
- Low-pass Filter (Rata-rata 5x5)
- High-pass Filter (Laplacian)

### 6. 🌊 Analisis Frekuensi (FFT)
- Fast Fourier Transform 2D
- Magnitude spectrum visualization
- Inverse FFT untuk rekonstruksi
- Error analysis

---

## 🛠️ Teknologi yang Digunakan

### Core Technologies
```yaml
Bahasa Pemrograman: Python 3.9+
Framework Web: Streamlit 1.28.0+
Library Utama:
  - OpenCV: Operasi pengolahan citra
  - NumPy: Manipulasi array numerik
  - SciPy: Fast Fourier Transform
  - PIL/Pillow: Image enhancement
  - Matplotlib: Visualisasi grafik
