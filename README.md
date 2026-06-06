# 🚗 Görüntü İşleme Tabanlı Plaka Okuma ve LCD Göstergeli Hız Tanıma Sistemi

Kamera ile araç plakasını tanıyan, ultrasonik sensörle hızını ölçen ve sonuçları LCD ekranda gösteren Arduino + Python tabanlı akıllı trafik kontrol sistemi.

---

## 📌 Proje Hakkında

Bu proje iki ana bileşenden oluşmaktadır:

- **Python + YOLOv11:** Kamera görüntüsünden araç plakasını gerçek zamanlı olarak tespit eder ve okur.
- **Arduino + Ultrasonik Sensör:** Araçların hızını ölçer ve plaka bilgisiyle birlikte LCD ekrana yansıtır.

---

## 🛠️ Kullanılan Teknolojiler

| Bileşen | Teknoloji |
|---|---|
| Plaka Tespiti | YOLOv11, Python, Ultralytics |
| Görüntü İşleme | OpenCV |
| Mikrodenetleyici | Arduino |
| Hız Ölçümü | HC-SR04 Ultrasonik Sensör |
| Ekran | 16x2 LCD (I2C) |

---

## 📁 Proje Yapısı

```
plaka-hiz-tespit-sistemi/
├── egitim.py        # YOLOv11 model eğitimi
├── kamera.py        # Gerçek zamanlı plaka tespiti
├── arduino/
│   └── hiz_olcum.ino  # Arduino hız ölçüm kodu
├── data.yaml        # Veri seti yapılandırması
└── README.md
```

---

## 🚀 Kurulum

### Gereksinimler

```bash
pip install ultralytics opencv-python
```

### Modeli Eğit

```bash
python egitim.py
```

### Kamerayı Başlat

```bash
python kamera.py
```

---

## ⚙️ Nasıl Çalışır?

1. Ultrasonik sensör aracın hızını ölçer ve Arduino'ya iletir.
2. Kamera görüntüsü Python tarafından alınır, YOLOv11 modeli plakayı tespit eder.
3. Plaka bilgisi ve hız verisi Arduino üzerinden LCD ekranda gösterilir.

---

## 📊 Model Bilgisi

- **Model:** YOLOv11n
- **Veri Seti:** 2958 Türk plaka görseli (Roboflow)
- **Eğitim:** 50 epoch, 640px

---

## 👩‍💻 Geliştirici

**Şevval** — [github.com/sevvalhub](https://github.com/sevvalhub)
