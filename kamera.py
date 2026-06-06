from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import easyocr
import serial
import serial.tools.list_ports
import threading
import time
import re

# ================================================================
#   PLAKA TANIMA SISTEMI v2 - DUZELTILMIS
#   YOLOv11 + EasyOCR + Arduino Serial Haberlesme
# ================================================================

# --- AYARLAR ---
MODEL_YOLU    = r"runs\detect\train3\weights\best.pt"
KAMERA_ID     = 0
GUVEN_ESIGI   = 0.4          # 0.5'ten dusurduk, daha fazla tespit
KAYIT_DOSYASI = "plaka_kayitlari.txt"
SERIAL_PORT   = "COM3"
BAUD_RATE     = 9600
OKUMA_SURESI  = 15           # Plakayı kaç saniye boyunca ara

# ================================================================

def arduino_bul():
    portlar = serial.tools.list_ports.comports()
    for p in portlar:
        if any(x in p.description for x in ["Arduino", "CH340", "USB Serial"]):
            print(f"Arduino bulundu: {p.device}")
            return p.device
    return SERIAL_PORT

def plaka_temizle(metin):
    return "".join(k for k in metin.upper() if k.isalnum()).strip()

def kayit_yaz(plaka, hiz, saat, tarih):
    with open(KAYIT_DOSYASI, "a", encoding="utf-8") as f:
        f.write(f"{tarih} {saat} | Plaka: {plaka} | Hiz: {hiz} km/h\n")

# --- Harf/rakam karışıklığı düzeltme ---
# Plaka formatı: 2 rakam + 1-3 harf + 2-4 rakam (örn: 55ABC123)
# Konum bazlı: ilk 2 = rakam, orta = harf, son = rakam
HARF_DUZELT  = {"0": "O", "1": "I", "8": "B", "5": "S", "6": "G", "2": "Z"}
RAKAM_DUZELT = {"O": "0", "I": "1", "B": "8", "S": "5", "G": "6", "Z": "2",
                "T": "7", "L": "1", "Q": "0", "D": "0", "U": "0", "J": "1"}

def plaka_format_duzelt(ham):
    """
    Ham OCR metnini Türk plakası formatına göre düzeltir.
    Format: 2 rakam + 1-3 harf + 2-4 rakam
    """
    metin = re.sub(r'\s+', '', ham).upper()
    if len(metin) < 4:
        return metin

    # Zaten doğru formattaysa dokunma
    if re.match(r'^\d{2}[A-Z]{1,3}\d{2,4}$', metin):
        return metin

    sonuc = list(metin)

    # 1. İlk 2 karakter → rakam olmalı
    for i in range(min(2, len(sonuc))):
        c = sonuc[i]
        if not c.isdigit():
            sonuc[i] = RAKAM_DUZELT.get(c, c)

    # 2. Harf bölgesini bul ve düzelt (index 2'den başla)
    i = 2
    harf_adedi = 0
    harf_bitti = False

    while i < len(sonuc) and not harf_bitti:
        c = sonuc[i]
        if harf_adedi >= 3:
            harf_bitti = True
            break

        if c.isalpha():
            # Harf bölgesindeyiz, rakam benzeri harfleri düzeltme (zaten harf)
            # Ama bazı harfler rakama benzeyenler varsa düzeltme yapma
            harf_adedi += 1
            i += 1
        elif c.isdigit():
            if harf_adedi == 0:
                # Henüz harf bölgesine girmedik, bu rakam aslında harf olabilir
                sonuc[i] = HARF_DUZELT.get(c, c)
                harf_adedi += 1
                i += 1
            else:
                # Harf bölgesi bitti
                harf_bitti = True
        else:
            i += 1

    # 3. Kalan karakterler → rakam olmalı
    while i < len(sonuc):
        c = sonuc[i]
        if not c.isdigit():
            donustur = RAKAM_DUZELT.get(c, c)
            sonuc[i] = donustur
        i += 1

    return "".join(sonuc)

def bolgeyi_buyut(bolge, hedef_genislik=500):
    """Plaka bölgesini büyüt, çözünürlüğü artır."""
    h, w = bolge.shape[:2]
    if w == 0 or h == 0:
        return bolge
    oran = hedef_genislik / w
    yeni_h = max(int(h * oran), 80)
    return cv2.resize(bolge, (hedef_genislik, yeni_h), interpolation=cv2.INTER_LANCZOS4)

def goruntu_on_isle(bolge):
    """
    OCR için çeşitli ön işleme versiyonları üretir.
    Yakın/düz kamera için optimize edilmiştir.
    """
    buyuk = bolgeyi_buyut(bolge, hedef_genislik=500)
    gri   = cv2.cvtColor(buyuk, cv2.COLOR_BGR2GRAY)

    # Gürültü azalt
    gri_blur = cv2.GaussianBlur(gri, (3, 3), 0)

    # CLAHE - kontrast iyileştirme
    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    clahe_img = clahe.apply(gri_blur)

    # Otsu eşikleme
    _, otsu = cv2.threshold(clahe_img, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ters Otsu (koyu zemin, açık yazı)
    _, otsu_inv = cv2.threshold(clahe_img, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptif eşikleme
    adaptif = cv2.adaptiveThreshold(
        clahe_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Keskinleştirme
    k_keskin = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    keskin   = cv2.filter2D(clahe_img, -1, k_keskin)
    keskin   = np.clip(keskin, 0, 255).astype(np.uint8)

    # Morfolojik işlemler
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilate = cv2.dilate(otsu, kernel, iterations=1)
    erozyon = cv2.erode(otsu_inv, kernel, iterations=1)

    return [
        ("renkli",    buyuk),       # Renkli - bazen en iyi sonuç
        ("clahe",     clahe_img),
        ("otsu",      otsu),
        ("otsu_inv",  otsu_inv),
        ("adaptif",   adaptif),
        ("keskin",    keskin),
        ("dilate",    dilate),
        ("erozyon",   erozyon),
    ]

def ocr_uygula(goruntu):
    """
    Tek görüntüye EasyOCR uygular.
    Güven eşiğini düşük tutarak daha fazla aday alırız.
    """
    try:
        # NOT: allowlist KALDIRILDI - kısıtlama OCR'ı bozuyordu
        # Bunun yerine sonuçları filtreliyoruz
        sonuclar = reader.readtext(
            goruntu,
            detail=1,
            paragraph=False,
            width_ths=0.8,
            link_threshold=0.4,
            low_text=0.3,       # Düşük kontrast metinleri de al
            text_threshold=0.5, # Daha hassas tespit
        )

        parcalar = []
        for (_, metin, olasilik) in sonuclar:
            if olasilik > 0.15:  # Eşiği düşürdük (0.2'den 0.15'e)
                # Sadece harf ve rakam bırak
                temiz = re.sub(r'[^A-Za-z0-9]', '', metin).upper()
                if temiz:
                    parcalar.append(temiz)

        birlesik = "".join(parcalar)
        return birlesik
    except Exception as e:
        print(f"  [OCR HATA] {e}")
        return ""

def plaka_skorla(metin):
    """
    Bir plaka metninin ne kadar Türk plakasına benzediğini puanlar.
    Yüksek puan = daha olası gerçek plaka.
    """
    if not metin or len(metin) < 4:
        return 0

    skor = 0

    # Uzunluk puanı (ideal: 7-9 karakter)
    if 7 <= len(metin) <= 9:
        skor += 30
    elif 5 <= len(metin) <= 11:
        skor += 15

    # Format eşleşmesi: 2 rakam + 1-3 harf + 2-4 rakam
    if re.match(r'^\d{2}[A-Z]{1,3}\d{2,4}$', metin):
        skor += 50

    # Kısmi eşleşme kontrolleri
    if re.match(r'^\d{2}', metin):
        skor += 10  # İlk 2 rakam
    if re.search(r'\d{2,4}$', metin):
        skor += 10  # Son rakamlar

    return skor

def plaka_oku_frame(frame):
    """
    Tek frame'de YOLO ile plaka tespit eder, OCR uygular.
    Geliştirilmiş oy + skor sistemi ile en iyi plakayı döndürür.
    """
    results = model(frame, conf=GUVEN_ESIGI, verbose=False)
    en_iyi_plaka = ""
    en_iyi_skor  = -1

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            guven = float(box.conf[0])

            # Kutuyu genişlet (%10 yerine sabit piksel)
            pad_x = max(10, int((x2 - x1) * 0.08))
            pad_y = max(5,  int((y2 - y1) * 0.10))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(frame.shape[1], x2 + pad_x)
            y2 = min(frame.shape[0], y2 + pad_y)

            bolge = frame[y1:y2, x1:x2]
            if bolge.size == 0:
                continue

            # 8 versiyona OCR uygula
            oy = {}
            for isim, goruntu in goruntu_on_isle(bolge):
                metin = ocr_uygula(goruntu)
                metin = plaka_temizle(metin)
                if metin and len(metin) >= 4:
                    oy[metin] = oy.get(metin, 0) + 1

            if not oy:
                continue

            # En iyi adayı seç: oy × format_skoru × YOLO_güven
            en_iyi_aday = ""
            en_iyi_aday_skor = -1
            for aday, oy_sayisi in oy.items():
                format_skor = plaka_skorla(aday)
                toplam = oy_sayisi * 2 + format_skor * 0.1 + guven * 5 + len(aday) * 0.3
                if toplam > en_iyi_aday_skor:
                    en_iyi_aday_skor = toplam
                    en_iyi_aday = aday

            if en_iyi_aday_skor > en_iyi_skor:
                en_iyi_skor  = en_iyi_aday_skor
                en_iyi_plaka = plaka_format_duzelt(en_iyi_aday)

    return en_iyi_plaka

# ================================================================
print("Model yukleniyor...")
model = YOLO(MODEL_YOLU)

print("OCR motoru baslatiliyor...")
# Türkçe VE İngilizce - ikisi birlikte daha iyi sonuç
reader = easyocr.Reader(['tr', 'en'], gpu=False)
# GPU varsa: reader = easyocr.Reader(['tr', 'en'], gpu=True)

print("Kamera aciliyor...")
cap = cv2.VideoCapture(KAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)     # Otofokus aç

print("Arduino baglantisi kuruluyor...")
port = arduino_bul()
try:
    arduino = serial.Serial(port, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Arduino baglandi: {port}")
    arduino_bagli = True
except Exception as e:
    print(f"[UYARI] Arduino baglanamadi: {e}")
    print("Kamera modu ile devam ediliyor...")
    arduino = None
    arduino_bagli = False

# ================================================================
lock          = threading.Lock()
son_plaka     = ""
son_hiz       = ""
durum_mesaji  = "Arac Bekleniyor..."
plaka_suresi  = 0.0
isleniyor     = False

# ================================================================
def arduino_dinle():
    """Arka planda Arduino'dan TETIK bekle, plaka oku, geri gönder."""
    global son_plaka, son_hiz, durum_mesaji, plaka_suresi, isleniyor

    while True:
        if not arduino_bagli or arduino is None:
            time.sleep(0.5)
            continue
        try:
            if arduino.in_waiting > 0:
                satir = arduino.readline().decode('utf-8', errors='ignore').strip()

                if not satir.startswith("TETIK:"):
                    continue

                try:
                    hiz_kmh = float(satir.replace("TETIK:", "").strip())
                except:
                    hiz_kmh = 0.0

                print(f"\n[TETIK] Hiz: {hiz_kmh:.1f} km/h - Plaka aranıyor ({OKUMA_SURESI} sn)...")

                with lock:
                    isleniyor    = True
                    durum_mesaji = f"HIZ: {hiz_kmh:.1f} km/h | Plaka okunuyor..."

                simdi     = datetime.datetime.now()
                saat_str  = simdi.strftime("%H:%M:%S")
                tarih_str = simdi.strftime("%d.%m.%Y")

                # --- PLAKA ARA: Oy tablosu ile en güvenilir sonucu bul ---
                oy_tablosu   = {}  # { plaka: oy_sayisi }
                skor_tablosu = {}  # { plaka: en_yuksek_skor }
                bitis        = time.time() + OKUMA_SURESI
                frame_sayisi = 0

                while time.time() < bitis:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    plaka = plaka_oku_frame(frame)
                    if plaka and len(plaka) >= 4:
                        oy_tablosu[plaka]   = oy_tablosu.get(plaka, 0) + 1
                        fmt_skor = plaka_skorla(plaka)
                        skor_tablosu[plaka] = max(skor_tablosu.get(plaka, 0), fmt_skor)
                        print(f"  [{frame_sayisi}] Aday: {plaka} (oy:{oy_tablosu[plaka]}, skor:{fmt_skor})")
                        # Anlık okunan en iyi plakayı ekranda göster
                        en_iyi_ani = max(
                            oy_tablosu,
                            key=lambda p: oy_tablosu[p] * 2 + skor_tablosu.get(p, 0)
                        )
                        with lock:
                            son_plaka    = en_iyi_ani
                            plaka_suresi = time.time() + 9999

                    frame_sayisi += 1
                    time.sleep(0.03)  # ~30fps

                # --- En iyi plakayı seç ---
                en_iyi = ""
                if oy_tablosu:
                    # Skor: (oy × 2) + format_skoru
                    en_iyi = max(
                        oy_tablosu,
                        key=lambda p: oy_tablosu[p] * 2 + skor_tablosu.get(p, 0)
                    )
                    print(f"\n[SONUÇ] {frame_sayisi} frame tarandı")
                    print(f"[SONUÇ] Tüm adaylar: {dict(sorted(oy_tablosu.items(), key=lambda x: -x[1]))}")
                    print(f"[SONUÇ] Seçilen: {en_iyi}")
                else:
                    print(f"\n[SONUÇ] {frame_sayisi} frame tarandı, plaka bulunamadı!")

                bulunan = en_iyi if en_iyi else "OKUNAMADI"

                # Arduino'ya gönder
                gonder = f"{bulunan}|{hiz_kmh:.1f}|{saat_str}|{tarih_str}\n"
                try:
                    arduino.write(gonder.encode('utf-8'))
                    print(f"[ARDUINO] Gonderildi: {gonder.strip()}")
                except Exception as e:
                    print(f"[SERIAL HATA] {e}")

                kayit_yaz(bulunan, f"{hiz_kmh:.1f}", saat_str, tarih_str)

                with lock:
                    son_plaka    = bulunan
                    son_hiz      = f"{hiz_kmh:.1f}"
                    durum_mesaji = f"Plaka: {bulunan}"
                    plaka_suresi = time.time()  # Buradan itibaren 5 sn göster
                    isleniyor    = False

        except Exception as e:
            print(f"[HATA] {e}")
            time.sleep(0.5)

# Thread başlat
t = threading.Thread(target=arduino_dinle, daemon=True)
t.start()

print("\nSistem hazir!")
if arduino_bagli:
    print("Arduino tetik bekleniyor... (Cikis: q)")
else:
    print("Arduino YOK - Space ile manuel test. (Cikis: q)")

# ================================================================
# ANA EKRAN DONGUSU
# ================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera okunamadi!")
        break

    simdi     = datetime.datetime.now()
    saat_str  = simdi.strftime("%H:%M:%S")
    tarih_str = simdi.strftime("%d.%m.%Y")

    # Canlı YOLO tespiti
    yolo_sonuc   = model(frame, conf=GUVEN_ESIGI, verbose=False)
    plaka_tespit = False

    for result in yolo_sonuc:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            guven = float(box.conf[0])
            plaka_tespit = True

            with lock:
                _isleniyor_ani = isleniyor

            kutu_renk = (0, 220, 255) if _isleniyor_ani else (0, 255, 0)
            uzunluk   = 20
            kalinlik  = 3
            cv2.line(frame, (x1, y1), (x1+uzunluk, y1), kutu_renk, kalinlik)
            cv2.line(frame, (x1, y1), (x1, y1+uzunluk), kutu_renk, kalinlik)
            cv2.line(frame, (x2, y1), (x2-uzunluk, y1), kutu_renk, kalinlik)
            cv2.line(frame, (x2, y1), (x2, y1+uzunluk), kutu_renk, kalinlik)
            cv2.line(frame, (x1, y2), (x1+uzunluk, y2), kutu_renk, kalinlik)
            cv2.line(frame, (x1, y2), (x1, y2-uzunluk), kutu_renk, kalinlik)
            cv2.line(frame, (x2, y2), (x2-uzunluk, y2), kutu_renk, kalinlik)
            cv2.line(frame, (x2, y2), (x2, y2-uzunluk), kutu_renk, kalinlik)

            etiket = f"Plaka %{guven*100:.0f}"
            (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+8, y1), kutu_renk, -1)
            cv2.putText(frame, etiket, (x1+4, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Üst bant
    cv2.rectangle(frame, (0, 0), (280, 40), (20, 20, 20), -1)
    cv2.putText(frame, tarih_str, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    with lock:
        _son_plaka  = son_plaka
        _son_hiz    = son_hiz
        _plaka_sure = plaka_suresi
        _isleniyor  = isleniyor
        _durum      = durum_mesaji

    if _son_plaka and (time.time() - _plaka_sure < 5):
        bilgi = f"Plaka: {_son_plaka}"
        (bw, _), _ = cv2.getTextSize(bilgi, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (0, 45), (bw + 20, 90), (0, 120, 0), -1)
        cv2.putText(frame, bilgi, (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if _isleniyor:
        durum_renk = (0, 200, 255)
    elif plaka_tespit:
        durum_renk = (0, 255, 0)
    elif _son_plaka and (time.time() - _plaka_sure < 5):
        durum_renk = (0, 220, 0)
    else:
        durum_renk = (0, 165, 255)

    cv2.rectangle(frame, (0, frame.shape[0]-40),
                  (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
    cv2.putText(frame, _durum, (10, frame.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, durum_renk, 2)

    if not arduino_bagli:
        cv2.putText(frame, "ARDUINO YOK | Space: manuel test",
                    (10, frame.shape[0]-55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 2)

    cv2.imshow("Plaka Tanima Sistemi", frame)

    tus = cv2.waitKey(1) & 0xFF
    if tus == ord('q'):
        break
    elif tus == ord(' ') and not arduino_bagli:
        print("\n[MANUEL TEST] Plaka aranıyor...")
        with lock:
            isleniyor    = True
            durum_mesaji = "Manuel: Plaka okunuyor..."

        oy_tablosu   = {}
        skor_tablosu = {}
        bitis        = time.time() + OKUMA_SURESI

        while time.time() < bitis:
            ret2, f2 = cap.read()
            if ret2:
                p = plaka_oku_frame(f2)
                if p and len(p) >= 4:
                    oy_tablosu[p]   = oy_tablosu.get(p, 0) + 1
                    skor_tablosu[p] = max(skor_tablosu.get(p, 0), plaka_skorla(p))
                    print(f"  Aday: {p}")
            time.sleep(0.03)

        if oy_tablosu:
            bulunan = max(oy_tablosu,
                          key=lambda p: oy_tablosu[p] * 2 + skor_tablosu.get(p, 0))
        else:
            bulunan = "OKUNAMADI"

        print(f"[MANUEL] Sonuc: {bulunan}")
        kayit_yaz(bulunan, "?", saat_str, tarih_str)
        with lock:
            son_plaka    = bulunan
            son_hiz      = "?"
            durum_mesaji = f"Plaka: {bulunan}"
            plaka_suresi = time.time()
            isleniyor    = False

cap.release()
cv2.destroyAllWindows()
print("Sistem kapatildi.")
