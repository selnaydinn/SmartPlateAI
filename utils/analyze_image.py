import cv2
import numpy as np
import torch
from utils.constants import FOOD_TR_MAP  # Türkçe isimlerin olduğu dosya

def calculate_nutrients(label_en, mask, cm_kare_birim, h_carpani, food_db):
    """
    Belirli bir maske ve etiket için besin değerlerini hesaplar.
    """
    pixel_count = int(np.sum(mask))
    alan_cm2 = pixel_count * cm_kare_birim
    label_key = label_en.lower()
    
    # Türkçe karşılığı sözlükten al, yoksa İngilizce ismi kullan
    label_tr = FOOD_TR_MAP.get(label_key, label_en).capitalize()
    
    # --- TERMINAL LOG (Gelişim İzleme) ---
    print(f"📍 {label_tr:15} | Piksel: {pixel_count:,} px | Alan: {alan_cm2:.2f} cm²")
    
    if label_key in food_db:
        db = food_db[label_key]
        besin_h = db.get("h", 1.0)
        besin_d = db.get("d", 1.0)
        
        # Gramaj Formülü: Alan * Yükseklik Çarpanı * Besin Özgül Yüksekliği * Özkütle
        gram = alan_cm2 * h_carpani * besin_h * besin_d
        
        return {
            "tr_name": label_tr,
            "gram": int(gram),
            "kcal": (gram * db.get("kcal", 0)) / 100,
            "pro": (gram * db.get("protein", 0)) / 100,
            "carb": (gram * db.get("carb", 0)) / 100,
            "fat": (gram * db.get("fat", 0)) / 100
        }
    
    # Veritabanında yoksa sadece isim ve hata bayrağı dön
    return {"tr_name": label_tr, "error": True}

def analyze_plate(image, model, food_db, device, gercek_cap, h_carpani):
    """
    Görseli analiz eder, nesneleri tespit eder ve rapor oluşturur.
    """
    # Görsel hazırlığı
    original_img = np.array(image)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    h_img, w_img, _ = original_img.shape

    # Piksel -> CM Kare dönüşüm katsayısı
    birim_piksel_cm = gercek_cap / w_img
    cm_kare_birim = birim_piksel_cm ** 2

    print("\n" + "="*50)
    print(f"📸 ANALİZ BAŞLADI | Tabak Çapı: {gercek_cap}cm | Eşik: 0.35")
    print("-" * 50)

    # YOLO Tahmini: conf=0.35 (Etleri yakalamak için), iou=0.45 (Üst üste binmeyi önlemek için)
    results = model.predict(
    source=original_img, 
    conf=0.35, 
    iou=0.45, 
    device=device, 
    verbose=False,
    agnostic_nms=True  # Farklı sınıflar arası çakışmayı önler
    )
    result = results[0]
    
    report = {"cal": 0, "pro": 0, "carb": 0, "fat": 0, "details": []}
    combined_masks_dict = {}
    output_img = original_img.copy()

    if result.masks is not None:
        # 1. MASKELERİ BİRLEŞTİR (Aynı sınıftan olanları tek grupta topla)
        for i, mask_data in enumerate(result.masks.data):
            cls_idx = int(result.boxes.cls[i])
            label_en = model.names[cls_idx].lower()
            
            mask_np = mask_data.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (w_img, h_img))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            if label_en not in combined_masks_dict:
                combined_masks_dict[label_en] = mask_binary
            else:
                combined_masks_dict[label_en] = cv2.bitwise_or(combined_masks_dict[label_en], mask_binary)

        # 2. BESİN HESAPLAMA VE GÖRSELLEŞTİRME
        for label_en, mask in combined_masks_dict.items():
            data = calculate_nutrients(label_en, mask, cm_kare_birim, h_carpani, food_db)
            label_tr = data["tr_name"]
            
            if "error" not in data:
                # Toplam değerleri güncelle
                report["cal"] += data["kcal"]
                report["pro"] += data["pro"]
                report["carb"] += data["carb"]
                report["fat"] += data["fat"]
                
                # --- GÜNCELLEME: label_en eklenerek app.py indeks uyumu sağlandı ---
                report["details"].append({
                    "yiyecek": label_tr,
                    "label_en": label_en, # app.py'daki selectbox indeksi için kritik
                    "gramaj": int(data['gram']),
                    "kalori": int(data['kcal'])
                })
                print(f"⚖️ {label_tr:15} | Tahmini: {int(data['gram'])}g | Enerji: {int(data['kcal'])} kcal")
            else:
                p_count = int(np.sum(mask))
                report["details"].append({
                    "yiyecek": label_tr,
                    "label_en": label_en,
                    "gramaj": f"{p_count:,} px",
                    "kalori": 0
                })
                print(f"⚠️ {label_tr:15} | Veritabanında (food_db) bulunamadı!")

            # MASKELERİ ÇİZ (Sade görünüm)
            color = [int(c) for c in np.random.randint(0, 255, 3)]
            overlay = output_img.copy()
            overlay[mask == 1] = color
            cv2.addWeighted(overlay, 0.4, output_img, 0.6, 0, output_img)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_img, contours, -1, color, 2)

        print("="*50 + "\n")
        final_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        return report, final_image
    
    print("❌ Hiçbir nesne tespit edilemedi.")
    print("="*50 + "\n")
    return None, None