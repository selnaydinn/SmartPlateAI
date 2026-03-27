import torch
import json
from ultralytics import YOLO
import streamlit as st

@st.cache_resource
def initialize_system():
    # 1. GPU Kontrolü
    gpu_status = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_status else "CPU (GPU Bulunamadı)"
    
    # 2. Model Yükleme
    try:
        model = YOLO('best.pt')
    except Exception as e:
        print(f"❌ HATA: Model yüklenemedi: {e}")
        return None, None, False, "Hata", "Bilinmiyor"
    
    # --- MODEL TÜRÜ TESPİTİ (Yeni Eklenen Kısım) ---
    params = sum(p.numel() for p in model.model.parameters()) / 1e6 # Milyon cinsinden
    
    if params < 5:
        model_type = "Nano (n)"
    elif params < 15:
        model_type = "Small (s)"
    elif params < 30:
        model_type = "Medium (m)"
    elif params < 50:
        model_type = "Large (l)"
    else:
        model_type = "X-Large (x)"
    # ----------------------------------------------
    
    # 3. Model Başarı Skoru (mAP50)
    map_score = "Bilinmiyor"
    if hasattr(model, 'ckpt') and model.ckpt is not None:
        if 'train_metrics' in model.ckpt:
            # Hem Box hem Mask mAP değerlerini kontrol et
            metrics = model.ckpt['train_metrics']
            val = metrics.get('metrics/mAP50(B)') or metrics.get('metrics/mAP50(M)') or 0
            map_score = f"%{val * 100:.2f}"
    
    # --- TERMINAL LOGLARI (Güncellendi) ---
    print("\n" + "="*45)
    print("🚀 SMARTPLATE AI SİSTEM BAŞLATILDI")
    print("="*45)
    print(f"🖥️  DONANIM       : {device_name}")
    print(f"⚙️  GPU DURUMU    : {'✅ AKTİF' if gpu_status else '❌ PASİF'}")
    print(f"📊 MODEL TÜRÜ    : {model_type} ({params:.2f}M Parametre)")
    print(f"🎯 MODEL BAŞARI  : {map_score}")
    print(f"📦 SINIF SAYISI  : {len(model.names)}")
    print("="*45 + "\n")
    
    # 4. Besin Veri Seti Yükleme
    try:
        with open('food_dataset.json', 'r', encoding='utf-8') as f:
            food_db = json.load(f)
    except FileNotFoundError:
        print("⚠️ UYARI: food_dataset.json bulunamadı!")
        food_db = {}
        
    return model, food_db, gpu_status, device_name, map_score