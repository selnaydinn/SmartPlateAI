import streamlit as st
from PIL import Image
import datetime
import pandas as pd
import os
import torch
import json
import time
import plotly.graph_objects as go
import plotly.express as px

# --- 0. SİSTEM VE VERİTABANI HAZIRLIĞI ---
from utils.system_config import initialize_system
from utils.analyze_image import analyze_plate
from utils.db_manager import (
    init_db, 
    save_meal, 
    get_daily_totals, 
    delete_meal, 
    get_weekly_data, 
    get_today_meals_detailed, 
    get_user_profile, 
    save_user_profile
)
from utils.constants import FOOD_TR_MAP, AYLAR_MAP
from utils.ai_advisor import get_ai_dietitian_feedback

# Veritabanı başlatma
init_db() 

# --- INITIAL LOAD (Verileri Cache'e Yükleme) ---
if "initial_load_done" not in st.session_state:
    today_meals = get_today_meals_detailed(1) 
    
    for meal in today_meals:
        m_name, cal, pro, carb, fat, d_json = meal
        
        try:
            d_list = json.loads(d_json) if d_json else []
        except (json.JSONDecodeError, TypeError):
            d_list = []
            
        st.session_state[f"res_{m_name}"] = ({
            "cal": cal, 
            "pro": pro, 
            "carb": carb, 
            "fat": fat, 
            "details": d_list
        }, None)
        
    st.session_state.initial_load_done = True

# Sistem Parametreleri
model, food_db, is_gpu, gpu_name, model_accuracy = initialize_system()
device = '0' if is_gpu else 'cpu'

# Besin İsimleri Sözlüğü
valid_food_names = sorted([name.lower() for name in food_db.keys()])
display_food_names = {
    name: FOOD_TR_MAP.get(name, name).capitalize() 
    for name in valid_food_names
}

# Sayfa Durumu
if 'page' not in st.session_state:
    st.session_state.page = "Öğün Analizi"

# --- 1. SAYFA AYARLARI VE CSS ---
st.set_page_config(
    page_title="SmartPlate AI", 
    page_icon="🥗", 
    layout="wide"
)

st.markdown("""
    <style>
    :root {
        --primary-green: #27ae60;
        --secondary-green: #2ecc71;
        --light-green: #f1f8f4;
        --dark-green: #1b5e20;
    }

    /* Genel Bileşenler */
    div[data-baseweb="tab-highlight"] { 
        background-color: var(--primary-green) !important; 
    }
    
    button[aria-selected="true"] p { 
        color: var(--primary-green) !important; 
        font-weight: bold; 
    }
    
    button[kind="primary"] { 
        background-color: var(--primary-green) !important; 
        border: none !important; 
        color: white !important; 
    }

    /* Özet Kartları Tasarımı */
    .total-card { 
        background-color: var(--light-green); 
        padding: 20px; 
        border-radius: 15px; 
        border-left: 5px solid var(--primary-green); 
        text-align: center; 
        margin-bottom: 20px; 
    }
    
    .total-val { 
        font-size: 32px; 
        font-weight: bold; 
        color: var(--primary-green); 
    }
    
    .total-label { 
        color: #555; 
        font-size: 14px; 
        font-weight: 600; 
    }

    /* Bilgi Kutusu (Info Box) */
    .info-box {
        background-color: var(--light-green);
        color: var(--primary-green);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid var(--secondary-green);
        margin-bottom: 20px;
        font-size: 14px;
        display: flex;
        align-items: center;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.05);
    }
    
    .info-icon { 
        font-size: 20px; 
        margin-right: 12px; 
    }
    
    .info-text b { 
        color: var(--dark-green); 
    }

    /* Resim Standardizasyonu */
    [data-testid="stImage"] img { 
        border-radius: 10px; 
        display: block; 
        margin: auto; 
        max-height: 400px; 
        width: auto !important; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR ---
profile_data = get_user_profile(1)

if profile_data:
    _, d_gender, d_age, d_height, d_weight, d_target, d_activity, d_goal = profile_data
else:
    _, d_gender, d_age, d_height, d_weight, d_target, d_activity, d_goal = \
        1, "Erkek", 25, 175, 75.0, 70.0, "Orta", 2000

with st.sidebar:
    st.title("🥗 SmartPlate AI")
    
    now = datetime.datetime.now()
    st.markdown(
        f'<div style="margin-bottom: 25px; font-size: 15px; font-weight: bold; color: #444;">'
        f'🗓️ Bugün: <span style="font-weight: normal; color: #666;">'
        f'{now.day} {AYLAR_MAP[now.month]} {now.year}</span></div>', 
        unsafe_allow_html=True
    )
    
    st.divider()

    st.markdown(f"""
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 15px; border-left: 5px solid #27ae60; margin-bottom: 20px;">
            <p style="color: #27ae60; margin: 0; font-weight: bold; font-size: 13px; letter-spacing: 1px;">GÜNLÜK HEDEF</p>
            <p style="color: #1b5e20; margin: 0; font-size: 30px; font-weight: 800;">{d_goal} <span style="font-size: 16px;">kcal</span></p>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Profil Ayarları"):
        gender = st.selectbox("Cinsiyet", ["Erkek", "Kadın"], index=0 if d_gender == "Erkek" else 1)
        age = st.number_input("Yaş", 10, 100, int(d_age))
        height = st.number_input("Boy (cm)", 100, 250, int(d_height))
        user_weight = st.number_input("Mevcut (kg)", value=float(d_weight), step=0.1)
        target_weight = st.number_input("Hedef (kg)", value=float(d_target), step=0.1)
        activity = st.select_slider("Hareketlilik", options=["Az", "Hafif", "Orta", "Çok"], value=d_activity)

        if st.button("Bilgileri Güncelle", use_container_width=True, type="primary"):
            akt_map = {"Az": 1.2, "Hafif": 1.375, "Orta": 1.55, "Çok": 1.725}
            bmr = (10 * user_weight + 6.25 * height - 5 * age + (5 if gender == "Erkek" else -161))
            new_goal = int(bmr * akt_map[activity])
            save_user_profile(1, gender, age, height, user_weight, target_weight, activity, new_goal)
            st.toast("Profil güncellendi! 🎉")
            st.rerun()

    st.divider()
    
    if st.button("Öğün Analizi", use_container_width=True): 
        st.session_state.page = "Öğün Analizi"
        st.rerun()
    if st.button("Günlük Rapor", use_container_width=True): 
        st.session_state.page = "Günlük Rapor"
        st.rerun()
    if st.button("Haftalık Analiz", use_container_width=True): 
        st.session_state.page = "Haftalık Analiz"
        st.rerun()

# --- 3. SAYFA: ÖĞÜN ANALİZİ ---
if st.session_state.page == "Öğün Analizi":
    st.title("Öğün Analizi")
    total_cal, total_pro, total_carb, total_fat = get_daily_totals(1)
    
    st.markdown("### Bugünün Özeti")
    c1, c2, c3, c4 = st.columns(4)
    cards = [(c1, "Kalori", int(total_cal), ""), (c2, "Protein", int(total_pro), "g"), (c3, "Karb", int(total_carb), "g"), (c4, "Yağ", int(total_fat), "g")]
    
    for col, label, val, unit in cards:
        col.markdown(f'<div class="total-card"><p class="total-label">{label}</p><p class="total-val">{val}{unit}</p></div>', unsafe_allow_html=True)

    meal_list = ["Sabah", "Öğle", "Akşam", "Atistirmalik"]
    tabs = st.tabs(meal_list)
    
    for meal_name, tab in zip(meal_list, tabs):
        with tab:
            if f"res_{meal_name}" in st.session_state:
                rep = st.session_state[f"res_{meal_name}"][0]
                st.success(f"{meal_name} Kaydedildi")
                st.table(pd.DataFrame(rep["details"])[["yiyecek", "gramaj", "kalori"]])
                
                m_cols = st.columns(4)
                m_cols[0].metric("Enerji", f"{int(rep['cal'])} kcal")
                m_cols[1].metric("Protein", f"{int(rep['pro'])}g")
                m_cols[2].metric("Karb", f"{int(rep['carb'])}g")
                m_cols[3].metric("Yağ", f"{int(rep['fat'])}g")
                
                if st.button("Sil ve Yenile", key=f"del_{meal_name}"):
                    delete_meal(1, meal_name)
                    if f"res_{meal_name}" in st.session_state: del st.session_state[f"res_{meal_name}"]
                    if "initial_load_done" in st.session_state: del st.session_state.initial_load_done
                    st.rerun()
            else:
                col_in, col_out = st.columns([1, 1.2])
                with col_in:
                    t_tipi = st.selectbox("Tabak Boyutu", ["Standart (24cm)", "Küçük (18cm)", "Büyük (30cm)"], key=f"t_{meal_name}")
                    st.markdown('<div class="info-box"><span class="info-icon">💡</span><div class="info-text"><b>İpucu:</b> Analizin doğru olması için tabağı üstten ve tam kadraja alarak fotoğraflayın.</div></div>', unsafe_allow_html=True)
                    uploaded = st.file_uploader("Fotoğraf Yükle", type=['jpg','png','jpeg'], key=f"f_{meal_name}")
                    
                    if uploaded:
                        st.image(Image.open(uploaded))
                        if st.button("Analiz Et", key=f"b_{meal_name}", use_container_width=True, type="primary"):
                            cap = 24.0 if "Standart" in t_tipi else (18.0 if "Küçük" in t_tipi else 30.0)
                            h_carp = 1.0 if "Standart" in t_tipi else (1.15 if "Küçük" in t_tipi else 0.85)
                            with st.spinner("AI analiz ediyor..."):
                                report, analyzed_img = analyze_plate(Image.open(uploaded), model, food_db, device, cap, h_carp)
                                print("LOG AI görseli inceledi...")

                                for i, det in enumerate(report["details"]):
                                    # Her besine benzersiz bir u_id atıyoruz
                                    det["u_id"] = f"{det.get('label_en', 'food')}_{int(time.time())}_{i}"
            
                                st.session_state[f"temp_report_{meal_name}"] = report
                                st.session_state[f"temp_img_{meal_name}"] = analyzed_img
                                st.rerun()

                with col_out:
                    if f"temp_report_{meal_name}" in st.session_state:
                        if f"edited_list_{meal_name}" not in st.session_state:
                            print(f"--- [YÜKLEME] {meal_name} listesi İLK KEZ oluşturuluyor ---")
                            st.session_state[f"edited_list_{meal_name}"] = st.session_state[f"temp_report_{meal_name}"]["details"]

                        with st.popover("🖼️ Görsel Analizi Gör"): 
                            st.image(st.session_state[f"temp_img_{meal_name}"], use_container_width=True)
                        
                        st.markdown("### 📝 Tespitleri Onayla")
                        current_list = st.session_state[f"edited_list_{meal_name}"]

                        # --- DEBUG ---
                        print(f"\n--- [RENDER] {meal_name} ŞU ANKİ LİSTE: ---")
                        for i, x in enumerate(current_list):
                            print(f"İndeks {i}: {x.get('label_en')} (ID: {x.get('u_id')})")
                        
                        for idx, item in enumerate(current_list):
                            u_id = item.get("u_id", f"old_{idx}") # Eşsiz ID'yi al
                            ce1, ce2, ce3 = st.columns([2.5, 1.2, 0.5])
                            
                            with ce1:
                                d_en = item.get("label_en", "").lower().strip()
                                d_idx = valid_food_names.index(d_en) if d_en in valid_food_names else 0
                                # KEY: u_id kullanarak sabitliyoruz
                                st.selectbox(f"Besin {idx+1}", valid_food_names, index=d_idx, format_func=lambda x: display_food_names.get(x, x), key=f"sel_{u_id}")
                            
                            with ce2:
                                safe_gram = max(1, int(item.get("gramaj", 100)))
                                # KEY: u_id kullanarak sabitliyoruz
                                st.number_input("Gram", value=safe_gram, min_value=1, key=f"gr_{u_id}")
                            
                            with ce3:
                                st.markdown("<br>", unsafe_allow_html=True)
                                if st.button("🗑️", key=f"btn_del_{u_id}"):
                                    # Listeden sil
                                    st.session_state[f"edited_list_{meal_name}"].pop(idx)
                                    
                                    # Session state temizliği
                                    if f"sel_{u_id}" in st.session_state: del st.session_state[f"sel_{u_id}"]
                                    if f"gr_{u_id}" in st.session_state: del st.session_state[f"gr_{u_id}"]
                                    
                                    print(f"LOG: {u_id} silindi, sayfa yenileniyor...")
                                    st.rerun()

                        if len(st.session_state[f"edited_list_{meal_name}"]) < 10:
                            with st.expander("➕ Eksik Besin Ekle"):
                                ae1, ae2 = st.columns([2, 1])
                                with ae1:
                                    add_name = st.selectbox("Besin Seç", valid_food_names, format_func=lambda x: display_food_names.get(x, x), key=f"add_n_{meal_name}")
                                with ae2:
                                    add_gram = st.number_input("Gramaj", value=100, step=10, key=f"add_g_{meal_name}")
                                if st.button("Listeye Ekle", use_container_width=True, key=f"add_btn_{meal_name}"):
                                    new_item_id = f"manual_{int(time.time())}"
                                    st.session_state[f"edited_list_{meal_name}"].append({"label_en": add_name, "gramaj": add_gram, "u_id": new_item_id})
                                    st.rerun()

                        st.divider()
                        if st.button("✅ Öğünü Kaydet", key=f"save_{meal_name}", type="primary", use_container_width=True):
                            final_details = []
                            # Liste üzerinden dönüp u_id ile widget değerlerini topluyoruz
                            for itm in st.session_state[f"edited_list_{meal_name}"]:
                                c_uid = itm.get("u_id")
                                s_food = st.session_state[f"sel_{c_uid}"]
                                s_gram = st.session_state[f"gr_{c_uid}"]
                                
                                base = food_db[s_food]
                                ratio = s_gram / 100.0
                                final_details.append({
                                    "yiyecek": display_food_names.get(s_food, s_food), 
                                    "label_en": s_food, 
                                    "gramaj": s_gram,
                                    "u_id": c_uid,
                                    "kalori": base.get('kcal', 0) * ratio, 
                                    "pro": base.get('protein', 0) * ratio,
                                    "carb": base.get('carb', 0) * ratio, 
                                    "fat": base.get('fat', 0) * ratio
                                })
                            
                            final_rep = {
                                "cal": sum(d['kalori'] for d in final_details), 
                                "pro": sum(d['pro'] for d in final_details),
                                "carb": sum(d['carb'] for d in final_details), 
                                "fat": sum(d['fat'] for d in final_details),
                                "details": final_details
                            }
                            save_meal(1, meal_name, final_rep)
                            st.session_state[f"res_{meal_name}"] = (final_rep, None)
                            if f"temp_report_{meal_name}" in st.session_state: del st.session_state[f"temp_report_{meal_name}"]
                            if f"edited_list_{meal_name}" in st.session_state: del st.session_state[f"edited_list_{meal_name}"]
                            st.toast("Öğün başarıyla kaydedildi! 🍎")
                            st.rerun()

# --- 4. SAYFA: GÜNLÜK RAPOR ---
elif st.session_state.page == "Günlük Rapor":
    st.title("Günlük Beslenme Analizi")
    
    # Verileri çek
    t_cal, t_pro, t_carb, t_fat = get_daily_totals(1)
    
    # --- Üst Kısım: Kalori Hedef Barı ---
    oran = t_cal / d_goal if d_goal > 0 else 0
    yuzde = int(oran * 100)
    
    # Renk belirleme (Hedefi aşarsa kırmızı, yaklaşırsa yeşil)
    bar_color = "#27ae60" if oran <= 1.0 else "#e74c3c"
    
    st.markdown(f"**Günlük Kalori Hedefi: {int(t_cal)} / {int(d_goal)} kcal (%{yuzde})**")

    # Progress Bar rengini dinamik olarak (yeşil veya kırmızı) ayarlar
    st.markdown(f"""
        <style>
        div[data-testid="stProgress"] > div > div > div > div {{
            background-color: {bar_color} !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    st.progress(min(oran, 1.0))
    
    if oran > 1.0:
        st.warning(f"⚠️ Günlük kalori hedefini {int(t_cal - d_goal)} kcal aştınız!")

    st.divider()

    # --- Orta Kısım: Grafikler ---
    col_pie, col_bar = st.columns([1, 1.2])

    with col_pie:
        st.subheader("Makro Dengesi")
        if (t_pro + t_carb + t_fat) > 0:
            # Şık bir Donut Chart
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Protein', 'Karb', 'Yağ'],
                values=[t_pro, t_carb, t_fat],
                hole=.5,
                marker=dict(colors=['#27ae60', '#2ecc71', '#a2d149']),
                textinfo='percent+label',
                hoverinfo='label+value'
            )])
            fig_donut.update_layout(
                margin=dict(t=30, b=0, l=0, r=0),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("Henüz veri yok.")

    with col_bar:
        st.subheader("Öğün Dağılımı")
        today_meals = get_today_meals_detailed(1)
        if today_meals:
            # Öğün isimlerini ve kalorilerini listeye al
            m_names = [m[0] for m in today_meals]
            m_cals = [m[1] for m in today_meals]
            
            # İnteraktif Bar Chart
            fig_meals = px.bar(
                x=m_names, 
                y=m_cals,
                labels={'x': '', 'y': 'Kalori (kcal)'},
                color=m_names,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_meals.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_meals, use_container_width=True)
        else:
            st.info("Bugün kaydedilmiş öğün bulunmuyor.")

    st.divider()

    # --- Alt Kısım: Büyük Metrikler ---
    st.markdown("### Besin Değerleri")
    m1, m2, m3, m4 = st.columns(4)
    
    # Tasarımı güzelleştirmek için kart formatında metrikler
    m1.metric("Toplam Kalori", f"{int(t_cal)} kcal", delta=f"{int(d_goal - t_cal)} kalan" if d_goal > t_cal else "Limit aşıldı", delta_color="normal")
    m2.metric("Protein", f"{int(t_pro)}g", "Kas Yapımı")
    m3.metric("Karbonhidrat", f"{int(t_carb)}g", "Enerji")
    m4.metric("Yağ", f"{int(t_fat)}g", "Sağlıklı Yağ")

# --- 5. SAYFA: HAFTALIK ANALİZ ---
elif st.session_state.page == "Haftalık Analiz":
    st.title("Haftalık Analiz")
    df_weekly = get_weekly_data(1) 
    if not df_weekly.empty:
        st.line_chart(df_weekly.set_index('date')['total_cal'], color="#27ae60")
        if "weekly_report_cache" in st.session_state:
            st.markdown(st.session_state["weekly_report_cache"])
            if st.button("Raporu Yenile 🔄"): 
                del st.session_state["weekly_report_cache"]
                st.rerun()
        else:
            if st.button("Haftalık Analiz Raporu Al", type="primary"): 
                with st.spinner("AI Diyetisyen inceliyor..."):
                    rapor = get_ai_dietitian_feedback(f"Cinsiyet: {d_gender}, Yaş: {d_age}", df_weekly.to_string())
                    print("LOG AI raporu gönderdi...")
                    if rapor: 
                        st.session_state["weekly_report_cache"] = rapor
                        st.rerun()
    else: st.info("Haftalık veri bulunmuyor.")