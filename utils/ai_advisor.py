import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- 1. IMPORT VE API KEY KISMI ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)

def get_ai_dietitian_feedback(user_profile, weekly_data):
    if not api_key:
        return "⚠️ API Anahtarı bulunamadı!"

    # Kotası en stabil olan model ismi
    selected_model_name = 'models/gemini-2.5-flash'

    try:
        model = genai.GenerativeModel(selected_model_name)
        
        # --- 2. GÜNCELLENMİŞ PROMPT KISMI ---
        prompt = f"""
        Sen 'SmartPlate Asistanı' adında, veriye dayalı konuşan uzman bir diyetisyensin.
        Kullanıcı Profili: {user_profile}
        Haftalık Beslenme Verileri: {weekly_data}

        Lütfen şu kurallara göre Türkçe, profesyonel ve motive edici bir analiz yap:

        1. **Giriş**: Gereksiz selamlamaları ("Merhaba", "Verileri inceledim" vb.) atla. Doğrudan "### 🥗 Haftalık Analiz Raporu" başlığıyla başla.
        2. **Analiz Yapısı**: Yanıtı tam olarak şu 3 ana başlıkta topla:
           - 📊 **Genel Durum ve Makro Dengesi**: Haftalık kalori ortalamasını ve protein/yağ/karbonhidrat dağılımını değerlendir. Hedeflenen günlük kaloriye ne kadar yaklaşıldığını ve makroların birbirine oranını (özellikle protein yeterliliğini) açıkla. (3-4 cümle)
           - ⚖️ **Kritik Tespitler**: Verilerde gördüğün en belirgin sapmayı (örneğin; hafta sonu aşırı kalori alımı veya karbonhidrat ağırlıklı beslenme) vurgula. Neden dikkat edilmesi gerektiğini kısaca belirt. (2-3 cümle)
           - 🚀 **Önümüzdeki Hafta İçin Strateji**: Kullanıcının hedefine ulaşması için bu hafta uygulaması gereken, somut ve ölçülebilir 2 net aksiyon öner.

        3. **Format**: Tamamen Markdown formatında yaz. Önemli sayısal değerleri **kalın** yap ve uygun emojiler kullan.
        """
        
        print(f"[LOG] {selected_model_name} ile detaylı rapor isteniyor...")
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text
        else:
            return "⚠️ AI şu an yanıt üretemedi, lütfen tekrar dene."

    except Exception as e:
        error_msg = str(e)
        print(f"[LOG] HATA DETAYI: {error_msg}")
        
        if "429" in error_msg:
            return "⚠️ Kota sınırı. Lütfen terminaldeki süreyi bekleyip tekrar dene."
        elif "404" in error_msg:
            return "❌ Model ismi bulunamadı."
        
        return f"❌ Bağlantı hatası: {error_msg}"