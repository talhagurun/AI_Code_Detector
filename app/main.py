import streamlit as st
import joblib
import re
import pandas as pd
import time


st.set_page_config(
    page_title="AI Kod Dedektörü",
    page_icon="🕵️‍♂️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #FF4B4B; text-align: center; font-weight: bold;}
    .sub-header {font-size: 1.5rem; color: #31333F; margin-bottom: 20px;}
    .stTextArea textarea {font-family: 'Consolas', 'Courier New', monospace; font-size: 14px;}
    .result-box-ai {padding: 20px; border-radius: 10px; background-color: #f8d7da; border-left: 5px solid #dc3545; text-align: center;}
    .result-box-human {padding: 20px; border-radius: 10px; background-color: #d4edda; border-left: 5px solid #28a745; text-align: center;}
    .result-text {font-size: 1.8rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)




def clean_python_code(code):
    """Eğitimdeki temizlik fonksiyonunun aynısı."""
    if not isinstance(code, str): return ""
    # Markdown ve temel temizlik
    code = code.replace("```python", "").replace("```", "")
    # Yorum satırları ve docstring temizliği
    code = re.sub(r'(?m)^ *#.*\n?', '', code)
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    # Fazla boşlukları sil
    code = re.sub(r'\n\s*\n', '\n', code)
    return code.strip()



@st.cache_resource
def load_system():

    vectorizer = None
    models = {}
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        model_files = {
            "Logistic Regression": "model_logistic_regression.pkl",
            "Naive Bayes": "model_naive_bayes.pkl",
            "Random Forest": "model_random_forest.pkl"
        }
        for name, filename in model_files.items():
            models[name] = joblib.load(filename)
    except FileNotFoundError as e:
        st.error(f"Kritik Dosya Eksik: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"Sistem yüklenirken hata oluştu: {e}")
        st.stop()

    return vectorizer, models




st.markdown('<p class="main-header">🕵️‍♂️ AI vs. İNSAN: Kod Analizörü</p>', unsafe_allow_html=True)
st.write(
    "Bu araç, girilen Python kodunun yapısını 3 farklı Makine Öğrenmesi modeli ile analiz eder ve yapay zeka tarafından yazılma ihtimalini hesaplar.")
st.divider()


with st.spinner('Yapay Zeka Beyinleri Yükleniyor... Lütfen bekleyin...'):
    vec, loaded_models = load_system()

    time.sleep(0.5)

st.success("✅ Sistem Hazır! Modeller başarıyla yüklendi.")


st.subheader("Analiz Edilecek Kodu Yapıştırın:")
user_code_input = st.text_area(
    label="Kod Girişi",
    height=250,
    placeholder="def my_function():\n    print('Hello World')",
    label_visibility="collapsed"
)


analyze_button = st.button("🚀 ANALİZİ BAŞLAT", type="primary", use_container_width=True)

if analyze_button:
    if not user_code_input or len(clean_python_code(user_code_input)) < 5:
        st.warning("⚠️ Lütfen analiz etmek için geçerli, yorum harici içeriği olan bir kod girin.")
    else:

        with st.spinner('Kod inceleniyor, modeller karar veriyor...'):

            cleaned = clean_python_code(user_code_input)

            vectorized = vec.transform([cleaned])

            total_ai_prob = 0
            results_data = []

            st.subheader("📊 3 Farklı Modelin Kararı")


            col1, col2, col3 = st.columns(3)
            columns = [col1, col2, col3]


            i = 0
            for name, model in loaded_models.items():
                probs = model.predict_proba(vectorized)[0]
                ai_prob_percent = probs[1] * 100
                total_ai_prob += ai_prob_percent


                with columns[i]:
                    st.metric(label=name, value=f"%{ai_prob_percent:.1f} AI")

                    st.progress(probs[1], text=f"Yapay Zeka Olasılığı")
                i += 1


            avg_ai_prob = total_ai_prob / len(loaded_models)
            st.divider()

            st.subheader("📢 NİHAİ SONUÇ")

            if avg_ai_prob > 50:

                st.markdown(f"""
                <div class="result-box-ai">
                    <p>Bu kodun ortalama <strong>%{avg_ai_prob:.1f}</strong> ihtimalle</p>
                    <p class="result-text">🤖 YAPAY ZEKA (AI)</p>
                    <p>tarafından yazıldığı tahmin edilmektedir.</p>
                </div>
                """, unsafe_allow_html=True)
            else:

                st.markdown(f"""
                <div class="result-box-human">
                    <p>Bu kodun ortalama <strong>%{100 - avg_ai_prob:.1f}</strong> ihtimalle</p>
                    <p class="result-text">👤 İNSAN (HUMAN)</p>
                    <p>tarafından yazıldığı tahmin edilmektedir.</p>
                </div>
                """, unsafe_allow_html=True)


st.divider()
st.caption("Not: Bu sistem %93+ doğrulukla eğitilmiş olsa da, sonuçlar sadece birer tahmindir ve kesinlik taşımaz.")