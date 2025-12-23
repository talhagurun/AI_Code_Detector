# AI Code Detector 

Bu proje, bir kod bloğunun bir insan tarafından mı yoksa yapay zeka (LLM) tarafından mı yazıldığını tespit etmek amacıyla geliştirilmiştir.

## Özellikler
- **Çoklu Model Desteği:** Logistic Regression, Naive Bayes ve Random Forest modelleri.
- **Yüksek Doğruluk:** Temizlenmiş veri setleri ile optimize edilmiş modeller.
- **Kullanıcı Dostu Arayüz:** Streamlit tabanlı web arayüzü.

## Yazılım Kalite ve Test
- **SonarQube:** Kod kalitesi "A" seviyesinde doğrulanmıştır.
- **White Box Testing:** `unittest` ile fonksiyonel birim testleri yapılmıştır.

## Kurulum
1. `git clone https://github.com/kullanici_adin/AI_Code_Detector.git`
2. `pip install -r requirements.txt`
3. `streamlit run app.py`