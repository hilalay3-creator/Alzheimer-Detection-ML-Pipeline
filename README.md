🧠 Alzheimer Erken Teşhis - Advanced ML Pipeline (Hilal Ay)

🔗 Veri Seti Linki (Kaggle/OASIS) Kullanılan veri setine ve tıbbi detaylarına buradan erişebilirsiniz:
OASIS-1: (https://www.kaggle.com/code/hyunseokc/detecting-early-alzheimer-s/input)

🚀 Kurulum ve Ortam Hazırlığı
 Projenin taşınabilirliği ve sürümler arası uyumluluğu için version pinning (sürüm sabitleme) uygulanmıştır. Terminalinizi açın ve proje ana dizinine gidin. 
 
 Gerekli tüm kütüphaneleri kurmak için:
 Bashpip install -r requirements.txt

Projenin ana giriş noktası main.py dosyasıdır.

🛠️ Teknik Mimari ve Mühendislik Kararları

Proje, tıbbi verilerin hassasiyeti ve küçük örneklem boyutu göz önüne alınarak şu 5 kritik sütun üzerine inşa edilmiştir:
1. Modüler Klasör Yapısı
Proje, monolitik yapıdan kurtarılarak endüstri standardı olan modüler yapıya geçirilmiştir:
src/preprocessing.py: Z-Score aykırı değer temizliği ve akıllı SimpleImputer stratejileri
src/model.py: Model mimarisi, GridSearchCV hiperparametre optimizasyonu.
src/evaluation.py: Performans analizleri, Cross-Validation ve görsel dashboard yönetimi.
main.py: Tüm pipeline'ı tek komutla yöneten orkestra şefi.

2. Pipeline Mimarisi & ColumnTransformerVeri sızıntısını (Data Leakage) önlemek için tüm adımlar bir Scikit-learn Pipeline içine alınmıştır:
Sayısal Veriler: SimpleImputer(strategy='median') + StandardScaler()
Kategorik Veriler: OneHotEncoder(handle_unknown='ignore')

Bu yapı, eğitim setindeki istatistiklerin test setine sızmasını engelleyerek tam güvenlik sağlar.

3. Z-Score & Outlier CleaningTıbbi verilerde yanıltıcı uç değerlerin (outliers) etkisini kırmak için scipy.stats kullanılarak Z-skoru 3'ten büyük olan gürültülü satırlar veri setinden ayıklanmıştır. 

Bu, modelin genelleme yeteneğini doğrudan artırmıştır.

4. Karar Eşiği Optimizasyonu (Threshold Engineering)Standart modellerde eşik 0.50'dir. Ancak yanlış alarmları (sağlıklı kişiye hasta denmesi) minimize etmek için modelin karar eşiği 0.65 olarak optimize edilmiştir. 

Bu stratejik hamle, Precision oranını ciddi şekilde yükseltmiştir.

5. Model Persistence (Kalıcılık)Eğitilen şampiyon model, joblib kütüphanesi ile paketlenerek final_alzheimer_model.joblib olarak kaydedilmiştir.

📊 Güncel Performans MetrikleriYeni mimaride elde edilen dürüst doğrulama sonuçları:

Model NameAccuracy (%)Precision (Hasta)F1-Score (Ort.)Balyoz Random Forest (Tuned) 🏆82.610.710 83

XGBoost (Baseline)78.500.650.76
Random Forest (Default)67.000.560.67

🧠 Design Choices & FAQ (Teknik Kararlar)

1. Neden Şampiyon Random Forest (Tuned) Oldu? 

Veri setinin kısıtlı olması nedeniyle XGBoost gibi kompleks modeller aşırı öğrenmeye (overfitting) meyilliydi. Random Forest, 1000 ağaçlık yapısı ve min_samples_leaf kısıtlamalarıyla, düşük örneklemde bile en kararlı (stable) sonuçları vermiştir.

2. Neden Feature Importance Analizi Yapıldı?

 Problem: Modelin neden "hasta" dediğini anlamak tıbbi güvenilirlik için şarttı.Çözüm: Yapılan analizde MMSE (Zeka Testi) %33.07 etki oranı ile lider çıkmıştır. Bu durum, modelin kararlarını rastgele değil, tıbbi literatürle uyumlu şekilde verdiğini kanıtlamıştır.
 
 📈 Görsel ÇıktılarEğitim tamamlandığında sistem otomatik olarak şu raporları üretir:
 
 final_confusion_matrix.png: Karar eşiği uygulanmış hata matrisi.feature_importance.
 png: Karar sürecindeki en etkili faktörlerin görsel sıralaması.
 sonuc_raporu.txt: Çalışmanın özet tıbbi raporu.