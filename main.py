import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from src.preprocessing import handle_missing_and_clean, engineer_features, get_preprocessor
from src.model import get_best_rf_model
from src.evaluation import run_full_evaluation, run_cross_validation # Importu buraya aldık
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def run_pipeline():
    # 1. Veri Okuma ve Radikal Temizlik
    print("📂 Veri okunuyor ve gürültüler temizleniyor...")
    df = pd.read_csv('data/oasis_longitudinal.csv')
    df = df[df['Visit'] == 1]
    df = handle_missing_and_clean(df) # Z-Score temizliği burada
    
    # Hedef Düzenleme
    df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
    df['Group'] = df['Group'].map({'Nondemented': 0, 'Demented': 1})
    
    # 2. Özellik Mühendisliği (Feature Engineering)
    df = engineer_features(df)
    
    # 3. Veri Ayırma
    X = df.drop(columns=['Group', 'Subject ID', 'MRI ID', 'Hand', 'Visit', 'MR Delay', 'CDR'])
    y = df['Group']
    
    # Test setini biraz daha küçültüp eğitimi güçlendiriyoruz
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # 4. İşleme ve Eğitim
    preprocessor = get_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    print("🚀 Hiperparametre optimizasyonu ile Random Forest eğitiliyor...")
    grid_search = get_best_rf_model()
    grid_search.fit(X_train_scaled, y_train)
    
    champion_model = grid_search.best_estimator_
    print(f"✅ En İyi Ayarlar: {grid_search.best_params_}")
    
    # 5. MANUEL THRESHOLD (EŞİK) AYARI
    print("\n⚖️ Karar eşiği optimize ediliyor (Threshold: 0.65)...")
    y_probs = champion_model.predict_proba(X_test_scaled)[:, 1]
    custom_preds = (y_probs >= 0.65).astype(int)
    
    # 6. ÖZEL DEĞERLENDİRME
    print("\n📊 OPTİMİZE EDİLMİŞ TEST SONUÇLARI (N=23):")
    print(classification_report(y_test, custom_preds))
    
    # 7. Kayıt
    joblib.dump(champion_model, 'final_alzheimer_model.joblib')
    print("✅ Şampiyon model kaydedildi.")

    # 8. GÖRSELLEŞTİRME
    cm_optimized = confusion_matrix(y_test, custom_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Sağlıklı', 'Hasta'], 
                yticklabels=['Sağlıklı', 'Hasta'])
    
    current_acc = accuracy_score(y_test, custom_preds)
    plt.title(f'Optimize Edilmiş Karışıklık Matrisi (Eşik: 0.65)\nAccuracy: {current_acc:.2f}')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin Edilen')
    
    plt.savefig('final_confusion_matrix.png')
    print(f"\n📈 Yeni Karışıklık Matrisi 'final_confusion_matrix.png' olarak kaydedildi.")
    
    # 9. TÜM VERİ ÜZERİNDEN TUTARLILIK KONTROLÜ (DÜZELTİLDİ: Fonksiyon içine alındı)
    run_cross_validation(champion_model, X_train_scaled, y_train)
    
    # 10. ÖZELLİK ÖNEM SIRASI VE FİNAL RAPOR
    from src.evaluation import plot_feature_importance, save_final_report_text
    
    # Preprocessing dosyasındaki numeric_features + categorical (M/F) sırasıyla:
    # 1.Age, 2.EDUC, 3.MMSE, 4.eTIV, 5.nWBV, 6.Atrophy_Rate, 7.Cinsiyet_E
    feature_names = ['Yaş', 'Eğitim', 'MMSE (Zeka Testi)', 'eTIV (Kafa Hacmi)', 
                     'nWBV (Beyin Hacmi)', 'Atrofi Oranı', 'Cinsiyet (Erkek)']
    
    # Grafiği çiz ve en önemli özelliği al
    most_important_feat, most_important_score = plot_feature_importance(champion_model, feature_names)
    
    # Türkçe metin raporunu kaydet
    final_acc = accuracy_score(y_test, custom_preds)
    save_final_report_text(most_important_feat, most_important_score, final_acc)
    
    # 11. Kayıt 
    joblib.dump(champion_model, 'final_alzheimer_model.joblib')

if __name__ == "__main__":
    run_pipeline()