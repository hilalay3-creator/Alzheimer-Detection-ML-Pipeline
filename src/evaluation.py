import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score 
import warnings
warnings.filterwarnings("ignore") # Tüm sarı uyarıları susturur

def run_full_evaluation(model, name, X_test, y_test):
    pass # Bu fonksiyonun içeriği main.py içine taşındı, çünkü orada daha fazla bilgiye erişim var (örneğin X_train_scaled, y_train gibi) ve bu sayede cross-validation da yapılabiliyor.

def run_cross_validation(model, X, y):
    print("\n🔄 5-Katlı Çapraz Doğrulama (Cross-Validation) Başlatılıyor...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    
    print(f"📈 Her Katın F1 Skorları: {cv_scores}")
    print(f"🏆 Ortalama F1 Başarısı: %{np.mean(cv_scores)*100:.2f}")
    print(f"🔍 Standart Sapma (Tutarlılık): +/- {np.std(cv_scores)*100:.2f}")
    
    if np.std(cv_scores) < 0.15:
        print("✅ Model çok tutarlı, farklı veri gruplarında benzer başarıyı gösteriyor.")
    else:
        print("⚠️ Model bazı veri gruplarında zorlanıyor, veri miktarını artırmak iyi olabilir.")
def plot_feature_importance(model, feature_names):
    """
    Modelin en önemli bulduğu özellikleri Türkçe grafik olarak basar.
    """
    print("\n📊 Özellik Önem Sırası (Feature Importance) Analizi Başlatılıyor...")
    
    # Random Forest'tan önem katsayılarını al
    importances = model.feature_importances_
    
    # Veriyi tabloya dök ve sırala
    feat_imp_df = pd.DataFrame({
        'Özellik': feature_names,
        'Önem Skoru': importances
    }).sort_values(by='Önem Skoru', ascending=False)

    # Grafik Çizimi
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Önem Skoru', y='Özellik', data=feat_imp_df, palette='viridis')
    plt.title('Alzheimer Teşhisinde En Etkili Tıbbi Faktörler (Random Forest)')
    plt.xlabel('Modelin Kararındaki Etki Oranı')
    plt.ylabel('Tıbbi Veri / Test Türü')
    plt.tight_layout()
    
    # Grafiği kaydet
    plt.savefig('feature_importance.png')
    print("📈 Özellik Önem Grafiği 'feature_importance.png' olarak kaydedildi.")
    
    # En önemli özelliği bul
    most_important_feat = feat_imp_df.iloc[0]['Özellik']
    most_important_score = feat_imp_df.iloc[0]['Önem Skoru']
    
    return most_important_feat, most_important_score

def save_final_report_text(most_important_feat, most_important_score, accuracy):
    """
    Raporu hem .txt olarak kaydeder hem de ekranda bir figure penceresi olarak açar.
    """
        
    # --- YENİ: GÖRSEL RAPOR PENCERESİ (FIGURE) ---
    plt.figure(figsize=(10, 7), facecolor='#f0f0f0')
    plt.axis('off') # Eksenleri kapatıyoruz, sadece yazı yazacağız
    
    rapor_baslik = "ALZHEIMER ERKEN TEŞHİS MODELİ - FİNAL ÖZETİ"
    rapor_icerik = (
        f"----------------------------------------------------------------------\n"
        f"*** MODEL PERFORMANSI ***\n" 
        f"Genel Doğruluk (Accuracy): %{accuracy*100:.2f}\n"
        f"Karar Eşiği (Threshold): 0.65\n"
        f"----------------------------------------------------------------------\n\n"
        f"--- TIBBİ ANALİZ SONUCU ---\n"
        f"Alzheimer riskini belirleyen EN KRİTİK FAKTÖR:\n\n"
        f"-> {most_important_feat} <-\n\n"
        f"Modelin kararlarındaki etki gücü: %{most_important_score*100:.2f}\n"
        f"----------------------------------------------------------------------\n"
        f"NOT: Bu sonuç, {most_important_feat} skorunun\n"
        f"erken teşhis için en güçlü biyobelirteç olduğunu kanıtlar.\n"
        f"----------------------------------------------------------------------"
    )

    # Yazıları pencereye yerleştir
    plt.text(0.5, 0.9, rapor_baslik, fontsize=16, fontweight='bold', ha='center', color='#2c3e50')
    plt.text(0.5, 0.4, rapor_icerik, fontsize=13, ha='center', va='center', 
             bbox=dict(boxstyle='round,pad=1', fc='white', ec='#34495e', alpha=0.9))

    plt.tight_layout()
    plt.show() 