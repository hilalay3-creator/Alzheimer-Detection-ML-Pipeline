import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import cross_val_score 
import warnings

warnings.filterwarnings("ignore")

def run_cross_validation(model, X, y):
    print("\n🔄 5-Katlı Çapraz Doğrulama (Cross-Validation) Başlatılıyor...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    print(f"🏆 Ortalama F1 Başarısı: %{np.mean(cv_scores)*100:.2f}")

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Özellik': feature_names, 'Önem Skoru': importances}).sort_values(by='Önem Skoru', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Önem Skoru', y='Özellik', data=feat_imp_df, palette='viridis')
    plt.title('Özellik Önem Sırası')
    plt.savefig('feature_importance.png')
    return feat_imp_df.iloc[0]['Özellik'], feat_imp_df.iloc[0]['Önem Skoru']

def save_final_report_text(most_important_feat, most_important_score, accuracy):
    print(f"\n📄 Final Raporu Hazırlanıyor... En Önemli Özellik: {most_important_feat}")

def plot_roc_curve(model, X_test, y_test):
    print("\n📈 ROC Eğrisi Analizi Başlatılıyor...")
    y_probs = model.predict_proba(X_test)[:, 1] 
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Yanlış Pozitif Oranı (FPR)')
    plt.ylabel('Doğru Pozitif Oranı (TPR)')
    plt.title('Alzheimer Teşhis Modeli ROC Eğrisi')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png')
    print(f"✅ ROC Eğrisi 'roc_curve.png' olarak kaydedildi.")

def plot_prediction_analysis(model, X_test, y_test, threshold=0.65):
    print("\n🎯 Tahmin Analizi ve Hata Payı Grafiği Hazırlanıyor...")
    y_probs = model.predict_proba(X_test)[:, 1]
    results_df = pd.DataFrame({
        'Gerçek Değer': y_test.values,
        'Tahmin Olasılığı': y_probs,
        'Vaka No': range(1, len(y_test) + 1)
    })
    results_df['Tahmin'] = (results_df['Tahmin Olasılığı'] >= threshold).astype(int)
    results_df['Durum'] = np.where(results_df['Gerçek Değer'] == results_df['Tahmin'], 'Doğru', 'Hatalı')

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=results_df, x='Vaka No', y='Tahmin Olasılığı', 
                    hue='Durum', style='Gerçek Değer', palette={'Doğru': '#2ecc71', 'Hatalı': '#e74c3c'},
                    s=100, markers={0: 'o', 1: 'P'})
    plt.axhline(y=threshold, color='black', linestyle='--', label=f'Karar Eşiği ({threshold})')
    plt.ylim(-0.05, 1.05)
    plt.title('Vaka Bazlı Tahmin Güveni ve Hata Analizi')
    plt.xlabel('Test Verisindeki Hasta Sırası (1-23)')
    plt.ylabel('Modelin Alzheimer Tahmin Olasılığı')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('prediction_error_analysis.png')
    print("✅ Tahmin analizi grafiği kaydedildi.")