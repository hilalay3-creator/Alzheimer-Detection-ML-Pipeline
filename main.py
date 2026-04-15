import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from src.preprocessing import handle_missing_and_clean, engineer_features, get_preprocessor
from src.model import get_best_rf_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.evaluation import (run_cross_validation, plot_roc_curve, 
                            plot_feature_importance, save_final_report_text, 
                            plot_prediction_analysis)

def run_pipeline():
    print("📂 Veri okunuyor...")
    df = pd.read_csv('data/oasis_longitudinal.csv')
    df = df[df['Visit'] == 1]
    df = handle_missing_and_clean(df) 
    
    df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
    df['Group'] = df['Group'].map({'Nondemented': 0, 'Demented': 1})
    
    df = engineer_features(df)
    X = df.drop(columns=['Group', 'Subject ID', 'MRI ID', 'Hand', 'Visit', 'MR Delay', 'CDR'])
    y = df['Group']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    preprocessor = get_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    print("🚀 Model eğitiliyor...")
    grid_search = get_best_rf_model()
    grid_search.fit(X_train_scaled, y_train)
    champion_model = grid_search.best_estimator_
    
    y_probs = champion_model.predict_proba(X_test_scaled)[:, 1]
    custom_preds = (y_probs >= 0.65).astype(int)
    
    print("\n📊 SONUÇLAR:")
    print(classification_report(y_test, custom_preds))
    
    # Grafikler
    plot_roc_curve(champion_model, X_test_scaled, y_test)
    plot_prediction_analysis(champion_model, X_test_scaled, y_test, threshold=0.65)
    
    feature_names = ['Yaş', 'Eğitim', 'MMSE', 'eTIV', 'nWBV', 'Atrofi Oranı', 'Cinsiyet']
    most_important_feat, most_important_score = plot_feature_importance(champion_model, feature_names)
    
    save_final_report_text(most_important_feat, most_important_score, accuracy_score(y_test, custom_preds))
    print("✅ BÜTÜN GRAFİKLER OLUŞTURULDU.")

if __name__ == "__main__":
    run_pipeline()