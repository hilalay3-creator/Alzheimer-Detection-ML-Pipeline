from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def get_best_rf_model():
    # class_weight='balanced' yaparak hasta sayısının azlığını telafi ediyoruz
    param_grid = {
        'n_estimators': [500, 1000], 
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }
    
    rf = RandomForestClassifier(random_state=42)
    return GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)