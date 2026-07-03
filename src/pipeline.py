import pandas as pd
from sklearn.model_selection import train_test_split
from src.models import MODEL_REGISTRY
from src.preprocessing import get_features
from src.evaluate import evaluate_model, get_model_scores

def train_and_evaluate(toxic_df, model_name, feature ):
    
    train_fn=MODEL_REGISTRY[model_name]
    X = toxic_df['comment_text']
    classes = ['toxic', 'severe_toxic','obscene', 'threat', 'insult','identity_hate' ]
    results = []
    
    for class_name in classes:
        try:
            print(f'\n--- {class_name} ---')
            y = toxic_df[class_name]
        
            X_train_text, X_test_text, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
            )
        
            X_train, X_test = get_features(X_train_text, X_test_text, method=feature)
        
            model = train_fn(X_train, y_train)
            y_pred = model.predict(X_test)
            y_scores = get_model_scores(model, X_test)
        
            metrics = evaluate_model(y_test, y_pred, y_scores)
            metrics.update({'class': class_name, 'model': model_name, 'feature': feature})
            results.append(metrics)
        
        except Exception as e:
            print(f'Error for {class_name}: {e}')
    return results

def run_full_comparison(df, models=('LR', 'SVM', 'MLP'), 
                         feature_methods=('tfidf', 'word2vec')):
    all_results = []
    for feature in feature_methods:
        for model_name in models:
            all_results.extend(train_and_evaluate(df, model_name, feature))
    return pd.DataFrame(all_results)