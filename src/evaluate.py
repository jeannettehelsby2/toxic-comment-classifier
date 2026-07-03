from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score


def evaluate_model(y_test, y_pred, y_scores):
    """
    Calculates classification metrics for a single model run.
    
    @param y_test: true labels for the test set
    @param y_pred: predicted labels from the model
    @param y_scores: probability or decision function scores for AUC
    @return: dict containing accuracy, f1, recall, precision and auc
    """

    return { 
        'accuracy': accuracy_score(y_test, y_pred),
        'f1':f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'auc': roc_auc_score(y_test,y_scores)
    }
    

def get_model_scores(model, X_test):
    """
    Returns probability-like scores for each test sample, handling both
    predict_proba and decision_function models.
    
    @param model: fitted sklearn classifier
    @param X_test: feature matrix for the test set
    @return: 1D array of scores for AUC calculation
    """

    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)

