from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


def train_logistic_regression(X_train, y_train):
    """
    [what it does]
    @param X_train: training feature matrix
    @param y_train: training labels
    @return: fitted model
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """
    [what it does]
    @param X_train: training feature matrix
    @param y_train: training labels
    @return: fitted model
    """

    #create SVM model
    svm_model = LinearSVC(C=1, max_iter=1000)
        
    #train
    svm_model.fit(X_train, y_train)
    return svm_model

def train_mlp(X_train, y_train):
    """
    [what it does]
    @param X_train: training feature matrix
    @param y_train: training labels
    @return: fitted model
    """
    # create MLP model
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=50, random_state=42)
        
    # fit
    clf.fit(X_train, y_train)
    return clf

MODEL_REGISTRY = {
    'LR': train_logistic_regression,
    'SVM': train_svm,
    'MLP': train_mlp,
}
       