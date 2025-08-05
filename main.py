import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBClassifier
import os
import seaborn as sns
from datetime import datetime
from sklearn.metrics import roc_curve, auc, precision_score, f1_score, classification_report, accuracy_score, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
import joblib




DECISION_TREE_PARAMS = {
    'criterion': ['entropy'],
    'max_depth': [20],
    'min_samples_split': [10],
    'min_samples_leaf': [1],
    'max_features': [None]
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': [50, 75, 100],
    'max_depth': [20],
    'min_samples_split': [2],
    'min_samples_leaf': [5, 7, 10],
    'max_features': ['log2']
}

EXTRA_TREES_PARAMS = {
    'n_estimators': [200, 300, 500],
    'criterion': ['gini'],
    'max_depth': [20, 30, 50, 100],
    'min_samples_split': [5],
    'min_samples_leaf': [5, 7, 15],
    'max_features': ['log2']
}

MLP_PARAMS = {
    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [1e-4, 1e-3, 1e-2],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [200, 400, 500, 700, 1000]
}

HISTOGRAM_GBM_PARAMS = {
    'loss': ['auto', 'log_loss'],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [10, 20, 50],
    'max_leaf_nodes': [None, 31, 63]
}

XGBOOST_PARAMS = {
    'n_estimators': [200],
    'max_depth': [7],
    'learning_rate': [0.05],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}


LOGISTIC_REGRESSION_PARAMS = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear'], 'max_iter': [100000]}
LDA_PARAMS = {'solver': ['svd', 'lsqr']}
QDA_PARAMS = {}
KNN_PARAMS = {'n_neighbors': [3, 5, 7]}
GRADIENT_BOOSTING_PARAMS = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
LIGHTGBM_PARAMS = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5], 'verbose': [-1]}
CATBOOST_PARAMS = {'iterations': [50, 100], 'learning_rate': [0.01, 0.1], 'depth': [3, 5]}
SVM_PARAMS = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
NU_SVC_PARAMS = {'nu': [0.1, 0.5], 'kernel': ['rbf', 'linear']}
LINEAR_SVC_PARAMS = {'C': [0.1, 1, 10]}
GAUSSIAN_NB_PARAMS = {}
GAUSSIAN_PROCESS_PARAMS = {'max_iter_predict': [100], 'kernel': [1.0 * RBF(1.0)]}
SGD_PARAMS = {'max_iter': [500, 1000], 'tol': [1e-3, 1e-4]}
PASSIVE_AGGRESSIVE_PARAMS = {'C': [0.1, 1, 10], 'max_iter': [500, 1000], 'tol': [1e-3, 1e-4]}
PERCEPTRON_PARAMS = {'max_iter': [500, 1000], 'tol': [1e-3, 1e-4]}
RIDGE_PARAMS = {'alpha': [0.1, 1, 10]}
BAGGING_PARAMS = {'n_estimators': [5, 10]}
ADABOOST_PARAMS = {'n_estimators': [50, 100], 'learning_rate': [0.1, 1.0]}

ENABLE_GRID_SEARCH = False

def get_used_hyperparameters(model, model_name):
    used_params = model.get_params()
    return used_params


def load_and_preprocess_data(file_path):
    print("Loading and preprocessing data...")
    
    data = pd.read_csv(file_path)
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data loaded and preprocessed successfully.")
    return X_train, X_test, y_train, y_test
    

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, params):
    print(f"Training and evaluating {model_name}...")
    starting_time = datetime.now()
    print("Recieved date and time.")
    if ENABLE_GRID_SEARCH and hasattr(model, 'get_params'):
        try:
            grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        except Exception as e:
            print(f"Grid search failed for {model_name}: {str(e)}")
            print("Falling back to default parameters.")
            model.fit(X_train, y_train)
    else:
        print("Fitting model:")
        model.fit(X_train, y_train)
        print("Model Fit")

    print("Running cross validation:")
    cv = StratifiedKFold(n_splits=5)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print("Predicting:")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    print("Metrics Calculated:")
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_binarized.shape[1]

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)
    else:
        y_scores = model.decision_function(X_test)


    
    font_size = 25
    legend_font_size = 20
    print("Creating Precision Recall Plot:")
    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_binarized[:, i], y_scores[:, i])
        pr_auc = auc(recall_curve, precision_curve)
        plt.plot(recall_curve, precision_curve, lw=2, label=f'Class {i} (area = {pr_auc:.4f})')

    plt.xlabel('Recall', fontsize=font_size)  # Larger and bold label
    plt.ylabel('Precision', fontsize=font_size)  # Larger and bold label
    plt.legend(loc='lower left', fontsize=legend_font_size)  # Larger legend
    plt.xticks(fontsize=font_size-2)  # Larger x-axis ticks
    plt.yticks(fontsize=font_size-2)  # Larger y-axis ticks
    plt.savefig(f"../results/precision_recall_curves/{model_name}_PR_curve.jpeg")
    plt.close()

    print("Creating ROC Plot:")
    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate', fontsize=font_size)  # Larger and bold label
    plt.ylabel('True Positive Rate', fontsize=font_size)  # Larger and bold label
    plt.legend(loc='lower right', fontsize= legend_font_size)  # Larger legend
    plt.xticks(fontsize=font_size-2)  # Larger x-axis ticks
    plt.yticks(fontsize=font_size-2)  # Larger y-axis ticks
    plt.savefig(f"../results/ROC_curves/{model_name}_ROC_curve.jpeg")
    plt.close()

    print("Creating Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), annot_kws={"size": 20})  # Increase annotation font size
    plt.xlabel("Predicted Label", fontsize=font_size)
    plt.ylabel("True Label", fontsize=font_size)
    plt.title(f"Confusion Matrix for {model_name}", fontsize=font_size)
    plt.xticks(fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    plt.savefig(f"../results/confusion_matrices/{model_name}_confusion_matrix.jpeg")
    plt.close()


    used_params = get_used_hyperparameters(model, model_name)


    print(f"{model_name} trained and evaluated successfully. Accuracy: {accuracy:.4f}")
    end_time = datetime.now()
    total_time = end_time-starting_time
    return model, accuracy, precision, f1, cv_scores, report, used_params, total_time
    





def save_model(model, model_name, params):
    print(f"Saving {model_name}...")
    try:
        save_dir = "Models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        param_str = '_'.join([f"{k}_{v}" for k, v in params.items() if k != 'estimators'])[:100]
        filename = f"pkl/{model_name}.pkl"  # Use .pkl extension
        filepath = os.path.join(save_dir, filename)

        if hasattr(model, "fit"):  # Scikit-learn model
            joblib.dump(model, filepath)
        else:
            raise ValueError("Unsupported model type: Cannot save.")

        print(f"{model_name} saved successfully at {filepath}.")
    except Exception as e:
        print(f"Error in saving {model_name}: {str(e)}")



def run_pipeline(input_file):


    print("Starting the pipeline...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(input_file)
    
    n_classes = len(np.unique(y_train))
    print(f"Number of classes in the dataset: {n_classes}")
    if n_classes < 2:
        raise ValueError("The dataset must contain at least 2 classes for classification tasks.")
    
    models = [
        (DecisionTreeClassifier(), "DecisionTree", DECISION_TREE_PARAMS),
        (RandomForestClassifier(), "RandomForest", RANDOM_FOREST_PARAMS),
        (ExtraTreesClassifier(), "ExtraTrees", EXTRA_TREES_PARAMS),
        (XGBClassifier(), "XGBoost", XGBOOST_PARAMS),
        (MLPClassifier(), "MLP", MLP_PARAMS),
        (HistGradientBoostingClassifier(), "HistGradientBoosting", HISTOGRAM_GBM_PARAMS),
    ]
    
    results = []
    
    for model, model_name, params in models:
        
        trained_model, accuracy, precision, f1,cv_scores, report, used_params, total_time = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, params)
        if trained_model is not None:
            save_model(trained_model, model_name, params)
            results.append({
                
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'F1' : f1,
                'CV Scores': cv_scores,
                'Total runtime' : total_time,
                'Hyperparameters': str(used_params),  # Convert the hyperparameters dictionary to string
                'Report': report
            })
        
    
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"../results/Results_{timestamp}.csv", index=False)
    print(f"Results saved to Results_{timestamp}.csv")
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    print(os.getcwd())
    input_file = "enhanced_dataset.csv"
    run_pipeline(input_file)


