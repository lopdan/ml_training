#!/usr/bin/env python
# coding: utf-8

from distutils.command.sdist import sdist
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def main():
    """
    Main function
    """
    initialize()
    
def initialize():
    """
    Initialize the training of multiple classifiers to compare the results
    """
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    X, y = X.to_numpy(), y.to_numpy()
    some_digit = X[0]
    
    # SGDClassifier
    y = y.astype(np.uint8) # convert to string to unsigned integer
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    y_train_5 = (y_train==5)
    y_test_5 = (y_test==5)

    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    sgd_clf.predict([some_digit])
    strattified_sampling(X_train, y_train_5, sgd_clf)
    threshold_90_precision, recall_90_precision, y_scores = get_performance_measurements(sgd_clf, X_train, y_train_5)
    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
    fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)] 
    plot_roc_curve(fpr, tpr)
    
    X_train = scale_inputs(sdg_clf, X_train, y_train_5)
    
    random_forest_classification(X_train, y_train_5, fpr_90)
    one_vs_one_classification(X_train, y_train_5)
     
def get_performance_measurements(sgd_clf, X_train, y_train_5):
    """
    Get performance measurements from the sgd_clf
    """
    y_train_predict = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    confusion_matrix(y_train_5, y_train_predict)
    
    print(precision_score(y_train_5, y_train_predict)) #Times it is correct
    print(recall_score(y_train_5, y_train_predict)) #TImes it is detected

    f1_score(y_train_5, y_train_predict)
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)] # 3370
    recall_90_precision = recalls[np.argmax(precisions >= 0.90)] # 0.43
    
    y_train_pred_90 = (y_scores >= threshold_90_precision)
    precision_score(y_train_5, y_train_pred_90)
    recall_score(y_train_5, y_train_pred_90)
    
    return threshold_90_precision, recall_90_precision, y_scores
    
def strattified_sampling(X_train, y_train_5, sgd_clf):
    """
    Split the dataset into train and test sets
    """
    skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = (y_train_5[train_index])
        X_test_fold = X_train[test_index]
        y_test_fold = (y_train_5[test_index])

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))
    
def plot_digit(data):
    """
    Plot a digit
    """
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

def plot_digits(instances, images_per_row=10, **options):
    """
    Plot a list of digit images
    """
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
    plt.show()
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_precision_recall_curve(y_true, y_score, **options):
    """
    Plot the precision-recall curve
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    plt.plot(recall, precision, **options)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    
def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, label: str = None):
    """
    Plot the ROC curve
    """
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate (Fall-Out)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.grid(True)
    plt.axis([0, 1, 0, 1])  
    
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Plot the precision-recall curve for varying thresholds.
    """
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()

def plot_feature_importances(model):
    """
    Plot the feature importances
    """
    n_features = model.coef_.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), model.feature_names_)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()
    
def random_forest_classification(X_train, y_train_5, fpr_90):
    """
    Random forest classification training and validation with performance scores
    """
    forest_clf = RandomForestClassifier(random_state=42)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1]
    
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
    recall_for_forest = tpr_forest[np.argmax(fpr_forest >= fpr_90)]
    
    roc_auc_score(y_train_5, y_scores_forest)
    y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
    print(precision_score(y_train_5, y_train_pred_forest))
    print(recall_score(y_train_5, y_train_pred_forest))
    
def one_vs_one_classification(X_train, y_train):
    ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
    ovo_clf.fit(X_train, y_train)
    ovo_clf.predict([0])
    
    some_digit_scores = ovo_clf.decision_function([0])
    np.argmax(some_digit_scores)
    
def scale_inputs(sgd_clf, X_train, y_train):
    """
    Scale the inputs using StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
    
    return X_train_scaled

def multilabel_classification(X_train, y_train):
    """
    Multilabel classification with KNeighborsClassifier (takes up to 20h)
    """
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd]

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)

    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
    f1_score(y_multilabel, y_train_knn_pred, average="macro")
    
def delete_noise(X_train, X_test):
    """
    Delete noise from the data image using PCA
    """
    noise = np.random.randint(0, 100, (len(X_train), 784))
    X_train_mod = X_train + noise
    noise = np.random.randint(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise
    y_train_mod = X_train
    y_test_mod = X_test
    
    
                        
