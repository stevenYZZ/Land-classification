import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import os
import pandas as pd
import csv



def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_pred, y_test):
    classification = classification_report(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    # Test_Loss =  score[0]*100
    # Test_accuracy = score[1]*100

    return classification, confusion, np.array([oa, aa, kappa] + list(each_acc)) * 100


def save_report(classification, confusion, metrics, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save classification report
    with open(os.path.join(save_path, "classification_report.txt"), "w") as file:
        file.write(classification)

    # Save confusion matrix
    with open(os.path.join(save_path, "confusion_matrix.csv"), "w", newline='') as file:
        writer = csv.writer(file)
        
        # Add column numbers (predicted classes)
        writer.writerow([""] + [f"{col_num}" for col_num in range(confusion.shape[1])])
        
        # Add row numbers (actual classes) and matrix values
        for row_num, row in enumerate(confusion):
            writer.writerow([f"{row_num}"] + [f"{value}" for value in row])

    # Save metrics
    with open(os.path.join(save_path, "metrics.csv"), "w", newline='') as file:
        writer = csv.writer(file)
        for metric_name, metric_value in zip(["OA", "AA", "Kappa"] + [f"Class {i} Accuracy" for i in range(len(metrics) - 3)], metrics):
            writer.writerow([metric_name, f"{metric_value}"])


def read_report(save_path):
    classification_report_path = os.path.join(save_path, "classification_report.txt")
    confusion_matrix_path = os.path.join(save_path, "confusion_matrix.csv")
    metrics_path = os.path.join(save_path, "metrics.csv")

    with open(classification_report_path, "r") as file:
        classification_report = file.read()

    confusion_matrix = pd.read_csv(confusion_matrix_path, index_col=0)
    metrics = pd.read_csv(metrics_path, header=None, index_col=0)

    report_dict = {
        'classification_report': classification_report,
        'confusion_matrix': confusion_matrix,
        'metrics': metrics
    }

    return report_dict

def visualize_report(report_dict):
    print("Classification Report:\n")
    print(report_dict['classification_report'])

    print("\nConfusion Matrix:")
    print(report_dict['confusion_matrix'])

    print("\nMetrics:")
    print(report_dict['metrics'])