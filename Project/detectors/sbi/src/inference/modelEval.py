import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix, accuracy_score, classification_report

## MODEL EVALUATION CODE
class ModelEvaluator:
    '''
    Class: ModelEvaluator

    Description:
    The `ModelEvaluator` class is designed to facilitate the evaluation of machine learning models, particularly those used for binary classification tasks. It provides methods to execute model evaluation, calculate performance metrics such as ROC-AUC, Precision-Recall, and Confusion Matrix, and generate visualizations to assess the model's performance. The class enables easy access to key evaluation results and the option to save visualizations and reports to an output folder.

    Methods:
    - `__init__(self, model, dataloader, output_folder="output")`: Initializes the evaluator with a trained model, a data loader for evaluation, and an optional output folder for saving evaluation results.

    - `execute_model_eval(self)`: Evaluates the model on the provided dataset and generates labels and predictions. This method sets internal variables for labels and predictions.

    - `calculate_roc(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]`: Computes the Receiver Operating Characteristic (ROC) curve, calculates the Area Under the Curve (AUC), and identifies the Equal Error Rate (EER) point. Saves the ROC curve as an image and returns relevant metrics.

    - `calculate_pr(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]`: Computes the Precision-Recall (PR) curve, calculates the AUC, and identifies the F1 score's maximum point. Saves the PR curve as an image and returns relevant metrics.

    - `calculate_confusion_matrix(self) -> Tuple[np.ndarray, float, str]`: Computes the confusion matrix, accuracy, and classification report. Saves the confusion matrix as an image and returns relevant metrics.

    - `calculate_classification_report(self) -> str`: Generates and saves the classification report as a text file. Returns the classification report as a string.

    Attributes:
    - `model`: The machine learning model to be evaluated.

    - `dataloader`: The data loader used to fetch samples for evaluation.

    - `output_folder`: The folder where evaluation results, visualizations, and reports are saved.

    - `all_labels`: An array storing the true labels from the evaluation dataset.

    - `all_preds`: An array storing the model's predictions on the evaluation dataset.

    '''

    def __init__(self, model, dataloader, output_folder="output"):
        # Initialize the model, dataloader, and output folder
        self.modelEval = modelEval
        self.dataloader = dataloader
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        # Initialize the labels and predictions
        self.all_labels = []
        self.all_preds = []


    def execute_model_eval(self):
        self.model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.dataloader):
                data = data.to(device)

                recon_batch, mu, logvar = self.model(data)

                # Assuming binary classification
                predictions = (recon_batch > 0.5).float()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())

        self.all_labels = np.array(all_labels)
        self.all_preds = np.array(all_preds)

    def calculate_roc(self):
        fpr, tpr, thresholds = roc_curve(self.all_labels, self.all_preds)
        roc_auc = roc_auc_score(self.all_labels, self.all_preds)

        # Calculate EER
        eer = 1.0
        for i in range(len(fpr)):
            if fpr[i] + tpr[i] >= 1:
                eer = fpr[i]
                break

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
        plt.scatter(eer, 1 - eer, marker='o', color='red', label='EER point ({:.4f}, {:.4f})'.format(eer, 1 - eer))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        roc_curve_path = os.path.join(self.output_folder, 'roc_curve.png')
        plt.savefig(roc_curve_path)
        plt.close()

        return fpr, tpr, thresholds, roc_auc

    def calculate_pr(self):
        precision, recall, thresholds_pr = precision_recall_curve(self.all_labels, self.all_preds)
        pr_auc = auc(recall, precision)

        # Calculate F1 at threshold
        f1_scores = [2 * (p * r) / (p + r + 1e-10) for p, r in zip(precision, recall)]
        f1_max_idx = np.argmax(f1_scores)
        f1_max_threshold = thresholds_pr[f1_max_idx]

        # Plot Precision-Recall curve
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label='PR curve (AUC = {:.2f})'.format(pr_auc))
        plt.scatter(recall[f1_max_idx], precision[f1_max_idx], marker='o', color='red', label='F1 point ({:.4f}, {:.4f})'.format(recall[f1_max_idx], precision[f1_max_idx]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall (PR) Curve')
        plt.legend(loc="lower left")
        pr_curve_path = os.path.join(self.output_folder, 'pr_curve.png')
        plt.savefig(pr_curve_path)
        plt.close()

        return precision, recall, thresholds_pr, pr_auc

    def calculate_confusion_matrix(self):
        confusion = confusion_matrix(self.all_labels, (self.all_preds > 0.5).astype(int))
        accuracy = accuracy_score(self.all_labels, (self.all_preds > 0.5).astype(int))
        classification_rep = classification_report(self.all_labels, (self.all_preds > 0.5).astype(int))

        # Save confusion matrix as an image
        plt.figure()
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.output_folder, 'confusion_matrix.png'))
        plt.close()

        return confusion, accuracy, classification_rep
        
    def calculate_classification_report(self):
        classification_rep = classification_report(self.all_labels, (self.all_preds > 0.5).astype(int))

        # Save classification report to a text file
        report_file_path = os.path.join(self.output_folder, 'classification_report.txt')
        with open(report_file_path, 'w') as report_file:
            report_file.write("Classification Report:\n")
            report_file.write(classification_rep + '\n')

        return classification_rep