import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

# Path to the test results
test_results_path = 'runs/1000_emb1_dlm0_run/test_results.txt'

# Initialize lists to store true labels and predictions
true_labels = []
predictions = []

# Read the test results
with open(test_results_path, 'r') as f:
    next(f)  # Skip the header line
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Assuming the format: label predict score (separated by whitespace)
        parts = line.split()
        true_label = int(parts[0])
        predicted_label = int(parts[1])
        true_labels.append(true_label)
        predictions.append(predicted_label)

# Map labels from -1 and 1 to 0 and 1
true_labels = [0 if label == -1 else 1 for label in true_labels]
predictions = [0 if label == -1 else 1 for label in predictions]

# Convert lists to numpy arrays
true_labels = np.array(true_labels)
predictions = np.array(predictions)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

# Compute metrics
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tpr  # Same as TPR
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print the results
print(f"True Positive Rate (TPR/Recall): {tpr:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Print a classification report
print("\nClassification Report:")
print(classification_report(true_labels, predictions, digits=4))
