from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import numpy as np
non_nan_indices_yTest = ~np.isnan(yTest)
yTest_cleaned = yTest[non_nan_indices_yTest]
yPred_cleaned = yPred[non_nan_indices_yTest]

accuracy = accuracy_score(yTest_cleaned, yPred_cleaned)
precision = precision_score(yTest_cleaned, yPred_cleaned)
recall = recall_score(yTest_cleaned, yPred_cleaned)
f1 = f1_score(yTest_cleaned, yPred_cleaned)
mcc = matthews_corrcoef(yTest_cleaned, yPred_cleaned)

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

conf_matrix = confusion_matrix(yTest_cleaned, yPred_cleaned)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
