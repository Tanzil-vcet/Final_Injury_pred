import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    RocCurveDisplay, 
    roc_curve, 
    auc, 
    classification_report
)
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib # Or pickle
import numpy as np
import shap

# --- Load your data ---
print("Loading data...")
df = pd.read_excel("data/balanced_sheet1.xlsx")

# Separate features and targets
X = df.drop(columns=["Injury Severity", "Injury Location"])
y_severity = df["Injury Severity"]
y_location = df["Injury Location"]

# Encode Injury Location using the saved encoder to match the trained model
location_encoder = joblib.load("results/best_models/location_encoder.pkl")
y_location_encoded = location_encoder.transform(y_location)

# Ensure all values are non-negative
if y_location_encoded.min() < 0:
    shift = abs(y_location_encoded.min())
    y_location_encoded += shift
    print(f"Shifted Injury Location values up by {shift} to make all classes non-negative.")

# Split datasets (same way as model training)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_severity, test_size=0.2, random_state=42)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X, y_location_encoded, test_size=0.2, random_state=42)

# Ensure numeric data
X_test_s = X_test_s.astype(float)
X_test_l = X_test_l.astype(float)

# --- Load your trained models ---
print("Loading models...")
model_severity = joblib.load("results/best_models/best_severity_model.pkl")
model_location = joblib.load("results/best_models/best_location_model.pkl")

# --- Get predictions ---
print("Making predictions for severity...")
y_pred_severity = model_severity.predict(X_test_s)
y_proba_severity = model_severity.predict_proba(X_test_s)

print("Making predictions for location...")
y_pred_location = model_location.predict(X_test_l)
y_proba_location = model_location.predict_proba(X_test_l)

# --- Define class names (crucial for plotting) ---
# Make sure these are in the same order as the model's .classes_ attribute
severity_classes = model_severity.classes_  # numeric or label values as trained
location_classes_numeric = model_location.classes_
try:
    human_readable_location_classes = location_encoder.inverse_transform(location_classes_numeric)
except Exception:
    human_readable_location_classes = [str(c) for c in location_classes_numeric]

print(f"Severity classes: {severity_classes}")
print(f"Location classes: {location_classes_numeric}")


# Now you can create visualizations using:
# - y_test_s, y_pred_severity, y_proba_severity for severity
# - y_test_l, y_pred_location, y_proba_location for location

# # 1. Calculate the Confusion Matrix
# cm_severity = confusion_matrix(y_test_s, y_pred_severity, labels=severity_classes)

# # 2. Plot as a heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     cm_severity, 
#     annot=True, 
#     fmt='d', 
#     cmap='Blues', 
#     xticklabels=severity_classes, 
#     yticklabels=severity_classes
# )
# plt.title('Confusion Matrix for Injury Severity (Hybrid4 Model)', fontsize=14)
# plt.ylabel('Actual Label', fontsize=12)
# plt.xlabel('Predicted Label', fontsize=12)
# plt.tight_layout()

# # 3. Save the figure
# plt.savefig("confusion_matrix_severity.png", dpi=300)
# plt.show()

# print("Generating Plot 2: Location Confusion Matrix...")

# # 1. Calculate the Confusion Matrix for Location
# cm_location = confusion_matrix(
#     y_test_l, 
#     y_pred_location, 
#     labels=location_classes
# )

# # 2. Plot as a heatmap
# plt.figure(figsize=(12, 10)) 
# sns.heatmap(
#     cm_location, 
#     annot=True, 
#     fmt='d', 
#     cmap='Greens', 
#     xticklabels=location_classes, 
#     yticklabels=location_classes
# )
# plt.title('Confusion Matrix for Injury Location (Hybrid RF+XGB Model)', fontsize=14)
# plt.ylabel('Actual Label', fontsize=12)
# plt.xlabel('Predicted Label', fontsize=12)
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()

# # 3. Save the figure
# plt.savefig("confusion_matrix_location.png", dpi=300, bbox_inches='tight')
# plt.savefig("confusion_matrix_location.pdf", format='pdf', bbox_inches='tight')
# plt.show()

# print("Plot 2 saved as confusion_matrix_location.png/pdf")

# # 1. Binarize the output labels
# y_test_severity_bin = label_binarize(y_test_s, classes=severity_classes)
# n_classes_severity = len(severity_classes)

# # 2. Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# for i in range(n_classes_severity):
#     fpr[i], tpr[i], _ = roc_curve(y_test_severity_bin[:, i], y_proba_severity[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # 3. Plot all ROC curves
# plt.figure(figsize=(10, 8))

# colors = ['blue', 'red', 'green'] # Add more if you have more classes
# for i, color in zip(range(n_classes_severity), colors):
#     plt.plot(
#         fpr[i], 
#         tpr[i], 
#         color=color, 
#         lw=2, 
#         label=f'ROC curve for class {severity_classes[i]} (AUC = {roc_auc[i]:.2f})'
#     )

# plt.plot([0, 1], [0, 1], 'k--', lw=2) # Plot the diagonal line
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate', fontsize=12)
# plt.ylabel('True Positive Rate', fontsize=12)
# plt.title('Multi-Class ROC Curve for Injury Severity (Hybrid4)', fontsize=14)
# plt.legend(loc="lower right")
# plt.tight_layout()

# # 4. Save the figure
# plt.savefig("roc_curve_severity.png", dpi=300)
# plt.show()

# # 1. Binarize the output labels
# y_test_location_bin = label_binarize(y_test_l, classes=location_classes)
# n_classes_location = len(location_classes)

# # 2. Compute ROC curve and ROC area for each class
# fpr_loc = dict()
# tpr_loc = dict()
# roc_auc_loc = dict()

# for i in range(n_classes_location):
#     fpr_loc[i], tpr_loc[i], _ = roc_curve(y_test_location_bin[:, i], y_proba_location[:, i])
#     roc_auc_loc[i] = auc(fpr_loc[i], tpr_loc[i])

# # 3. Plot all ROC curves
# plt.figure(figsize=(12, 10))
# # Use a colormap to get unique colors
# colors = plt.cm.get_cmap('tab20', n_classes_location)

# for i in range(n_classes_location):
#     plt.plot(
#         fpr_loc[i], 
#         tpr_loc[i], 
#         color=colors(i), 
#         lw=2, 
#         label=f'{location_classes[i]} (AUC = {roc_auc_loc[i]:.2f})'
#     )

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate', fontsize=12)
# plt.ylabel('True Positive Rate', fontsize=12)
# plt.title('Multi-Class ROC Curve for Injury Location (Hybrid RF+XGB)', fontsize=14)
# plt.legend(loc="lower right", fontsize='small') # 'small' to fit more labels
# plt.tight_layout()

# # 4. Save the figure
# plt.savefig("roc_curve_location.png", dpi=300)
# plt.savefig("roc_curve_location.pdf", format='pdf', bbox_inches='tight')
# plt.show()

# print("Visualization complete! All plots saved.")

# print("Generating Plot 5: Location F1-Score Bar Chart...")

# 1. Generate the classification report as a dictionary
# report_dict = classification_report(
#     y_test_l, 
#     y_pred_location, 
#     labels=location_classes_numeric, 
#     target_names=human_readable_location_classes,
#     output_dict=True
# )

# # 2. Convert to a DataFrame and extract F1-scores
# report_df = pd.DataFrame(report_dict).transpose()
# # Keep only the classes, remove averages
# report_df = report_df.loc[human_readable_location_classes] 
# report_df.reset_index(inplace=True)
# report_df.rename(columns={'index': 'Location'}, inplace=True)

# # 3. Plot the F1-scores
# plt.figure(figsize=(10, 8))
# sns.barplot(
#     data=report_df.sort_values('f1-score', ascending=False), 
#     x='f1-score', 
#     y='Location',
#     palette='viridis'
# )
# plt.title('F1-Score per Injury Location (Hybrid RF+XGB)', fontsize=14)
# plt.xlabel('F1-Score', fontsize=12)
# plt.ylabel('Anatomical Location', fontsize=12)
# plt.xlim(0, 1.0)
# plt.tight_layout()

# # 4. Save the figure
# plt.savefig("f1_scores_location.png", dpi=300, bbox_inches='tight')
# plt.savefig("f1_scores_location.pdf", format='pdf', bbox_inches='tight')
# plt.show()

# print("Plot 5 saved as f1_scores_location.png/pdf")

# print("Generating Plot 6: SHAP Summary Plot (this may take a few minutes)...")

# # 1. Create a summary background dataset from training data
# # We use shap.kmeans to create a weighted summary (e.g., 100 points)
# X_train_s_summary = shap.kmeans(X_train_s, 100)

# # 2. Find the index for the 'Severe' class (fallback to last class if not string-labeled)
# try:
#     severe_class_index = list(severity_classes).index('Severe')
# except ValueError:
#     severe_class_index = len(severity_classes) - 1
# print(f"Explaining class index: {severe_class_index} (classes: {list(severity_classes)})")

# # 3. Define a prediction function for the explainer
# def predict_fn_severe(x):
#     # Convert numpy array back to DataFrame with correct feature names
#     x_df = pd.DataFrame(x, columns=X.columns)
#     # Get probabilities from the model
#     probas = model_severity.predict_proba(x_df)
#     # Return only the probabilities for the 'Severe' class
#     return probas[:, severe_class_index]

# # 4. Initialize the KernelExplainer
# explainer_severity = shap.KernelExplainer(predict_fn_severe, X_train_s_summary)

# # 5. Calculate SHAP values
# # We'll use a subset of the test set (e.g., 50 samples) for speed.
# X_test_s_sample = X_test_s.iloc[0:50]
# print(f"Calculating SHAP values for {len(X_test_s_sample)} samples...")
# shap_values_severity = explainer_severity.shap_values(X_test_s_sample)

# # 6. Generate the summary plot
# print("Generating SHAP Summary Plot...")
# shap.summary_plot(
#     shap_values_severity, 
#     X_test_s_sample,
#     feature_names=X.columns,
#     show=False
# )
# plt.title('SHAP Summary Plot for "Severe" Injury Risk (Hybrid4)', fontsize=14)

# # 7. Save the figure
# plt.savefig("shap_summary_plot_severity.png", dpi=300, bbox_inches='tight')
# plt.savefig("shap_summary_plot_severity.pdf", format='pdf', bbox_inches='tight')
# plt.show()

# print("Plot 6 saved as shap_summary_plot_severity.png/pdf")

# print("Generating Plot 7: SHAP Force Plots...")

# # --- 7a. High-Risk Individual ---
# # Find an athlete in our sample with the highest overall SHAP magnitude
# per_sample_magnitude = np.sum(np.abs(shap_values_severity), axis=1)
# high_risk_index = int(np.argmax(per_sample_magnitude))

# print(f"Generating Force Plot for High-Risk Individual (Index: {high_risk_index})")
# # We must use shap.force_plot with matplotlib=True to save it
# force_plot_high = shap.force_plot(
#     explainer_severity.expected_value, 
#     shap_values_severity[high_risk_index, :], 
#     X_test_s_sample.iloc[high_risk_index, :],
#     feature_names=X.columns,
#     matplotlib=True, # Set to True for saving
#     show=False
# )
# force_plot_high.savefig("shap_force_plot_high_risk.png", dpi=300, bbox_inches='tight')
# plt.close() # Close the plot to avoid display issues
# print("Plot 7a (High Risk) saved.")

# # --- 7b. Low-Risk Individual ---
# # Find an athlete in our sample with the lowest overall SHAP magnitude
# low_risk_index = int(np.argmin(per_sample_magnitude))

# print(f"Generating Force Plot for Low-Risk Individual (Index: {low_risk_index})")
# force_plot_low = shap.force_plot(
#     explainer_severity.expected_value, 
#     shap_values_severity[low_risk_index, :], 
#     X_test_s_sample.iloc[low_risk_index, :],
#     feature_names=X.columns,
#     matplotlib=True,
#     show=False
# )
# force_plot_low.savefig("shap_force_plot_low_risk.png", dpi=300, bbox_inches='tight')
# plt.close()
# print("Plot 7b (Low Risk) saved.")

# To VIEW the plots interactively in your notebook, run this in a new cell:
# shap.initjs()
# shap.force_plot(explainer_severity.expected_value, shap_values_severity, X_test_s_sample)

print("Generating Plot 8: EDA Plots...")

# --- 8a. Histograms for Key Features ---
key_features_eda = ['Weekly Training Hours', 'Trunk Flexion (cm)', 'Stick Test (cm)', 'BMI']

plt.figure(figsize=(12, 10))
for i, feature in enumerate(key_features_eda, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[feature], kde=True, bins=20)
    plt.title(f'Distribution of {feature}', fontsize=12)
plt.tight_layout()
plt.savefig("eda_histograms.png", dpi=300, bbox_inches='tight')
plt.savefig("eda_histograms.pdf", format='pdf', bbox_inches='tight')
plt.show()
print("Plot 8a (Histograms) saved.")

# --- 8b. Correlation Heatmap ---
# Ensure you only correlate numerical features
numerical_features_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 10))
corr_matrix = numerical_features_df.corr()
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt='.2f', 
    linewidths=0.5
)
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.savefig("eda_correlation_heatmap.pdf", format='pdf', bbox_inches='tight')
plt.show()
print("Plot 8b (Correlation Heatmap) saved.")