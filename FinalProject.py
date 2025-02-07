
"""
امیرمحمد شفیعی
حجت اله شاه پوری
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

data = pd.read_csv(r'C:\Users\kavosh\Desktop\diabetes_data.csv')

print("اولین چند ردیف داده‌ها:")
print(data.head())

print("\nاطلاعات اولیه درباره داده‌ها:")
print(data.info())

print("\nآمار توصیفی داده‌ها:")
print(data.describe())

print("\nداده‌های گم‌شده:")
print(data.isnull().sum())

data = data.drop(columns=['Date', 'Time'], errors='ignore')

data.fillna(data.mean(), inplace=True)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(data[['Code', 'Value']])

y = (data['Code'] % 2 == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

pca = PCA(n_components=2)
pipeline = make_pipeline(StandardScaler(), pca)

X_train_pca = pipeline.fit_transform(X_train)
X_test_pca = pipeline.transform(X_test)

print("\nتعداد ویژگی‌های اصلی پس از PCA: ", X_train_pca.shape[1])
print("نسبت واریانس هر یک از ویژگی‌ها پس از PCA:", pca.explained_variance_ratio_)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, y_train)
y_pred_knn = knn.predict(X_test_pca)

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_pca, y_train)
y_pred_dt = dt_model.predict(X_test_pca)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_pca, y_train)
y_pred_rf = rf_model.predict(X_test_pca)

gb_model = GradientBoostingClassifier(learning_rate=0.01, random_state=42)
gb_model.fit(X_train_pca, y_train)
y_pred_gb = gb_model.predict(X_test_pca)

xgb_model = xgb.XGBClassifier(learning_rate=0.05, n_estimators=100, random_state=42)
xgb_model.fit(X_train_pca, y_train)
y_pred_xgb = xgb_model.predict(X_test_pca)


def evaluate_model(y_test, y_pred, model_name):
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    print(f"\nارزیابی مدل {model_name}:")
    print(f"Accuracy: {accuracy}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return accuracy

accuracies = {}
accuracies["KNN"] = evaluate_model(y_test, y_pred_knn, "KNN")
accuracies["Decision Tree"] = evaluate_model(y_test, y_pred_dt, "Decision Tree")
accuracies["Random Forest"] = evaluate_model(y_test, y_pred_rf, "Random Forest")
accuracies["Gradient Boosting"] = evaluate_model(y_test, y_pred_gb, "Gradient Boosting")
accuracies["XGBoost"] = evaluate_model(y_test, y_pred_xgb, "XGBoost")

print("\nدقت مدل‌ها:")
for model_name, accuracy in accuracies.items():
    print(f"{model_name}: {accuracy}")

save_path = r'C:\Users\kavosh\Desktop\output_graphs\\'

if not os.path.exists(save_path):
    os.makedirs(save_path)

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Class 0', 'Class 1'])
    plt.yticks(tick_marks, ['Class 0', 'Class 1'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{save_path}confusion_matrix_{model_name}.png')
    plt.close()

def plot_roc_curve(model_name, y_test, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_path}roc_curve_{model_name}.png')
    plt.close()

models = {
    "KNN": knn,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model
}

for model_name, model in models.items():
    if model_name == "XGBoost":
        y_pred_prob = model.predict_proba(X_test_pca)[:, 1]
    else:
        y_pred_prob = model.predict_proba(X_test_pca)[:, 1]

    y_pred = model.predict(X_test_pca)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name)

    plot_roc_curve(model_name, y_test, y_pred_prob)

print(f"\nنمودارها در مسیر {save_path} ذخیره شدند.")
