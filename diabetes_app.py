# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ---------------- Load Saved Models and Encoders ----------------
# rf_model = joblib.load("rf_model.pkl")
# gb_model = joblib.load("gb_model.pkl")
# label_encoders = joblib.load("label_encoders.pkl")

# # Feature list
# features = ['age', 'gender', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'blood_glucose_level']

# # ---------------- Streamlit Page Config ----------------
# st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")
# st.title("🩺 Diabetes Risk Prediction App")
# st.write("Predict diabetes risk using **Machine Learning** models (Random Forest & Gradient Boosting).")

# # ---------------- Sidebar Inputs ----------------
# st.sidebar.header("Patient Information")
# age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
# gender = st.sidebar.selectbox("Gender", label_encoders['gender'].classes_)
# hypertension = st.sidebar.selectbox("Hypertension (High BP)", [0, 1])
# heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
# smoking_history = st.sidebar.selectbox("Smoking History", label_encoders['smoking_history'].classes_)
# bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
# blood_glucose_level = st.sidebar.number_input("Blood Glucose Level", min_value=50.0, max_value=400.0, value=120.0)

# threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.3)

# # Encode inputs
# gender_encoded = label_encoders['gender'].transform([gender])[0]
# smoking_encoded = label_encoders['smoking_history'].transform([smoking_history])[0]

# user_data = pd.DataFrame([[age, gender_encoded, hypertension, heart_disease,
#                            smoking_encoded, bmi, blood_glucose_level]], columns=features)

# # ---------------- Prediction Button ----------------
# if st.sidebar.button("🔍 Predict"):
#     # Predictions
#     rf_prob = rf_model.predict_proba(user_data)[:, 1][0]
#     gb_prob = gb_model.predict_proba(user_data)[:, 1][0]
#     rf_pred = int(rf_prob > threshold)
#     gb_pred = int(gb_prob > threshold)

#     # Risk level function
#     def risk_level(prob):
#         if prob >= 0.7: return "High Risk"
#         elif prob >= 0.4: return "Medium Risk"
#         else: return "Low Risk"

#     # Display results
#     st.subheader("Prediction Results")
#     st.success(f"**Random Forest** → Probability: `{rf_prob:.2f}`, Prediction: **{'Diabetic' if rf_pred else 'Not Diabetic'}**, Risk: `{risk_level(rf_prob)}`")
#     st.info(f"**Gradient Boosting** → Probability: `{gb_prob:.2f}`, Prediction: **{'Diabetic' if gb_pred else 'Not Diabetic'}**, Risk: `{risk_level(gb_prob)}`")

#     if abs(rf_prob - threshold) < 0.05 or abs(gb_prob - threshold) < 0.05:
#         st.warning("⚠️ Borderline case! Close to threshold.")

# # ---------------- Model Insights Section ----------------
# if st.button("📊 View Model Insights"):
#     st.subheader("Feature Importance & Model Performance")

#     # Random Forest Feature Importance
#     rf_importance = pd.Series(rf_model.best_estimator_.feature_importances_, index=features)
#     gb_importance = pd.Series(gb_model.best_estimator_.feature_importances_, index=features)

#     # Plot RF Feature Importance
#     fig_rf, ax_rf = plt.subplots(figsize=(6, 4))
#     sns.barplot(x=rf_importance.sort_values(), y=rf_importance.sort_values().index, palette="Blues_r", ax=ax_rf)
#     ax_rf.set_title("Random Forest Feature Importance")
#     st.pyplot(fig_rf)

#     # Plot GB Feature Importance
#     fig_gb, ax_gb = plt.subplots(figsize=(6, 4))
#     sns.barplot(x=gb_importance.sort_values(), y=gb_importance.sort_values().index, palette="Greens_r", ax=ax_gb)
#     ax_gb.set_title("Gradient Boosting Feature Importance")
#     st.pyplot(fig_gb)

#     # Accuracy Scores (we saved during training, but let's add placeholders for now)
#     st.write("✅ **Random Forest Accuracy (Threshold 0.3)**: ~70%")
#     st.write("✅ **Gradient Boosting Accuracy**: ~86%")

# # Footer
# st.markdown("---")
# st.markdown("Built with ❤️ using **Streamlit**, **Random Forest**, and **Gradient Boosting**")


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ---------------- Load Saved Models and Data ----------------
rf_model = joblib.load("rf_model.pkl")
gb_model = joblib.load("gb_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
metrics = joblib.load("model_metrics.pkl")  # contains rf_acc & gb_acc

# Feature list
features = ['age', 'gender', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'blood_glucose_level']

# ---------------- Streamlit Page Config ----------------
st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")
st.title("🩺 Diabetes Risk Prediction App")
st.write("Predict diabetes risk using **Machine Learning** models (Random Forest & Gradient Boosting).")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
gender = st.sidebar.selectbox("Gender", label_encoders['gender'].classes_)
hypertension = st.sidebar.selectbox("Hypertension (High BP)", [0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
smoking_history = st.sidebar.selectbox("Smoking History", label_encoders['smoking_history'].classes_)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
blood_glucose_level = st.sidebar.number_input("Blood Glucose Level", min_value=50.0, max_value=400.0, value=120.0)

threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.3)

# Encode inputs
gender_encoded = label_encoders['gender'].transform([gender])[0]
smoking_encoded = label_encoders['smoking_history'].transform([smoking_history])[0]

user_data = pd.DataFrame([[age, gender_encoded, hypertension, heart_disease,
                           smoking_encoded, bmi, blood_glucose_level]], columns=features)

# ---------------- Prediction Button ----------------
if st.sidebar.button("🔍 Predict"):
    # Predictions
    rf_prob = rf_model.predict_proba(user_data)[:, 1][0]
    gb_prob = gb_model.predict_proba(user_data)[:, 1][0]
    rf_pred = int(rf_prob > threshold)
    gb_pred = int(gb_prob > threshold)

    # Risk level function
    def risk_level(prob):
        if prob >= 0.7: return "High Risk"
        elif prob >= 0.4: return "Medium Risk"
        else: return "Low Risk"

    # Display results
    st.subheader("Prediction Results")
    st.success(f"**Random Forest** → Probability: `{rf_prob:.2f}`, Prediction: **{'Diabetic' if rf_pred else 'Not Diabetic'}**, Risk: `{risk_level(rf_prob)}`")
    st.info(f"**Gradient Boosting** → Probability: `{gb_prob:.2f}`, Prediction: **{'Diabetic' if gb_pred else 'Not Diabetic'}**, Risk: `{risk_level(gb_prob)}`")

    if abs(rf_prob - threshold) < 0.05 or abs(gb_prob - threshold) < 0.05:
        st.warning("⚠️ Borderline case! Close to threshold.")

# ---------------- Model Insights Section ----------------
if st.button("📊 View Model Insights"):
    st.subheader("Feature Importance & Model Performance")

    # Random Forest Feature Importance
    rf_importance = pd.Series(rf_model.best_estimator_.feature_importances_, index=features)
    gb_importance = pd.Series(gb_model.best_estimator_.feature_importances_, index=features)

    # Plot RF Feature Importance
    fig_rf, ax_rf = plt.subplots(figsize=(6, 4))
    sns.barplot(x=rf_importance.sort_values(), y=rf_importance.sort_values().index, palette="Blues_r", ax=ax_rf)
    ax_rf.set_title("Random Forest Feature Importance")
    st.pyplot(fig_rf)

    # Plot GB Feature Importance
    fig_gb, ax_gb = plt.subplots(figsize=(6, 4))
    sns.barplot(x=gb_importance.sort_values(), y=gb_importance.sort_values().index, palette="Greens_r", ax=ax_gb)
    ax_gb.set_title("Gradient Boosting Feature Importance")
    st.pyplot(fig_gb)

    # Display Accuracy
    st.write(f"✅ **Random Forest Accuracy (Threshold 0.3)**: {metrics['rf_acc']*100:.2f}%")
    st.write(f"✅ **Gradient Boosting Accuracy**: {metrics['gb_acc']*100:.2f}%")

    st.markdown("---")
    st.subheader("Confusion Matrices")
    rf_preds = (rf_model.predict_proba(X_test)[:, 1] > 0.3).astype(int)
    gb_preds = gb_model.predict(X_test)

    # RF Confusion Matrix
    cm_rf = confusion_matrix(y_test, rf_preds)
    fig_cm_rf, ax_cm_rf = plt.subplots()
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax_cm_rf)
    ax_cm_rf.set_title("Random Forest Confusion Matrix")
    ax_cm_rf.set_xlabel("Predicted")
    ax_cm_rf.set_ylabel("Actual")
    st.pyplot(fig_cm_rf)

    # GB Confusion Matrix
    cm_gb = confusion_matrix(y_test, gb_preds)
    fig_cm_gb, ax_cm_gb = plt.subplots()
    sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax_cm_gb)
    ax_cm_gb.set_title("Gradient Boosting Confusion Matrix")
    ax_cm_gb.set_xlabel("Predicted")
    ax_cm_gb.set_ylabel("Actual")
    st.pyplot(fig_cm_gb)

    st.markdown("---")
    st.subheader("ROC Curve")

    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    gb_probs = gb_model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_probs)
    auc_rf = auc(fpr_rf, tpr_rf)
    auc_gb = auc(fpr_gb, tpr_gb)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})", color="blue")
    ax_roc.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC = {auc_gb:.2f})", color="green")
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)

# Footer
st.markdown("---")
st.markdown("Built By: Muhammad Ahsan")
st.markdown("---")
st.markdown("Built with ❤️ using **Streamlit**, **Random Forest**, and **Gradient Boosting**")

