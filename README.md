Diabetes Risk Prediction App
A simple yet powerful web application built with Streamlit to predict the likelihood of diabetes using Machine Learning models (Random Forest & Gradient Boosting).

📌 Features
✅ User-friendly interface powered by Streamlit
✅ Predicts diabetes risk based on health metrics
✅ Utilizes Random Forest and Gradient Boosting for high accuracy
✅ Displays prediction results instantly

📊 Dataset
The app uses a dataset containing medical information like:

Glucose Level

Blood Pressure

BMI

Age
(Example: Pima Indians Diabetes dataset or similar)

⚙️ Tech Stack
Python 3.9+

Streamlit for frontend & app deployment

Pandas, NumPy for data handling

Scikit-learn for Machine Learning

Matplotlib for visualizations

🚀 Installation & Setup
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/ahsen-dev86/diabetes-prediction-app.git
cd diabetes-prediction-app
2. Create a virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Run the app locally
bash
Copy
Edit
streamlit run app.py
🌐 Deployment
The app is deployed on Streamlit Cloud.
👉 Live Demo: https://diabetes-risk-app.streamlit.app/

📈 Model Performance
Base accuracy: ~70%

After optimization: ~86% using GridSearchCV

🛠 Future Improvements
Add user authentication

Integrate cloud database for logging predictions

Enable API access for predictions

🤝 Contributing
Pull requests are welcome! For major changes, open an issue first to discuss what you’d like to change.

📜 License
MIT License
