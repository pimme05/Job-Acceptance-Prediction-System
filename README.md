# 📊 Job Acceptance Prediction System

A machine learning–based web application that predicts whether a candidate is likely to be **Placed** or **Not Placed** based on academic background, skills, experience, and job market factors.

The project includes end-to-end steps: data preprocessing, model training, evaluation, explainability, and deployment as an interactive web application.

---

## 🚀 Live Demo

🔗 **Web App:**  
https://job-acceptance-prediction-system.streamlit.app

---

## 🎯 Problem Statement

Campus placement and job acceptance decisions depend on multiple factors such as academic performance, skills, experience, and interview performance.  
This project aims to build a **predictive system** that estimates a candidate’s placement probability using historical placement data.

---

## 🌍 Dataset Context

- **Education & Job Context:** 🇮🇳 India  
- Based on the **Indian education system** (SSC, HSC, Degree, MBA)
- Reflects a **campus placement environment**
- Dataset used for **educational and research purposes only**

---

## 🧾 Features Used

### 🎓 Academic Background
- Gender  
- SSC Percentage (10th)  
- SSC Board (Central / Others)  
- HSC Percentage (12th)  
- HSC Board (Central / Others)  
- HSC Subject (Science / Commerce / Arts)  
- Degree Percentage  
- Undergraduate Degree Type  
- MBA Specialisation  
- MBA Percentage  

### 💼 Experience & Skills
- Work Experience (Yes / No)  
- Years of Experience  
- Internship Completed  
- Skills Match Percentage  
- Number of Certifications  

### 🧪 Assessments
- Employability / Aptitude Test Percentage  
- Interview Score  

### 🏢 Job Market Context
- Company Tier (Startup / Mid-size / MNC)  
- Job Competition Level (Low / Medium / High)  

### 🎯 Target Variable
- **Status:** Placed / Not Placed  

---

## 🧠 Models Implemented

| Model | Purpose |
|----|----|
| Logistic Regression | Baseline model |
| Random Forest | Best balance of accuracy & explainability |
| CatBoost | Strong handling of categorical features |

**Final selected model for deployment:**  
✅ **Random Forest Classifier**

---

## 📈 Model Performance

- **Accuracy:** ~86%  
- Strong recall for *Placed* candidates  
- Robust handling of non-linear feature interactions  

---

## 🔍 Explainability

The application provides:
- **Top 10 feature importance** (Random Forest)
- Candidate-specific input summary
- Business-friendly explanations of predictions

### 🔑 Key Influencing Factors
- Academic performance (SSC, HSC, Degree, MBA)
- Skills match percentage
- Interview score
- Work experience
- Employability test score

---

## 🌐 Web Application

### Built With
- **Streamlit** – frontend & deployment  
- **scikit-learn** – machine learning  
- **joblib** – model serialization  
- **matplotlib** – visualization  

### UI Features
- Dropdowns for categorical inputs  
- Sliders for scores & percentages  
- Checkboxes for yes/no attributes  
- Placement probability bar  
- Feature importance visualization  

---

## 🛠 Tech Stack

- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, joblib, matplotlib  
- **Deployment:** Streamlit Community Cloud  
- **Training Environment:** Google Colab  

---


## ⚙️ Installation & Run Locally

```bash
git clone https://github.com/your-username/job-acceptance-prediction-system.git
cd job-acceptance-prediction-system
pip install -r requirements.txt
streamlit run app.py
