import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Job Acceptance Prediction", page_icon="📊", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("job_acceptance_model.pkl")

model = load_model()

st.title("📊 Job Acceptance Prediction System")
st.write("Fill in candidate information and click **Predict** to estimate placement probability.")

# ---------- Input UI ----------
with st.form("candidate_form"):
    st.subheader("🎓 Academic Background")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        ssc_percentage = st.number_input("SSC Percentage (10th)", 0.0, 100.0, 70.0)
        ssc_board = st.selectbox("SSC Board", ["Central", "Others"])
        hsc_percentage = st.number_input("HSC Percentage (12th)", 0.0, 100.0, 70.0)
        hsc_board = st.selectbox("HSC Board", ["Central", "Others"])
    with col2:
        hsc_subject = st.selectbox("HSC Subject", ["Science", "Commerce", "Arts"])
        degree_percentage = st.number_input("Degree Percentage", 0.0, 100.0, 70.0)
        undergrad_degree = st.selectbox("Undergrad Degree", ["Sci&Tech", "Comm&Mgmt", "Others"])
        specialisation = st.selectbox("MBA Specialisation", ["Marketing & HR", "Marketing & Finance"])
        mba_percent = st.number_input("MBA Percentage", 0.0, 100.0, 70.0)

    st.subheader("💼 Experience & Skills")

    col3, col4 = st.columns(2)
    with col3:
        work_experience_chk = st.checkbox("Has prior work experience?")
        years_experience = st.slider("Years of Experience", 0, 15, 0)
        internship_chk = st.checkbox("Completed at least one internship?")
    with col4:
        skills_match_percent = st.slider("Skills Match Percent", 0, 100, 50)
        num_certifications = st.number_input("Number of Certifications", 0, 20, 0)

    st.subheader("🧪 Assessments & Interviews")

    col5, col6 = st.columns(2)
    with col5:
        emp_test_percentage = st.slider("Employability / Aptitude Test (%)", 0, 100, 50)
    with col6:
        interview_score = st.slider("Interview Score (0–100)", 0, 100, 50)

    st.subheader("🏢 Job Market Context")

    col7, col8 = st.columns(2)
    with col7:
        company_tier = st.selectbox("Company Tier", ["Startup", "Mid-size", "MNC"])
    with col8:
        job_competition_level = st.selectbox("Job Competition Level", ["Low", "Medium", "High"])

    submitted = st.form_submit_button("🔮 Predict")

# ---------- Build input row ----------
if submitted:
    work_experience = "Yes" if work_experience_chk else "No"
    internship_completed = 1 if internship_chk else 0

    input_df = pd.DataFrame([{
        "gender": gender,
        "ssc_percentage": ssc_percentage,
        "ssc_board": ssc_board,
        "hsc_percentage": hsc_percentage,
        "hsc_board": hsc_board,
        "hsc_subject": hsc_subject,
        "degree_percentage": degree_percentage,
        "undergrad_degree": undergrad_degree,
        "specialisation": specialisation,
        "mba_percent": mba_percent,
        "work_experience": work_experience,
        "years_experience": years_experience,
        "internship_completed": internship_completed,
        "skills_match_percent": skills_match_percent,
        "num_certifications": num_certifications,
        "emp_test_percentage": emp_test_percentage,
        "interview_score": interview_score,
        "company_tier": company_tier,
        "job_competition_level": job_competition_level
    }])

    # ---------- Predict ----------
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]  # probability of Placed (class 1)

    st.divider()
    st.subheader("✅ Prediction Result")

    if pred == 1:
        st.success(f"Status: **Placed** ✅")
    else:
        st.error(f"Status: **Not Placed** ❌")

    st.metric("Placement Probability", f"{proba*100:.2f}%")

    # ---------- Probability bar ----------
    st.write("### Probability Bar")
    st.progress(int(proba * 100))

    # ---------- Explanations ----------
    st.write("### Why this result? (Model explanation)")

    # 1) Show overall feature importance (top 10) if model is Pipeline with RF
    try:
        pre = model.named_steps["preprocessor"]
        clf = model.named_steps["classifier"]

        feature_names = pre.get_feature_names_out()
        importances = clf.feature_importances_

        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        top10 = imp_df.head(10).reset_index(drop=True)

        st.write("**Top 10 most important features (global importance):**")
        st.dataframe(top10)

        # Bar chart (matplotlib, no custom colors)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(top10["feature"][::-1], top10["importance"][::-1])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Top 10 Feature Importances")
        plt.tight_layout()
        st.pyplot(fig)

        # 2) Simple local explanation: highlight the candidate's values for a few key features
        st.write("**Candidate values for key features:**")
        key_cols = [
            "ssc_percentage", "degree_percentage", "mba_percent",
            "skills_match_percent", "interview_score", "emp_test_percentage",
            "years_experience", "num_certifications", "work_experience"
        ]
        st.dataframe(input_df[key_cols])

        st.info(
            "Tip: Higher academic scores, stronger skills match, better interview score, and prior experience "
            "typically increase placement probability."
        )

    except Exception as e:
        st.warning("Feature importance explanation is not available for this saved model format.")
        st.caption(f"Debug info: {e}")
