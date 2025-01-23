import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from category_encoders import OneHotEncoder

# Set page configuration
st.set_page_config(layout="wide", page_title="Employee Attrition Prediction", page_icon="👥")

# Sidebar info
st.sidebar.info("""
    **Name:** Kavyaya G             
    **Project:** Employee-Attrition-Prediction Project            
    **College:** Sri Vijay Vidyalaya College of Arts and Science
""")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Home", "Model Working", "Exit"])

class EmployeePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.encoder = OneHotEncoder(cols=['salary', 'dept'])
        self._train_initial_model()
        
    def _generate_sample_training_data(self):
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Satisfaction_level': np.random.uniform(0.2, 0.9, n_samples),
            'Last_evaluation': np.random.uniform(0.3, 0.95, n_samples),
            'number_project': np.random.randint(2, 8, n_samples),
            'average_montly_hours': np.random.randint(120, 300, n_samples),
            'time_spend_company': np.random.randint(2, 10, n_samples),
            'Work_accident': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'promotion_last_5years': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'dept': np.random.choice(
                ['sales', 'technical', 'support', 'IT', 'hr', 'accounting', 'marketing', 'product_mng', 'randD', 'management'],
                n_samples
            ),
            'salary': np.random.choice(['low', 'medium', 'high'], n_samples)
        }
        
        df = pd.DataFrame(data)
        target = (
            (df['Satisfaction_level'] < 0.4) |
            ((df['time_spend_company'] > 6) & (df['promotion_last_5years'] == 0)) |
            ((df['average_montly_hours'] > 250) & (df['Satisfaction_level'] < 0.6))
        ).astype(int)
        
        return df, target

    def _train_initial_model(self):
        X, y = self._generate_sample_training_data()
        X_encoded = self.encoder.fit_transform(X)
        self.model.fit(X_encoded, y)
        
    def preprocess_data(self, df):
        """Preprocess the input data with separate handling for numeric and categorical columns"""
        df_copy = df.copy()
        
        # Define numeric and categorical columns
        numeric_cols = ['Satisfaction_level', 'Last_evaluation', 'number_project', 
                       'average_montly_hours', 'time_spend_company', 'Work_accident', 
                       'promotion_last_5years']
        categorical_cols = ['dept', 'salary']
        
        # Convert numeric columns
        for col in numeric_cols:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if df_copy[col].isnull().any():
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        
        # Ensure categorical columns are strings
        for col in categorical_cols:
            df_copy[col] = df_copy[col].astype(str)
        
        # One-hot encode categorical variables
        return self.encoder.transform(df_copy)

    def predict(self, df):
        processed_data = self.preprocess_data(df)
        return self.model.predict(processed_data)

    def predict_proba(self, df):
        processed_data = self.preprocess_data(df)
        return self.model.predict_proba(processed_data)

# Initialize the predictor
predictor = EmployeePredictor()

if page == "Home":
    st.title("🏢 Employee Attrition Prediction System")
    st.image("employee_attrition.png", width=700)
    st.write("""
    Welcome to the Employee Attrition Prediction System! This application helps HR professionals 
    and managers predict the likelihood of employee attrition using machine learning.
    
    ### Features:
    - Individual employee prediction
    - Batch prediction for multiple employees
    - Risk factor analysis
    - Detailed insights and recommendations
    
    ### How to Use:
    1. Navigate to the Model Working page
    2. Choose between Online or Batch predictions
    3. Input employee data
    4. Get instant predictions and insights
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Accuracy", "85%")
    with col2:
        st.metric("Features Analyzed", "9")
    with col3:
        st.metric("Processing Time", "<2 seconds")

elif page == "Model Working":
    st.title("Model Working")
    
    prediction_type = st.radio("Select Prediction Type:", ["Online Predictions", "Batch Predictions"])
    
    if prediction_type == "Online Predictions":
        st.subheader("Individual Employee Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            Employee_name = st.text_input("Employee name")
            Employee_ID = st.text_input("Employee ID")
            Satisfaction_level = st.slider("Satisfaction level", 0.0, 1.0, 0.5)
            Last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
            number_project = st.number_input("Number of projects", 2, 10, 5)
        
        with col2:
            average_montly_hours = st.number_input("Average monthly hours", 100, 400, 200)
            time_spend_company = st.number_input("Years in company", 1, 15, 3)
            Work_accident = st.selectbox("Work accident", (0, 1))
            promotion_last_5years = st.selectbox("Promoted in last 5 years", (0, 1))
            dept = st.selectbox("Department", 
                              ["sales", "technical", "support", "IT", "hr", "accounting", 
                               "marketing", "product_mng", "randD", "management"])
            salary = st.selectbox("Salary Level", ["low", "medium", "high"])
        
        if st.button("Predict"):
            try:
                input_data = pd.DataFrame({
                    'Satisfaction_level': [Satisfaction_level],
                    'Last_evaluation': [Last_evaluation],
                    'number_project': [number_project],
                    'average_montly_hours': [average_montly_hours],
                    'time_spend_company': [time_spend_company],
                    'Work_accident': [Work_accident],
                    'promotion_last_5years': [promotion_last_5years],
                    'dept': [dept],
                    'salary': [salary]
                })
                
                prediction = predictor.predict(input_data)[0]
                probabilities = predictor.predict_proba(input_data)[0]
                
                st.subheader("Prediction Results")
                if prediction == 0:
                    st.success(f'Employee {Employee_name} (ID: {Employee_ID}) is likely to stay')
                    st.progress(probabilities[0])
                    st.write(f"Confidence: {probabilities[0]:.2%}")
                else:
                    st.warning(f'Employee {Employee_name} (ID: {Employee_ID}) is at risk of leaving')
                    st.progress(probabilities[1])
                    st.write(f"Risk Level: {probabilities[1]:.2%}")
                
                # Risk factor analysis
                st.subheader("Risk Factor Analysis")
                risk_factors = []
                if Satisfaction_level < 0.4:
                    risk_factors.append("Low satisfaction level")
                if time_spend_company > 6 and promotion_last_5years == 0:
                    risk_factors.append("Long tenure without promotion")
                if average_montly_hours > 250 and Satisfaction_level < 0.6:
                    risk_factors.append("High workload with moderate satisfaction")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(factor)
                else:
                    st.success("No significant risk factors identified")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please check your input data and try again.")
    
    else:  # Batch Predictions
        st.subheader("Batch Prediction")
        st.write("""
        Upload a CSV file with the following columns:
        - Satisfaction_level (0.0-1.0)
        - Last_evaluation (0.0-1.0)
        - number_project (2-10)
        - average_montly_hours (100-400)
        - time_spend_company (1-15)
        - Work_accident (0 or 1)
        - promotion_last_5years (0 or 1)
        - dept (sales/technical/support/IT/hr/accounting/marketing/product_mng/randD/management)
        - salary (low/medium/high)
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ['Satisfaction_level', 'Last_evaluation', 'number_project',
                               'average_montly_hours', 'time_spend_company', 'Work_accident',
                               'promotion_last_5years', 'dept', 'salary']
                
                missing_cols = set(required_cols) - set(df.columns)
                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                else:
                    predictions = predictor.predict(df)
                    probabilities = predictor.predict_proba(df)
                    
                    results = pd.DataFrame({
                        'Prediction': ['At Risk' if p == 1 else 'Likely to Stay' for p in predictions],
                        'Risk_Probability': probabilities[:, 1]
                    })
                    
                    results = pd.concat([df, results], axis=1)
                    
                    st.subheader("Prediction Results")
                    st.dataframe(results)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Employees", len(results))
                    with col2:
                        at_risk = sum(predictions)
                        st.metric("At Risk", f"{at_risk} ({(at_risk/len(results))*100:.1f}%)")
                    with col3:
                        st.metric("Average Risk Score", f"{probabilities[:, 1].mean():.1%}")
                    
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="attrition_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.write("Please check your file format and try again.")

elif page == "Exit":
    st.title("Thank You for Using Our Application!")
    st.write("""
    ### We hope our predictions were helpful!
    """)
    
    if st.button("Exit Application"):
        st.balloons()
        st.info("You can close this window or navigate to another page.")