#!/usr/bin/env python3
"""
streamlit_app.py
Streamlit frontend for drug recommendation system - Deployed version
"""

import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, List, Any
import os

# Configuration - Update this URL after backend deployment
API_BASE_URL = os.getenv("API_BASE_URL", "https://wagon-bootcamp-462414.appspot.com")

# Page configuration
st.set_page_config(
    page_title="Drug Recommendation System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e6ffe6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #44ff44;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #888888;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"API connection error: {str(e)}")
        return False

def get_conditions() -> List[str]:
    """Get list of valid conditions from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/conditions", timeout=15)
        if response.status_code == 200:
            return response.json()["conditions"]
        return []
    except Exception as e:
        st.error(f"Failed to load conditions: {str(e)}")
        return []

def make_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction request to API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=data,
            timeout=30
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            error_detail = response.json().get("detail", "Unknown error") if response.text else "Unknown error"
            return {"success": False, "error": error_detail}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def main():
    # Header
    st.markdown('<h1 class="main-header">💊 Drug Recommendation System</h1>', unsafe_allow_html=True)
    
    # Show API URL info
    st.markdown(f"""
    <div class="info-box">
        <p><strong>Backend API:</strong> {API_BASE_URL}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API health
    with st.spinner("Connecting to backend API..."):
        api_healthy = check_api_health()
    
    if not api_healthy:
        st.markdown("""
        <div class="error-box">
            <h3>⚠️ API Connection Error</h3>
            <p>The backend API is not running or not accessible. Please ensure:</p>
            <ul>
                <li>The backend is deployed to GCP and running</li>
                <li>The API_BASE_URL is correctly configured</li>
                <li>The API endpoints are accessible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Single Prediction", "Batch Prediction", "API Info"])
    
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "API Info":
        api_info_page()

def single_prediction_page():
    """Single prediction interface"""
    st.header("🔍 Single Drug Prediction")
    
    # Get conditions from API
    with st.spinner("Loading medical conditions..."):
        conditions = get_conditions()
    
    if not conditions:
        st.error("Failed to load conditions from API")
        return
    
    # Create form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Age",
                min_value=0.0,
                max_value=120.0,
                value=35.0,
                step=1.0,
                help="Patient's age in years"
            )
            
            condition = st.selectbox(
                "Medical Condition",
                options=conditions,
                help="Select the medical condition"
            )
            
            ease_of_use = st.slider(
                "Ease of Use Rating",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
                help="How easy is the treatment to use? (1=Very Difficult, 5=Very Easy)"
            )
        
        with col2:
            effectiveness = st.slider(
                "Effectiveness Rating",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
                help="How effective is the treatment? (1=Not Effective, 5=Very Effective)"
            )
            
            satisfaction = st.slider(
                "Satisfaction Rating",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
                help="Overall satisfaction with treatment (1=Very Unsatisfied, 5=Very Satisfied)"
            )
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Drug", use_container_width=True)
        
        if submitted:
            # Prepare request data
            request_data = {
                "age": age,
                "condition": condition,
                "ease_of_use": ease_of_use,
                "effectiveness": effectiveness,
                "satisfaction": satisfaction
            }
            
            # Show loading spinner
            with st.spinner("Making prediction..."):
                result = make_prediction(request_data)
            
            # Display results
            if result["success"]:
                data = result["data"]
                
                # Main prediction
                st.markdown(f"""
                <div class="success-box">
                    <h3>🎯 Recommended Drug</h3>
                    <h2 style="color: #1f77b4;">{data['predicted_drug']}</h2>
                    <p><strong>Confidence:</strong> {data['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Top predictions
                st.subheader("📊 Top 5 Predictions")
                
                # Create DataFrame for better display
                top_preds_df = pd.DataFrame(data['top_predictions'])
                top_preds_df.index = range(1, len(top_preds_df) + 1)
                top_preds_df.columns = ['Drug', 'Probability', 'Confidence %']
                
                st.dataframe(
                    top_preds_df,
                    use_container_width=True,
                    column_config={
                        "Probability": st.column_config.ProgressColumn(
                            "Probability",
                            help="Prediction probability",
                            min_value=0,
                            max_value=1,
                        ),
                        "Confidence %": st.column_config.NumberColumn(
                            "Confidence %",
                            help="Confidence percentage",
                            format="%.2f%%"
                        )
                    }
                )
                
                # Show input summary
                with st.expander("📋 Input Summary"):
                    st.json(request_data)
                    
            else:
                st.markdown(f"""
                <div class="error-box">
                    <h3>❌ Prediction Failed</h3>
                    <p>{result['error']}</p>
                </div>
                """, unsafe_allow_html=True)

def batch_prediction_page():
    """Batch prediction interface"""
    st.header("📊 Batch Drug Prediction")
    
    st.info("Upload a CSV file with columns: age, condition, ease_of_use, effectiveness, satisfaction")
    
    # Sample CSV download
    sample_data = pd.DataFrame({
        'age': [25, 45, 65],
        'condition': ['Depression', 'High Blood Pressure', 'Diabetes'],
        'ease_of_use': [4.0, 3.5, 4.2],
        'effectiveness': [4.2, 3.8, 4.0],
        'satisfaction': [4.1, 3.9, 4.3]
    })
    
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_data.to_csv(index=False),
        file_name="sample_batch_data.csv",
        mime="text/csv"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with patient data for batch prediction"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Validate columns
            required_columns = ["age", "condition", "ease_of_use", "effectiveness", "satisfaction"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return
            
            # Batch prediction button
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                # Prepare requests
                requests_data = []
                for _, row in df.iterrows():
                    requests_data.append({
                        "age": float(row["age"]),
                        "condition": str(row["condition"]),
                        "ease_of_use": float(row["ease_of_use"]),
                        "effectiveness": float(row["effectiveness"]),
                        "satisfaction": float(row["satisfaction"])
                    })
                
                # Make batch request
                with st.spinner(f"Processing {len(requests_data)} predictions..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/batch_predict",
                            json=requests_data,
                            timeout=120
                        )
                        
                        if response.status_code == 200:
                            results = response.json()["results"]
                            
                            # Process results
                            predictions = []
                            errors = []
                            
                            for result in results:
                                if "error" in result:
                                    errors.append(result)
                                else:
                                    predictions.append(result)
                            
                            # Show results
                            if predictions:
                                st.success(f"✅ Successfully processed {len(predictions)} predictions")
                                
                                # Create results DataFrame
                                results_df = df.copy()
                                results_df["predicted_drug"] = ""
                                results_df["confidence"] = 0.0
                                
                                for pred in predictions:
                                    idx = pred["index"]
                                    results_df.loc[idx, "predicted_drug"] = pred["predicted_drug"]
                                    results_df.loc[idx, "confidence"] = pred["confidence"]
                                
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Download button
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name="drug_predictions.csv",
                                    mime="text/csv"
                                )
                            
                            if errors:
                                st.error(f"❌ {len(errors)} predictions failed")
                                st.json(errors)
                        
                        else:
                            st.error(f"Batch prediction failed: {response.text}")
                    
                    except Exception as e:
                        st.error(f"Error during batch prediction: {str(e)}")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

def api_info_page():
    """API information page"""
    st.header("ℹ️ API Information")
    
    # API status
    if check_api_health():
        st.markdown("""
        <div class="success-box">
            <h3>✅ API Status: Healthy</h3>
            <p>The backend API is running and ready to make predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="error-box">
            <h3>❌ API Status: Unavailable</h3>
            <p>The backend API is not accessible.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # API endpoints
    st.subheader("🔗 Available Endpoints")
    
    endpoints = [
        {"Method": "GET", "Endpoint": "/", "Description": "Root endpoint"},
        {"Method": "GET", "Endpoint": "/health", "Description": "Health check"},
        {"Method": "GET", "Endpoint": "/conditions", "Description": "Get valid conditions"},
        {"Method": "GET", "Endpoint": "/drugs", "Description": "Get possible drugs"},
        {"Method": "POST", "Endpoint": "/predict", "Description": "Single prediction"},
        {"Method": "POST", "Endpoint": "/batch_predict", "Description": "Batch prediction"},
    ]
    
    st.dataframe(pd.DataFrame(endpoints), use_container_width=True)
    
    # Show conditions and drugs if API is available
    if check_api_health():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏥 Available Conditions")
            try:
                conditions = get_conditions()
                st.write(f"Total conditions: {len(conditions)}")
                with st.expander("View all conditions"):
                    for condition in conditions:
                        st.write(f"• {condition}")
            except:
                st.error("Failed to load conditions")
        
        with col2:
            st.subheader("💊 Possible Drugs")
            try:
                response = requests.get(f"{API_BASE_URL}/drugs", timeout=15)
                if response.status_code == 200:
                    drugs = response.json()["drugs"]
                    st.write(f"Total drugs: {len(drugs)}")
                    with st.expander("View all drugs"):
                        for drug in drugs:
                            st.write(f"• {drug}")
            except:
                st.error("Failed to load drugs")

if __name__ == "__main__":
    main()

