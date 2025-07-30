# Drug Recommendation System - Frontend

This is the Streamlit frontend for the Drug Recommendation System.

## Features

- **Single Prediction**: Make individual drug recommendations based on patient data
- **Batch Prediction**: Upload CSV files for bulk predictions
- **API Information**: View available conditions, drugs, and API status

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Update the API_BASE_URL in `streamlit_app.py` to point to your deployed backend

3. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Deployment to Streamlit Cloud

1. Push this code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository and branch
5. Set the main file path to `streamlit_app.py`
6. Deploy!

## Environment Variables

- `API_BASE_URL`: The URL of your deployed backend API (defaults to GCP App Engine URL)

## CSV Format for Batch Prediction

The CSV file should contain the following columns:
- `age`: Patient age (0-120)
- `condition`: Medical condition (must be one of the valid conditions)
- `ease_of_use`: Ease of use rating (1-5)
- `effectiveness`: Effectiveness rating (1-5)
- `satisfaction`: Satisfaction rating (1-5)

