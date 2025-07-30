# Streamlit Cloud Deployment Instructions

## Prerequisites

1. A GitHub account
2. The backend deployed to GCP and the URL available

## Step 1: Prepare the Repository

1. Create a new GitHub repository (e.g., `drug-recommendation-frontend`)
2. Upload all files from the `frontend` folder to the repository:
   - `streamlit_app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `README.md`

## Step 2: Update Backend URL

Before deploying, update the `API_BASE_URL` in `streamlit_app.py`:

```python
# Replace with your actual GCP backend URL
API_BASE_URL = "https://your-actual-backend-url.appspot.com"
```

## Step 3: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account if not already connected
4. Select your repository (`drug-recommendation-frontend`)
5. Choose the main branch
6. Set the main file path to `streamlit_app.py`
7. Click "Deploy!"

## Step 4: Configure Environment Variables (Optional)

If you want to use environment variables instead of hardcoding the API URL:

1. In the Streamlit Cloud dashboard, go to your app settings
2. Add an environment variable:
   - Key: `API_BASE_URL`
   - Value: `https://your-backend-url.appspot.com`

## Step 5: Test the Deployment

Once deployed, test all features:
1. Single prediction functionality
2. Batch prediction with CSV upload
3. API information page
4. Error handling when backend is unavailable

## Troubleshooting

- **App won't start**: Check the logs in Streamlit Cloud dashboard
- **Backend connection issues**: Verify the API_BASE_URL is correct
- **CORS errors**: Ensure the backend has CORS properly configured
- **Timeout errors**: The backend might be cold-starting; try again after a few seconds

## Alternative: Local Testing

To test locally before deployment:

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

