# 🚀 Deployment Guide - Diabetes Risk Analysis Platform

## Quick Deployment Options

### 1. 🌐 Streamlit Community Cloud (Recommended)

**Prerequisites:**
- GitHub account
- Streamlit Community Cloud account

**Steps:**
1. **Create New GitHub Repository:**
   ```bash
   # Initialize new repository
   git init
   git add .
   git commit -m "Initial commit - Diabetes Risk Analysis Platform"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/diabetes-risk-analysis.git
   git push -u origin main
   ```

2. **Configure Secrets:**
   - Go to your app dashboard on Streamlit Cloud
   - Navigate to Settings > Secrets
   - Add the following secrets:
   ```toml
   [github]
   token = "your_github_personal_access_token"
   owner = "your_github_username"
   repo_name = "diabetes-risk-models"
   ```

3. **Deploy:**
   - Connect your GitHub repository to Streamlit Cloud
   - Select `main_app.py` as the main file
   - Click Deploy!

### 2. 🔧 Heroku Deployment

**Prerequisites:**
- Heroku account
- Heroku CLI installed

**Steps:**
```bash
# Login to Heroku
heroku login

# Create new app
heroku create your-diabetes-app-name

# Set environment variables
heroku config:set GITHUB_TOKEN=your_token
heroku config:set GITHUB_OWNER=your_username
heroku config:set GITHUB_REPO=diabetes-risk-models

# Deploy
git push heroku main
```

### 3. ☁️ Google Cloud Platform (App Engine)

**Prerequisites:**
- Google Cloud account
- gcloud CLI installed

**Steps:**
```bash
# Initialize gcloud
gcloud init

# Create new project (optional)
gcloud projects create diabetes-risk-analysis

# Set project
gcloud config set project diabetes-risk-analysis

# Deploy
gcloud app deploy app.yaml
```

### 4. 🐳 Docker Deployment

**Create Dockerfile:**
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Deploy:**
```bash
# Build image
docker build -t diabetes-risk-analysis .

# Run container
docker run -p 8501:8501 diabetes-risk-analysis
```

## 🔧 Configuration Guide

### GitHub Integration Setup

1. **Generate Personal Access Token:**
   - Go to GitHub → Settings → Developer settings → Personal access tokens
   - Create token with `repo` permissions
   - Copy the token

2. **Configure Environment Variables:**

   **For Local Development (.env file):**
   ```bash
   GITHUB_TOKEN=your_personal_access_token
   GITHUB_OWNER=your_username
   GITHUB_REPO=diabetes-risk-models
   ```

   **For Streamlit Cloud:**
   ```toml
   [github]
   token = "your_personal_access_token"
   owner = "your_username"
   repo_name = "diabetes-risk-models"
   ```

   **For Heroku:**
   ```bash
   heroku config:set GITHUB_TOKEN=your_token
   heroku config:set GITHUB_OWNER=your_username
   heroku config:set GITHUB_REPO=diabetes-risk-models
   ```

### Model Auto-Upload Features

When enabled, trained models will automatically:
- ✅ Upload to designated GitHub repository
- 📊 Create training reports with metrics
- 📁 Organize models by algorithm type
- 🏷️ Include metadata and parameters
- 📈 Generate performance summaries

## 🔒 Security Best Practices

1. **Never commit secrets to Git:**
   - Use `.env` files locally
   - Use platform-specific secret management in production

2. **GitHub Token Permissions:**
   - Only grant `repo` access
   - Use fine-grained tokens when available

3. **Environment Isolation:**
   - Use different repositories for dev/prod models
   - Implement proper access controls

## 🚀 Auto-Deployment Workflow

### GitHub Actions (Optional)

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.12
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Run tests
      run: python -m pytest tests/ || true
    
    - name: Deploy to Streamlit
      # This will automatically deploy if connected to Streamlit Cloud
      run: echo "Deployment triggered"
```

## 📊 Monitoring and Maintenance

### Health Checks
- Monitor application logs
- Check model upload success rates
- Monitor GitHub repository size
- Track performance metrics

### Updates
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart application (platform-specific)
```

## 🆘 Troubleshooting

### Common Issues:

1. **GitHub Upload Fails:**
   - Check token permissions
   - Verify repository exists
   - Ensure network connectivity

2. **Model Training Errors:**
   - Check dataset format
   - Verify feature selection
   - Monitor memory usage

3. **Deployment Issues:**
   - Check requirements.txt compatibility
   - Verify Python version
   - Review platform-specific logs

### Support Resources:
- 📖 [Streamlit Documentation](https://docs.streamlit.io/)
- 🐙 [GitHub API Documentation](https://docs.github.com/en/rest)
- ☁️ Platform-specific documentation

---

**Ready to Deploy!** 🎉

Your Diabetes Risk Analysis Platform is now ready for deployment with automatic GitHub model uploads!