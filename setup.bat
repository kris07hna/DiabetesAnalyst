@echo off
REM Quick setup script for DiabetesAnalyst platform on Windows

echo 🏥 Setting up DiabetesAnalyst Platform...

REM Install dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating directories...
if not exist "saved_models" mkdir saved_models
if not exist "uploads" mkdir uploads
if not exist "exports" mkdir exports
if not exist "data" mkdir data

echo 🔧 Setting up environment...
REM Copy environment template if .env doesn't exist
if not exist ".env" (
    copy .env.template .env
    echo ⚠️  Please edit .env file with your GitHub token
)

REM Copy Streamlit secrets template if secrets don't exist
if not exist ".streamlit\secrets.toml" (
    copy .streamlit\secrets.toml.template .streamlit\secrets.toml
    echo ⚠️  Please edit .streamlit\secrets.toml with your secrets
)

echo ✅ Setup complete!
echo.
echo 🚀 To run the application:
echo    streamlit run main_app.py
echo.
echo 📋 Next steps:
echo    1. Edit .env file with your GitHub Personal Access Token
echo    2. Edit .streamlit\secrets.toml with your configuration
echo    3. Run the application and train models
echo    4. Models will be automatically uploaded to GitHub!
echo.
echo 🌐 For deployment options, see DEPLOYMENT_GUIDE.md

pause