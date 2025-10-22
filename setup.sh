#!/bin/bash
# Quick setup script for DiabetesAnalyst platform

echo "🏥 Setting up DiabetesAnalyst Platform..."

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p saved_models uploads exports data

echo "🔧 Setting up environment..."
# Copy environment template if .env doesn't exist
if [ ! -f .env ]; then
    cp .env.template .env
    echo "⚠️  Please edit .env file with your GitHub token"
fi

# Copy Streamlit secrets template if secrets don't exist
if [ ! -f .streamlit/secrets.toml ]; then
    cp .streamlit/secrets.toml.template .streamlit/secrets.toml
    echo "⚠️  Please edit .streamlit/secrets.toml with your secrets"
fi

echo "✅ Setup complete!"
echo ""
echo "🚀 To run the application:"
echo "   streamlit run main_app.py"
echo ""
echo "📋 Next steps:"
echo "   1. Edit .env file with your GitHub Personal Access Token"
echo "   2. Edit .streamlit/secrets.toml with your configuration"
echo "   3. Run the application and train models"
echo "   4. Models will be automatically uploaded to GitHub!"
echo ""
echo "🌐 For deployment options, see DEPLOYMENT_GUIDE.md"