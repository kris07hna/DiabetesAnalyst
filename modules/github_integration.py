"""
GitHub Integration Module for Diabetes Risk Analysis Platform
Handles automatic uploading of trained models to GitHub repository
"""

import os
import base64
import io
import streamlit as st
from github import Github
from datetime import datetime
import joblib
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GitHubModelUploader:
    """Handles uploading trained models and metadata to GitHub repository"""
    
    def __init__(self):
        self.github_token = self._get_github_token()
        self.repo_name = self._get_repo_name()
        self.github_client = None
        self.repository = None
        
    def _get_github_token(self):
        """Get GitHub token from environment or Streamlit secrets"""
        token = os.getenv('GITHUB_TOKEN')
        if not token and hasattr(st, 'secrets'):
            try:
                token = st.secrets['github']['token']
            except:
                pass
        return token
    
    def _get_repo_name(self):
        """Get repository name from environment or Streamlit secrets"""
        repo = os.getenv('GITHUB_REPO', 'diabetes-risk-models')
        if hasattr(st, 'secrets'):
            try:
                repo = st.secrets['github']['repo_name']
            except:
                pass
        return repo
    
    def _get_repo_owner(self):
        """Get repository owner from environment or Streamlit secrets"""
        owner = os.getenv('GITHUB_OWNER')
        if not owner and hasattr(st, 'secrets'):
            try:
                owner = st.secrets['github']['owner']
            except:
                pass
        return owner
    
    def initialize_connection(self):
        """Initialize GitHub connection"""
        if not self.github_token:
            return False, "GitHub token not found. Please configure GITHUB_TOKEN in environment variables or Streamlit secrets."
        
        try:
            self.github_client = Github(self.github_token)
            # Test connection
            user = self.github_client.get_user()
            return True, f"Successfully connected to GitHub as {user.login}"
        except Exception as e:
            return False, f"Failed to connect to GitHub: {str(e)}"
    
    def get_or_create_repository(self):
        """Get existing repository or create new one"""
        if not self.github_client:
            return False, "GitHub client not initialized"
        
        try:
            owner = self._get_repo_owner()
            if owner:
                # Try to get repository from specific owner
                full_repo_name = f"{owner}/{self.repo_name}"
                self.repository = self.github_client.get_repo(full_repo_name)
            else:
                # Try to get repository from authenticated user
                user = self.github_client.get_user()
                try:
                    self.repository = user.get_repo(self.repo_name)
                except:
                    # Create new repository if it doesn't exist
                    self.repository = user.create_repo(
                        name=self.repo_name,
                        description="Trained models for Diabetes Risk Analysis Platform",
                        private=False,
                        auto_init=True
                    )
            
            return True, f"Repository ready: {self.repository.full_name}"
        except Exception as e:
            return False, f"Failed to access/create repository: {str(e)}"
    
    def upload_model_file(self, model_path, model_name, algorithm_name):
        """Upload model file to GitHub repository"""
        if not self.repository:
            return False, "Repository not initialized"
        
        try:
            # Read model file
            with open(model_path, 'rb') as file:
                model_content = file.read()
            
            # Encode to base64
            model_b64 = base64.b64encode(model_content).decode('utf-8')
            
            # Create file path in repository
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            repo_path = f"models/{algorithm_name}/{model_name}_{timestamp}.joblib"
            
            # Upload to GitHub
            self.repository.create_file(
                path=repo_path,
                message=f"Upload {algorithm_name} model trained on {timestamp}",
                content=model_b64
            )
            
            return True, f"Model uploaded successfully to {repo_path}"
        except Exception as e:
            return False, f"Failed to upload model: {str(e)}"
    
    def upload_model_metadata(self, metadata, algorithm_name):
        """Upload model training metadata to GitHub"""
        if not self.repository:
            return False, "Repository not initialized"
        
        try:
            # Create metadata JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_json = json.dumps(metadata, indent=2, default=str)
            
            # Create file path
            repo_path = f"metadata/{algorithm_name}/{algorithm_name}_metadata_{timestamp}.json"
            
            # Upload metadata
            self.repository.create_file(
                path=repo_path,
                message=f"Upload {algorithm_name} model metadata - {timestamp}",
                content=metadata_json
            )
            
            return True, f"Metadata uploaded successfully to {repo_path}"
        except Exception as e:
            return False, f"Failed to upload metadata: {str(e)}"
    
    def create_model_release(self, tag_name, model_paths, release_notes):
        """Create a GitHub release with model files"""
        if not self.repository:
            return False, "Repository not initialized"
        
        try:
            # Create release
            release = self.repository.create_git_release(
                tag=tag_name,
                name=f"Model Release {tag_name}",
                message=release_notes
            )
            
            # Upload model files as release assets
            for model_path in model_paths:
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as file:
                        release.upload_asset(
                            path=model_path,
                            content_type="application/octet-stream"
                        )
            
            return True, f"Release created successfully: {release.html_url}"
        except Exception as e:
            return False, f"Failed to create release: {str(e)}"
    
    def upload_training_report(self, report_content, algorithm_name):
        """Upload training report to GitHub"""
        if not self.repository:
            return False, "Repository not initialized"
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            repo_path = f"reports/{algorithm_name}/{algorithm_name}_training_report_{timestamp}.md"
            
            self.repository.create_file(
                path=repo_path,
                message=f"Upload training report for {algorithm_name} - {timestamp}",
                content=report_content
            )
            
            return True, f"Training report uploaded to {repo_path}"
        except Exception as e:
            return False, f"Failed to upload training report: {str(e)}"

def create_training_report(model_name, algorithm_name, metrics, parameters, timestamp):
    """Create a markdown training report"""
    report = f"""# {algorithm_name} Model Training Report

## Model Information
- **Model Name**: {model_name}
- **Algorithm**: {algorithm_name}
- **Training Date**: {timestamp}
- **Platform**: Diabetes Risk Analysis Platform

## Model Performance Metrics
"""
    
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, dict):
            report += f"\n### {metric_name}\n"
            for sub_metric, sub_value in metric_value.items():
                report += f"- **{sub_metric}**: {sub_value:.4f}\n"
        else:
            report += f"- **{metric_name}**: {metric_value:.4f}\n"
    
    report += f"\n## Model Parameters\n"
    for param_name, param_value in parameters.items():
        report += f"- **{param_name}**: {param_value}\n"
    
    report += f"""
## Usage Instructions

This model can be loaded using joblib:

```python
import joblib
model = joblib.load('{model_name}')
predictions = model.predict(X_test)
```

## Model Validation
- Cross-validation performed with 5-fold CV
- Training data: BRFSS 2015 Dataset
- Features: 21 health indicators
- Target: Diabetes risk levels (0, 1, 2)

Generated automatically by Diabetes Risk Analysis Platform
"""
    
    return report

# Streamlit UI Components for GitHub Integration
def render_github_config_sidebar():
    """Render GitHub configuration in sidebar"""
    with st.sidebar.expander("🔗 GitHub Integration", expanded=False):
        st.write("**Configure GitHub Settings**")
        
        # GitHub Token Input
        github_token = st.text_input(
            "GitHub Token",
            type="password",
            help="Personal Access Token with repo permissions"
        )
        
        # Repository Settings
        repo_owner = st.text_input("Repository Owner", help="GitHub username or organization")
        repo_name = st.text_input("Repository Name", value="diabetes-risk-models")
        
        # Auto-upload Settings
        auto_upload = st.checkbox("Auto-upload models after training", value=True)
        create_releases = st.checkbox("Create GitHub releases for models", value=False)
        
        if st.button("Test GitHub Connection"):
            if github_token:
                # Set temporary environment variables
                os.environ['GITHUB_TOKEN'] = github_token
                os.environ['GITHUB_OWNER'] = repo_owner
                os.environ['GITHUB_REPO'] = repo_name
                
                uploader = GitHubModelUploader()
                success, message = uploader.initialize_connection()
                
                if success:
                    st.success(message)
                    # Test repository access
                    repo_success, repo_message = uploader.get_or_create_repository()
                    if repo_success:
                        st.success(repo_message)
                    else:
                        st.error(repo_message)
                else:
                    st.error(message)
            else:
                st.warning("Please enter GitHub token")
        
        return {
            'token': github_token,
            'owner': repo_owner,
            'repo_name': repo_name,
            'auto_upload': auto_upload,
            'create_releases': create_releases
        }

def upload_model_to_github(model_path, model_name, algorithm_name, metrics, parameters):
    """Upload trained model to GitHub with metadata"""
    uploader = GitHubModelUploader()
    
    # Initialize connection
    conn_success, conn_message = uploader.initialize_connection()
    if not conn_success:
        return False, conn_message
    
    # Get or create repository
    repo_success, repo_message = uploader.get_or_create_repository()
    if not repo_success:
        return False, repo_message
    
    # Upload model file
    model_success, model_message = uploader.upload_model_file(model_path, model_name, algorithm_name)
    if not model_success:
        return False, model_message
    
    # Prepare metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        'model_name': model_name,
        'algorithm': algorithm_name,
        'training_timestamp': timestamp,
        'metrics': metrics,
        'parameters': parameters,
        'platform': 'Diabetes Risk Analysis Platform',
        'dataset': 'BRFSS 2015'
    }
    
    # Upload metadata
    meta_success, meta_message = uploader.upload_model_metadata(metadata, algorithm_name)
    
    # Create training report
    report_content = create_training_report(model_name, algorithm_name, metrics, parameters, timestamp)
    report_success, report_message = uploader.upload_training_report(report_content, algorithm_name)
    
    return True, f"Model, metadata, and report uploaded successfully to GitHub!"