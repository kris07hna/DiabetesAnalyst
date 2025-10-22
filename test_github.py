"""
Test script to verify GitHub integration is working properly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.github_integration import GitHubModelUploader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_github_connection():
    """Test GitHub connection and repository access"""
    print("🔄 Testing GitHub Integration...")
    print("-" * 50)
    
    # Initialize uploader
    uploader = GitHubModelUploader()
    
    # Test connection
    print("1. Testing GitHub connection...")
    conn_success, conn_message = uploader.initialize_connection()
    print(f"   Result: {conn_message}")
    
    if not conn_success:
        print("❌ GitHub connection failed!")
        return False
    
    # Test repository access
    print("\n2. Testing repository access...")
    repo_success, repo_message = uploader.get_or_create_repository()
    print(f"   Result: {repo_message}")
    
    if not repo_success:
        print("❌ Repository access failed!")
        return False
    
    # Test metadata upload (small test)
    print("\n3. Testing metadata upload...")
    test_metadata = {
        'test': True,
        'timestamp': '2025-10-22',
        'message': 'GitHub integration test'
    }
    
    meta_success, meta_message = uploader.upload_model_metadata(test_metadata, 'test')
    print(f"   Result: {meta_message}")
    
    if meta_success:
        print("\n✅ GitHub integration is working perfectly!")
        return True
    else:
        print("❌ Metadata upload failed!")
        return False

if __name__ == "__main__":
    success = test_github_connection()
    if success:
        print("\n🎉 GitHub integration is ready for automatic model uploads!")
    else:
        print("\n⚠️  Please check your GitHub token and repository settings.")