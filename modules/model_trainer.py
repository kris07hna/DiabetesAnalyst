"""
🧠 UNIFIED MODEL TRAINING MODULE - PROFESSIONAL PLATFORM
Streamlined ML training with robust UI/UX, unique identifiers, and permanent model storage
Features: 4 High-Accuracy Models, Pre-trained Model Dashboard, No Duplicate Keys
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import time
from datetime import datetime
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Advanced ML libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from modules.config import AppConfig
from modules.ui_components import UIComponents
from modules.github_integration import upload_model_to_github, render_github_config_sidebar


class ModelTrainer:
    """Unified Professional Model Training System"""
    
    def __init__(self):
        self.config = AppConfig()
        self.ui = UIComponents()
        self.models_dir = self.config.MODELS_DIR
        self.models_dir.mkdir(exist_ok=True)
        
        # 4 Professional high-accuracy models
        self.models = {
            "Random_Forest": {"name": "🌲 Random Forest", "class": RandomForestClassifier, "color": "#2ecc71"},
            "Logistic_Regression": {"name": "📊 Logistic Regression", "class": LogisticRegression, "color": "#3498db"},
            "XGBoost": {"name": "🚀 XGBoost", "class": XGBClassifier if XGBOOST_AVAILABLE else None, "color": "#e74c3c"},
            "LightGBM": {"name": "💡 LightGBM", "class": LGBMClassifier if LIGHTGBM_AVAILABLE else None, "color": "#f39c12"},
        }
        
        # Filter available models
        self.models = {k: v for k, v in self.models.items() if v["class"] is not None}
        
        # Setup permanent storage
        self.saved_models_dir = Path("saved_models")
        self.saved_models_dir.mkdir(exist_ok=True)
        self.load_saved_models()
    
    def load_saved_models(self):
        """Load all saved models from disk on startup"""
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        
        for model_file in self.saved_models_dir.glob("*.joblib"):
            try:
                # Safe model loading with version compatibility warnings suppressed
                import warnings
                with warnings.catch_warnings():
                    # Suppress scikit-learn version warnings
                    warnings.filterwarnings("ignore", message=".*version.*")
                    warnings.filterwarnings("ignore", category=UserWarning)
                    model_info = joblib.load(model_file)
                
                model_name = model_file.stem
                if model_name not in st.session_state.trained_models:
                    st.session_state.trained_models[model_name] = model_info
                    print(f"✅ Loaded: {model_name}")
            except Exception as e:
                print(f"⚠️ Failed to load {model_file.name}: {str(e)[:100]}...")
    
    def save_model_permanently(self, model_name, model_info):
        """Save model to permanent storage and optionally upload to GitHub"""
        try:
            # Save to saved_models directory
            model_path = self.saved_models_dir / f"{model_name}.joblib"
            joblib.dump(model_info, model_path)
            
            # Also save to session state
            if 'trained_models' not in st.session_state:
                st.session_state.trained_models = {}
            st.session_state.trained_models[model_name] = model_info
            
            # GitHub Upload (if configured)
            self.handle_github_upload(model_path, model_name, model_info)
            
            return True
        except Exception as e:
            st.error(f"Failed to save model: {e}")
            return False
    
    def handle_github_upload(self, model_path, model_name, model_info):
        """Handle GitHub upload if configured"""
        try:
            # Check if GitHub upload is enabled in session state
            github_config = st.session_state.get('github_config', {})
            
            if github_config.get('auto_upload', False) and github_config.get('token'):
                # Set environment variables for this upload
                import os
                os.environ['GITHUB_TOKEN'] = github_config['token']
                if github_config.get('owner'):
                    os.environ['GITHUB_OWNER'] = github_config['owner']
                if github_config.get('repo_name'):
                    os.environ['GITHUB_REPO'] = github_config['repo_name']
                
                # Extract algorithm name from model name
                algorithm_name = model_name.split('_')[0] if '_' in model_name else model_name
                
                # Extract metrics and parameters from model_info
                metrics = model_info.get('metrics', {})
                parameters = model_info.get('parameters', {})
                
                # Attempt upload
                success, message = upload_model_to_github(
                    str(model_path), model_name, algorithm_name, metrics, parameters
                )
                
                if success:
                    st.success(f"✅ Model uploaded to GitHub: {message}")
                else:
                    st.warning(f"⚠️ GitHub upload failed: {message}")
        except Exception as e:
            st.warning(f"⚠️ GitHub upload encountered an error: {str(e)}")
    
    def render(self):
        """Render unified training interface"""
        self.ui.create_header(
            "🧠 Professional ML Training Hub",
            "Train, manage, and deploy high-accuracy models with streamlined interface"
        )
        
        # GitHub Integration Sidebar
        github_config = render_github_config_sidebar()
        if 'github_config' not in st.session_state:
            st.session_state.github_config = {}
        st.session_state.github_config.update(github_config)
        
        # Display pre-trained models dashboard at top
        self.render_pretrained_models_dashboard()
        
        st.markdown("---")
        
        # Check for dataset
        if not self.check_dataset_availability():
            self.render_quick_upload_interface()
            return
        
        # Dataset selector
        self.render_dataset_selector()
        
        # Unified training interface
        st.markdown("### 🎯 Model Training Center")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            training_mode = st.selectbox(
                "Select Training Mode:",
                ["🚀 Quick Train All Models", "🎯 Train Single Model", "📊 Manage Models"],
                key="main_training_mode_selector"
            )
        
        with col2:
            st.metric("Available Models", len(self.models))
        
        st.markdown("---")
        
        if training_mode == "🚀 Quick Train All Models":
            self.render_quick_train_all()
        elif training_mode == "🎯 Train Single Model":
            self.render_single_model_training()
        else:
            self.render_model_management()
    
    def render_pretrained_models_dashboard(self):
        """Display pre-trained models in dashboard"""
        st.markdown("### 📊 Pre-Trained Models Dashboard")
        
        trained_models = st.session_state.get('trained_models', {})
        
        if not trained_models:
            st.info("🎯 No pre-trained models yet. Train your first model below!")
            return
        
        # Display model cards
        cols = st.columns(min(4, len(trained_models)))
        
        for idx, (model_name, model_info) in enumerate(list(trained_models.items())[:4]):
            with cols[idx % 4]:
                metrics = model_info.get('metrics', {})
                accuracy = metrics.get('accuracy', 0)
                model_type = model_info.get('model_type', 'Unknown')
                
                # Color based on accuracy
                if accuracy >= 0.90:
                    color = "#2ecc71"
                    status = "Excellent"
                elif accuracy >= 0.80:
                    color = "#3498db"
                    status = "Good"
                elif accuracy >= 0.70:
                    color = "#f39c12"
                    status = "Fair"
                else:
                    color = "#e74c3c"
                    status = "Needs Improvement"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}22 0%, {color}11 100%); 
                            padding: 1.2rem; border-radius: 12px; border-left: 4px solid {color}; margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: {color};">🎯 {model_type}</h4>
                    <p style="margin: 0.5rem 0; color: #bbb; font-size: 0.85rem;">{model_name[:30]}...</p>
                    <h2 style="margin: 0.5rem 0; color: {color};">{accuracy:.1%}</h2>
                    <p style="margin: 0; color: #888; font-size: 0.8rem;">{status}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Show total count
        if len(trained_models) > 4:
            st.info(f"📚 **{len(trained_models)} total models** available. View all in Model Management.")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        all_accuracies = [m.get('metrics', {}).get('accuracy', 0) for m in trained_models.values()]
        
        with col1:
            st.metric("Total Models", len(trained_models))
        with col2:
            st.metric("Avg Accuracy", f"{np.mean(all_accuracies):.1%}" if all_accuracies else "N/A")
        with col3:
            st.metric("Best Accuracy", f"{max(all_accuracies):.1%}" if all_accuracies else "N/A")
        with col4:
            excellent_count = sum(1 for acc in all_accuracies if acc >= 0.90)
            st.metric("Excellent Models", excellent_count)
    
    def check_dataset_availability(self):
        """Check if dataset is available"""
        return st.session_state.get('current_dataset') is not None
    
    def render_quick_upload_interface(self):
        """Quick dataset upload interface"""
        st.markdown("### 📤 Upload Dataset to Begin")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose your dataset file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV or Excel files",
                key="quick_upload_file"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.current_dataset = df
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.dataset_timestamp = timestamp
                    
                    # Auto-detect target
                    self.auto_detect_target(df)
                    
                    st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        with col2:
            st.markdown("**Quick Options:**")
            if st.button("📊 Load Sample Data", width="stretch", key="load_sample_btn"):
                self.load_sample_data()
    
    def auto_detect_target(self, df):
        """Auto-detect target column"""
        target_keywords = ['diabetes', 'target', 'class', 'label', 'outcome']
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                st.session_state.suggested_target = col
                return
        
        # Find column with few unique values
        for col in df.columns:
            if df[col].nunique() <= 10 and df[col].dtype in ['int64', 'float64']:
                st.session_state.suggested_target = col
                return
    
    def load_sample_data(self):
        """Load sample diabetes dataset"""
        try:
            sample_file = self.config.BASE_DIR / "diabetes_012_health_indicators_BRFSS2015.csv"
            if sample_file.exists():
                df = pd.read_csv(sample_file)
                st.session_state.current_dataset = df
                st.session_state.dataset_name = "Diabetes_Sample"
                st.session_state.dataset_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.suggested_target = "Diabetes_012"
                st.success("✅ Sample data loaded!")
                st.rerun()
            else:
                st.error("Sample dataset not found.")
        except Exception as e:
            st.error(f"Error: {e}")
    
    def render_dataset_selector(self):
        """Render dataset selector"""
        df = st.session_state.current_dataset
        dataset_name = st.session_state.get('dataset_name', 'Current Dataset')
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**📊 Dataset:** {dataset_name}")
        with col2:
            st.markdown(f"**📏 Shape:** {df.shape[0]} × {df.shape[1]}")
        with col3:
            st.markdown(f"**💾 Size:** {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        st.markdown("---")
    
    def render_quick_train_all(self):
        """Quick train all 4 models"""
        st.markdown("### 🚀 Quick Train All Models")
        st.markdown("Train all 4 professional models with optimized hyperparameters.")
        
        df = st.session_state.current_dataset
        
        # Configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_options = list(df.columns)
            default_idx = 0
            suggested_target = st.session_state.get('suggested_target')
            if suggested_target and suggested_target in target_options:
                default_idx = target_options.index(suggested_target)
            
            target_column = st.selectbox(
                "Target Variable:",
                target_options,
                index=default_idx,
                key="quick_train_target"
            )
        
        with col2:
            test_size = st.slider("Test Size:", 0.1, 0.4, 0.2, 0.05, key="quick_train_test_size")
        
        with col3:
            use_cv = st.checkbox("Cross-Validation", value=True, key="quick_train_cv")
        
        # Feature selection
        st.markdown("**Select Features:**")
        all_features = [col for col in df.columns if col != target_column]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selection_mode = st.radio(
                "Feature Selection:",
                ["All Features", "Select Manually"],
                horizontal=True,
                key="quick_train_feature_mode"
            )
        
        with col2:
            st.metric("Available Features", len(all_features))
        
        if selection_mode == "All Features":
            selected_features = all_features
        else:
            selected_features = st.multiselect(
                "Choose features:",
                all_features,
                default=all_features[:min(10, len(all_features))],
                key="quick_train_features"
            )
        
        if not selected_features:
            st.warning("⚠️ Please select at least one feature.")
            return
        
        st.markdown("---")
        
        # Training button
        if st.button("🚀 Train All 4 Models", type="primary", width="stretch", key="btn_train_all_models"):
            self.execute_quick_train_all(df, selected_features, target_column, test_size, use_cv)
    
    def execute_quick_train_all(self, df, features, target, test_size, use_cv):
        """Execute training for all models"""
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        st.markdown(f"### 🎯 Training Session: `{session_id}`")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        try:
            # Prepare data
            X = df[features].copy()
            y = df[target].copy()
            
            # Handle missing values and categorical features
            X = self.preprocess_data(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Train each model
            total_models = len(self.models)
            
            for idx, (model_key, model_config) in enumerate(self.models.items()):
                status_text.markdown(f"**Training {model_config['name']}** ({idx + 1}/{total_models})...")
                
                try:
                    # Create model with optimized parameters
                    model = self.create_optimized_model(model_key, model_config['class'])
                    
                    # Train
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Cross-validation
                    cv_score = None
                    if use_cv:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                        cv_score = cv_scores.mean()
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                    
                    # Metrics
                    metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                    
                    # Save model
                    model_name = f"{model_key}_{session_id}"
                    model_info = {
                        'model': model,
                        'model_type': model_config['name'],
                        'model_key': model_key,
                        'features': features,
                        'target': target,
                        'metrics': metrics,
                        'training_time': training_time,
                        'cv_score': cv_score,
                        'created_at': datetime.now().isoformat(),
                        'session_id': session_id,
                        'test_size': test_size
                    }
                    
                    self.save_model_permanently(model_name, model_info)
                    
                    results.append({
                        'model_name': model_config['name'],
                        'unique_name': model_name,
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1_score'],
                        'cv_score': cv_score,
                        'training_time': training_time,
                        'color': model_config['color']
                    })
                    
                except Exception as e:
                    st.warning(f"⚠️ {model_config['name']} failed: {e}")
                
                progress_bar.progress((idx + 1) / total_models)
            
            status_text.markdown("**✅ Training Complete!**")
            
            # Display results
            self.display_training_results(results)
            
        except Exception as e:
            st.error(f"Training failed: {e}")
    
    def create_optimized_model(self, model_key, model_class):
        """Create model with optimized hyperparameters"""
        
        if model_key == "Random_Forest":
            return model_class(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        elif model_key == "Logistic_Regression":
            return model_class(
                max_iter=1000,
                random_state=42,
                C=1.0,
                solver='lbfgs'
            )
        
        elif model_key == "XGBoost":
            return model_class(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        
        elif model_key == "LightGBM":
            return model_class(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        
        return model_class(random_state=42)
    
    def preprocess_data(self, X):
        """Preprocess data for training"""
        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].median())
        
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        return X
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
        }
        
        # Matthews correlation coefficient
        try:
            metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        except:
            metrics['matthews_corrcoef'] = None
        
        # ROC AUC
        if y_pred_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def display_training_results(self, results):
        """Display training results"""
        st.markdown("### 🏆 Training Results")
        
        # Sort by accuracy
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Winner
        winner = results[0]
        st.success(f"🥇 **Best Model:** {winner['model_name']} - **{winner['accuracy']:.1%}** Accuracy")
        
        # Metrics cards
        cols = st.columns(4)
        
        for idx, result in enumerate(results):
            with cols[idx % 4]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {result['color']}22 0%, {result['color']}11 100%);
                            padding: 1rem; border-radius: 10px; border-left: 4px solid {result['color']};">
                    <h4 style="margin: 0; color: {result['color']};">{result['model_name']}</h4>
                    <h2 style="margin: 0.5rem 0; color: {result['color']};">{result['accuracy']:.1%}</h2>
                    <p style="margin: 0; font-size: 0.85rem; color: #888;">
                        Precision: {result['precision']:.3f}<br>
                        Recall: {result['recall']:.3f}<br>
                        F1: {result['f1_score']:.3f}<br>
                        Time: {result['training_time']:.2f}s
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed table
        st.markdown("### 📊 Detailed Metrics")
        
        results_df = pd.DataFrame([{
            'Model': r['model_name'],
            'Unique Name': r['unique_name'],
            'Accuracy': f"{r['accuracy']:.3f}",
            'Precision': f"{r['precision']:.3f}",
            'Recall': f"{r['recall']:.3f}",
            'F1-Score': f"{r['f1_score']:.3f}",
            'CV Score': f"{r['cv_score']:.3f}" if r['cv_score'] else 'N/A',
            'Time (s)': f"{r['training_time']:.2f}"
        } for r in results])
        
        st.dataframe(results_df, width="stretch")
        
        # Comparison chart
        if len(results) > 1:
            fig = go.Figure()
            
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=[r['model_name'] for r in results],
                    y=[r[metric] for r in results],
                    text=[f"{r[metric]:.3f}" for r in results],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="📊 Model Performance Comparison",
                xaxis_title="Models",
                yaxis_title="Score",
                barmode='group',
                height=500,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, width="stretch")
    
    def render_single_model_training(self):
        """Train a single model with custom parameters"""
        st.markdown("### 🎯 Train Single Model")
        
        df = st.session_state.current_dataset
        
        # Model selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            model_key = st.selectbox(
                "Select Model:",
                list(self.models.keys()),
                format_func=lambda x: self.models[x]['name'],
                key="single_model_selector"
            )
        
        with col2:
            st.markdown(f"**Color:** <span style='color: {self.models[model_key]['color']}'>●</span> {self.models[model_key]['name']}", unsafe_allow_html=True)
        
        # Configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_options = list(df.columns)
            default_idx = 0
            suggested_target = st.session_state.get('suggested_target')
            if suggested_target and suggested_target in target_options:
                default_idx = target_options.index(suggested_target)
            
            target_column = st.selectbox(
                "Target Variable:",
                target_options,
                index=default_idx,
                key="single_train_target"
            )
        
        with col2:
            test_size = st.slider("Test Size:", 0.1, 0.4, 0.2, 0.05, key="single_train_test_size")
        
        with col3:
            use_cv = st.checkbox("Cross-Validation", value=True, key="single_train_cv")
        
        # Features
        all_features = [col for col in df.columns if col != target_column]
        selected_features = st.multiselect(
            "Select Features:",
            all_features,
            default=all_features,
            key="single_train_features"
        )
        
        if not selected_features:
            st.warning("⚠️ Please select at least one feature.")
            return
        
        # Hyperparameters
        with st.expander("⚙️ Hyperparameters (Optional)"):
            params = self.render_hyperparameters(model_key)
        
        st.markdown("---")
        
        # Train button
        if st.button("🚀 Train Model", type="primary", width="stretch", key=f"btn_train_single_{model_key}"):
            self.execute_single_model_training(df, selected_features, target_column, model_key, test_size, use_cv, params if 'params' in locals() else {})
    
    def render_hyperparameters(self, model_key):
        """Render hyperparameter controls"""
        params = {}
        
        if model_key == "Random_Forest":
            col1, col2 = st.columns(2)
            with col1:
                params['n_estimators'] = st.slider("Trees:", 50, 500, 200, 50, key=f"hp_rf_trees")
                params['max_depth'] = st.slider("Max Depth:", 5, 30, 15, 5, key=f"hp_rf_depth")
            with col2:
                params['min_samples_split'] = st.slider("Min Samples Split:", 2, 20, 5, key=f"hp_rf_split")
                params['min_samples_leaf'] = st.slider("Min Samples Leaf:", 1, 10, 2, key=f"hp_rf_leaf")
        
        elif model_key == "Logistic_Regression":
            col1, col2 = st.columns(2)
            with col1:
                params['C'] = st.slider("C (Regularization):", 0.1, 10.0, 1.0, 0.1, key=f"hp_lr_c")
                params['max_iter'] = st.slider("Max Iterations:", 100, 2000, 1000, 100, key=f"hp_lr_iter")
            with col2:
                params['solver'] = st.selectbox("Solver:", ['lbfgs', 'liblinear', 'saga'], key=f"hp_lr_solver")
        
        elif model_key == "XGBoost":
            col1, col2 = st.columns(2)
            with col1:
                params['n_estimators'] = st.slider("Estimators:", 50, 500, 200, 50, key=f"hp_xgb_est")
                params['max_depth'] = st.slider("Max Depth:", 3, 15, 6, key=f"hp_xgb_depth")
            with col2:
                params['learning_rate'] = st.slider("Learning Rate:", 0.01, 0.3, 0.1, 0.01, key=f"hp_xgb_lr")
        
        elif model_key == "LightGBM":
            col1, col2 = st.columns(2)
            with col1:
                params['n_estimators'] = st.slider("Estimators:", 50, 500, 200, 50, key=f"hp_lgb_est")
                params['max_depth'] = st.slider("Max Depth:", 3, 15, 6, key=f"hp_lgb_depth")
            with col2:
                params['learning_rate'] = st.slider("Learning Rate:", 0.01, 0.3, 0.1, 0.01, key=f"hp_lgb_lr")
        
        return params
    
    def execute_single_model_training(self, df, features, target, model_key, test_size, use_cv, params):
        """Execute single model training"""
        
        model_config = self.models[model_key]
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with st.spinner(f"Training {model_config['name']}..."):
            try:
                # Prepare data
                X = df[features].copy()
                y = df[target].copy()
                X = self.preprocess_data(X)
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Create model
                if params:
                    params['random_state'] = 42
                    model = model_config['class'](**params)
                else:
                    model = self.create_optimized_model(model_key, model_config['class'])
                
                # Train
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # CV
                cv_score = None
                if use_cv:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    cv_score = cv_scores.mean()
                
                # Predict
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Metrics
                metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Save
                model_name = f"{model_key}_{session_id}"
                model_info = {
                    'model': model,
                    'model_type': model_config['name'],
                    'model_key': model_key,
                    'features': features,
                    'target': target,
                    'metrics': metrics,
                    'training_time': training_time,
                    'cv_score': cv_score,
                    'created_at': datetime.now().isoformat(),
                    'session_id': session_id,
                    'parameters': params,
                    'test_size': test_size
                }
                
                self.save_model_permanently(model_name, model_info)
                
                # Display
                st.success(f"✅ Model trained successfully: `{model_name}`")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                
                if cv_score:
                    st.info(f"🔄 Cross-Validation Score: **{cv_score:.3f}**")
                
                st.metric("⏱️ Training Time", f"{training_time:.2f}s")
                
            except Exception as e:
                st.error(f"Training failed: {e}")
    
    def render_model_management(self):
        """Manage trained models"""
        st.markdown("### 📊 Model Management")
        
        trained_models = st.session_state.get('trained_models', {})
        
        if not trained_models:
            st.info("No trained models yet. Train your first model!")
            return
        
        # Filter and search
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_term = st.text_input("🔍 Search models:", key="model_search")
        
        with col2:
            sort_by = st.selectbox("Sort by:", ["Accuracy", "Date", "Name"], key="model_sort")
        
        with col3:
            st.metric("Total Models", len(trained_models))
        
        # Filter models
        filtered_models = list(trained_models.items())
        
        if search_term:
            filtered_models = [(k, v) for k, v in filtered_models if search_term.lower() in k.lower()]
        
        # Sort
        if sort_by == "Accuracy":
            filtered_models.sort(key=lambda x: x[1].get('metrics', {}).get('accuracy', 0), reverse=True)
        elif sort_by == "Date":
            filtered_models.sort(key=lambda x: x[1].get('created_at', ''), reverse=True)
        else:
            filtered_models.sort(key=lambda x: x[0])
        
        # Display models
        for model_name, model_info in filtered_models:
            with st.expander(f"📊 {model_name}"):
                metrics = model_info.get('metrics', {})
                model_type = model_info.get('model_type', 'Unknown')
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Type:** {model_type}")
                    st.markdown(f"**Created:** {model_info.get('created_at', 'Unknown')[:19]}")
                    st.markdown(f"**Features:** {len(model_info.get('features', []))}")
                
                with col2:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
                    st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
                
                with col3:
                    st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                    st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                
                # Actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Export", width="stretch", key=f"export_model_{model_name}"):
                        self.export_model(model_name, model_info)
                
                with col2:
                    if st.button("📋 Details", width="stretch", key=f"details_model_{model_name}"):
                        st.json(model_info.get('parameters', {}))
                
                with col3:
                    if st.button("🗑️ Delete", width="stretch", key=f"delete_model_{model_name}"):
                        self.delete_model(model_name)
                        st.rerun()
    
    def export_model(self, model_name, model_info):
        """Export model"""
        try:
            # Create export package
            export_data = {
                'model_name': model_name,
                'model_type': model_info.get('model_type'),
                'metrics': model_info.get('metrics'),
                'features': model_info.get('features'),
                'created_at': model_info.get('created_at')
            }
            
            # Save model file
            model_path = self.saved_models_dir / f"{model_name}.joblib"
            
            st.success(f"✅ Model exported to: `{model_path}`")
            st.json(export_data)
            
        except Exception as e:
            st.error(f"Export failed: {e}")
    
    def delete_model(self, model_name):
        """Delete model"""
        try:
            # Remove from session state
            if model_name in st.session_state.trained_models:
                del st.session_state.trained_models[model_name]
            
            # Remove from disk
            model_path = self.saved_models_dir / f"{model_name}.joblib"
            if model_path.exists():
                model_path.unlink()
            
            st.success(f"✅ Model deleted: {model_name}")
            
        except Exception as e:
            st.error(f"Delete failed: {e}")
