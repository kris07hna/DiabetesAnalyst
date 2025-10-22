"""
📋 APPLICATION CONFIGURATION
Enterprise settings and constants for DiabeticsAI Platform
"""

import os
import pandas as pd
import streamlit as st
from pathlib import Path

class AppConfig:
    """Application Configuration Class"""
    
    # Application Info
    APP_NAME = "DiabeticsAI Enterprise"
    APP_VERSION = "2.0.0"
    APP_DESCRIPTION = "Advanced Medical AI Analytics Platform"
    
    # File Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "saved_models"
    UPLOADS_DIR = BASE_DIR / "uploads"
    EXPORTS_DIR = BASE_DIR / "exports"
    
    @staticmethod
    def fix_arrow_compatibility(df):
        """Fix Arrow compatibility issues for Streamlit dataframes"""
        try:
            df_copy = df.copy()
            
            # Convert problematic dtypes
            for col in df_copy.columns:
                col_dtype = str(df_copy[col].dtype)
                
                # Handle mixed types in object columns
                if df_copy[col].dtype == 'object':
                    # Check if column contains mixed types
                    sample_values = df_copy[col].dropna().head(100)
                    if len(sample_values) > 0:
                        # Try to convert to numeric if possible
                        try:
                            numeric_series = pd.to_numeric(df_copy[col], errors='coerce')
                            if not numeric_series.isna().all():
                                # If most values can be converted to numeric, use numeric
                                valid_numeric = (~numeric_series.isna()).sum()
                                if valid_numeric / len(df_copy) > 0.8:
                                    df_copy[col] = numeric_series
                                else:
                                    df_copy[col] = df_copy[col].astype(str)
                            else:
                                df_copy[col] = df_copy[col].astype(str)
                        except:
                            # Force to string if conversion fails
                            df_copy[col] = df_copy[col].astype(str)
                
                # Handle nullable integer dtypes
                elif col_dtype.startswith('Int'):
                    try:
                        df_copy[col] = df_copy[col].astype('int64')
                    except:
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype('int64')
                
                # Handle nullable float dtypes
                elif col_dtype.startswith('Float'):
                    try:
                        df_copy[col] = df_copy[col].astype('float64')
                    except:
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype('float64')
                
                # Handle boolean dtypes
                elif col_dtype.startswith('boolean') or col_dtype == 'bool':
                    try:
                        df_copy[col] = df_copy[col].astype('bool')
                    except:
                        df_copy[col] = df_copy[col].astype(str)
                
                # Handle datetime columns
                elif 'datetime' in col_dtype:
                    try:
                        df_copy[col] = pd.to_datetime(df_copy[col])
                    except:
                        df_copy[col] = df_copy[col].astype(str)
                
                # Handle category columns
                elif col_dtype == 'category':
                    df_copy[col] = df_copy[col].astype(str)
            
            # Final check - ensure no problematic column names
            df_copy.columns = [str(col).replace(' ', '_').replace('-', '_') for col in df_copy.columns]
            
            return df_copy
            
        except Exception as e:
            st.warning(f"Arrow compatibility fix warning: {str(e)}")
            # Fallback: convert all object columns to string
            try:
                df_fallback = df.copy()
                for col in df_fallback.select_dtypes(include=['object']).columns:
                    df_fallback[col] = df_fallback[col].astype(str)
                return df_fallback
            except:
                return df
    
    @staticmethod
    def safe_dataframe_display(df, **kwargs):
        """Safely display dataframe with Arrow compatibility"""
        try:
            fixed_df = AppConfig.fix_arrow_compatibility(df)
            st.dataframe(fixed_df, **kwargs)
        except Exception as e:
            st.error(f"Display issue: {str(e)}")
            st.write("Data shape:", df.shape)
            st.write("Columns:")
            for i, col in enumerate(df.columns):
                st.write(f"{i}:\"{col}\"")
            
            # Try to display raw data as fallback
            try:
                st.write("Raw data preview:")
                st.write(df.head())
            except:
                st.write("Cannot display data preview")
    
    # Model Configuration
    SUPPORTED_MODELS = {
        "Random Forest": {
            "class": "RandomForestClassifier",
            "params": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        "XGBoost": {
            "class": "XGBClassifier",
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0]
            }
        },
        "Gradient Boosting": {
            "class": "GradientBoostingClassifier",
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0]
            }
        },
        "SVM": {
            "class": "SVC",
            "params": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto", 0.001, 0.01]
            }
        },
        "Logistic Regression": {
            "class": "LogisticRegression",
            "params": {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2", "elasticnet"],
                "solver": ["liblinear", "saga", "lbfgs"]
            }
        },
        "Neural Network": {
            "class": "MLPClassifier",
            "params": {
                "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                "activation": ["relu", "tanh", "logistic"],
                "alpha": [0.001, 0.01, 0.1],
                "learning_rate": ["constant", "adaptive"]
            }
        }
    }
    
    # Dataset Configuration
    DIABETES_FEATURES = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", 
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", 
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", 
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
    ]
    
    TARGET_COLUMN = "Diabetes_012"
    
    # UI Configuration
    COLORS = {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "success": "#2ca02c",
        "danger": "#d62728",
        "warning": "#ff7f0e",
        "info": "#17a2b8",
        "light": "#f8f9fa",
        "dark": "#343a40"
    }
    
    # Chart Configuration
    CHART_CONFIG = {
        "font_family": "Arial, sans-serif",
        "font_size": 12,
        "title_size": 16,
        "legend_size": 10,
        "grid_alpha": 0.3,
        "figure_size": (10, 6)
    }
    
    # Performance Metrics
    METRICS = [
        "accuracy", "precision", "recall", "f1_score", 
        "roc_auc", "log_loss", "confusion_matrix"
    ]
    
    # File Upload Settings
    MAX_FILE_SIZE = 200  # MB
    ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls", ".json", ".parquet"]
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.UPLOADS_DIR, cls.EXPORTS_DIR]:
            directory.mkdir(exist_ok=True)
    
    @classmethod
    def get_model_params(cls, model_name):
        """Get parameter suggestions for a model"""
        return cls.SUPPORTED_MODELS.get(model_name, {}).get("params", {})
