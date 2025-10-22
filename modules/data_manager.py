"""
📁 DATA MANAGEMENT MODULE
Advanced dataset upload, processing, and management for DiabeticsAI Enterprise
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io
import json
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from modules.config import AppConfig
from modules.ui_components import UIComponents

class DataManager:
    """Advanced Data Management System"""
    
    def __init__(self):
        self.config = AppConfig()
        self.ui = UIComponents()
        self.config.create_directories()
    
    def render(self):
        """Render data management interface"""
        self.ui.create_header(
            "📁 Data Management Center",
            "Upload, process, and manage your healthcare datasets"
        )
        
        # Create tabs for different data operations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📤 Upload Dataset", "🔍 Data Explorer", "🧹 Data Preprocessing", 
            "📊 Data Quality", "💾 Export Data"
        ])
        
        with tab1:
            self.render_upload_interface()
        
        with tab2:
            self.render_data_explorer()
        
        with tab3:
            self.render_preprocessing_interface()
        
        with tab4:
            self.render_data_quality_interface()
        
        with tab5:
            self.render_export_interface()
    
    def render_upload_interface(self):
        """Render dataset upload interface"""
        st.markdown("### 📤 Dataset Upload")
        
        # Upload options
        upload_method = st.radio(
            "Choose upload method:",
            ["📁 File Upload", "🌐 URL Import", "📋 Sample Data"]
        )
        
        if upload_method == "📁 File Upload":
            self.handle_file_upload()
        elif upload_method == "🌐 URL Import":
            self.handle_url_import()
        else:
            self.load_sample_data()
    
    def handle_file_upload(self):
        """Handle file upload with advanced validation"""
        st.markdown("#### Upload Your Dataset")
        
        # Enhanced file upload with multiple files support
        uploaded_files = st.file_uploader(
            "Choose file(s)",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Supported formats: CSV, Excel, JSON, Parquet",
            accept_multiple_files=True
        )
        
        # Dataset selection dropdown
        if 'uploaded_datasets' not in st.session_state:
            st.session_state.uploaded_datasets = {}
        
        # Show available datasets dropdown
        if st.session_state.uploaded_datasets or uploaded_files:
            available_datasets = list(st.session_state.uploaded_datasets.keys())
            if uploaded_files:
                available_datasets.extend([f.name for f in uploaded_files])
            
            selected_dataset = st.selectbox(
                "Select Dataset to Use:",
                options=["None"] + available_datasets,
                help="Choose which dataset to work with"
            )
            
            if selected_dataset != "None":
                if selected_dataset in st.session_state.uploaded_datasets:
                    # Load from session state
                    df = st.session_state.uploaded_datasets[selected_dataset]['data']
                    st.session_state.current_dataset = df
                    st.session_state.dataset_name = selected_dataset
                    st.session_state.dataset_timestamp = st.session_state.uploaded_datasets[selected_dataset]['timestamp']
                    self.ui.create_alert(f"✅ Selected dataset: {selected_dataset}", "success")
                    self.show_dataset_overview(df)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    # Show file details
                    file_details = {
                        "Filename": uploaded_file.name,
                        "File size": f"{uploaded_file.size / 1024:.2f} KB",
                        "File type": uploaded_file.type
                    }
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"📋 **File Details - {uploaded_file.name}:**")
                        for key, value in file_details.items():
                            st.write(f"- **{key}:** {value}")
                    
                    # Load data based on file type
                    df = self.load_file(uploaded_file)
                    
                    if df is not None:
                        # Show preview
                        with col2:
                            st.write("👀 **Data Preview:**")
                            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                        
                        # Store in session state with timestamp
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.uploaded_datasets[uploaded_file.name] = {
                            'data': df,
                            'timestamp': timestamp,
                            'file_type': uploaded_file.type
                        }
                        
                        # Set as current dataset
                        st.session_state.current_dataset = df
                        st.session_state.dataset_name = uploaded_file.name
                        st.session_state.dataset_timestamp = timestamp
                        
                        # Show success message
                        self.ui.create_alert(
                            f"✅ Successfully loaded {uploaded_file.name} at {timestamp}!", 
                            "success"
                        )
                        
                        # Auto-detect target column for ML
                        self.detect_target_column(df)
                        
                        # Show basic statistics
                        self.show_dataset_overview(df)
                        
                except Exception as e:
                    st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
    
    def detect_target_column(self, df):
        """Auto-detect potential target column for ML"""
        potential_targets = []
        
        # Common target column names for diabetes/health data
        target_keywords = ['diabetes', 'target', 'class', 'label', 'outcome', 'result', 
                          'diagnosis', 'disease', 'condition', 'status']
        
        for col in df.columns:
            col_lower = col.lower()
            # Check if column name contains target keywords
            if any(keyword in col_lower for keyword in target_keywords):
                potential_targets.append(col)
            # Check if column has binary values (common for classification)
            elif df[col].nunique() <= 10 and df[col].dtype in ['int64', 'float64']:
                potential_targets.append(col)
        
        if potential_targets:
            st.session_state.suggested_target = potential_targets[0]
            st.info(f"💡 Suggested target column: **{potential_targets[0]}** (Auto-detected)")
            
            # Show all potential targets
            if len(potential_targets) > 1:
                st.write("🎯 Other potential target columns:")
                for target in potential_targets[1:]:
                    st.write(f"   - {target}")
    
    def load_file(self, uploaded_file):
        """Load file based on its type with Arrow compatibility"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                st.error("Unsupported file format")
                return None
            
            # Fix Arrow serialization issues
            df = self.config.fix_arrow_compatibility(df)
            return df
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    
    def handle_url_import(self):
        """Handle URL-based data import"""
        st.markdown("#### Import from URL")
        
        url = st.text_input(
            "Enter dataset URL:",
            placeholder="https://example.com/dataset.csv"
        )
        
        if st.button("📥 Import Data") and url:
            try:
                with st.spinner("Importing data..."):
                    if url.endswith('.csv'):
                        df = pd.read_csv(url)
                    elif url.endswith('.json'):
                        df = pd.read_json(url)
                    else:
                        st.error("Unsupported URL format. Please use CSV or JSON.")
                        return
                    
                    # Fix Arrow compatibility
                    df = self.config.fix_arrow_compatibility(df)
                    st.session_state.current_dataset = df
                    st.session_state.dataset_name = "URL_Import"
                    
                    self.ui.create_alert("✅ Data imported successfully!", "success")
                    self.show_dataset_overview(df)
                    
            except Exception as e:
                st.error(f"Error importing from URL: {str(e)}")
    
    def load_sample_data(self):
        """Load sample diabetes dataset"""
        st.markdown("#### Load Sample Dataset")
        
        if st.button("📊 Load Diabetes Sample Data"):
            try:
                # Load the main diabetes dataset
                sample_file = self.config.BASE_DIR / "diabetes_012_health_indicators_BRFSS2015.csv"
                if sample_file.exists():
                    df = pd.read_csv(sample_file)
                    df = self.config.fix_arrow_compatibility(df)  # Fix Arrow issues
                    st.session_state.current_dataset = df
                    st.session_state.dataset_name = "Diabetes_Sample"
                    
                    self.ui.create_alert("✅ Sample dataset loaded!", "success")
                    self.show_dataset_overview(df)
                else:
                    st.error("Sample dataset not found. Please upload your own data.")
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    
    def show_dataset_overview(self, df):
        """Show comprehensive dataset overview"""
        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.ui.create_metric_card("Rows", f"{df.shape[0]:,}")
        with col2:
            self.ui.create_metric_card("Columns", f"{df.shape[1]:,}")
        with col3:
            self.ui.create_metric_card("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        with col4:
            missing_count = df.isnull().sum().sum()
            self.ui.create_metric_card("Missing Values", f"{missing_count:,}")
        
        # Data preview
        st.markdown("#### 🔍 Data Preview")
        self.config.safe_dataframe_display(df.head(10), width="stretch")
    
    def render_data_explorer(self):
        """Render data exploration interface"""
        if st.session_state.current_dataset is None:
            self.ui.create_alert("⚠️ No dataset loaded. Please upload a dataset first.", "warning")
            return
        
        df = st.session_state.current_dataset
        st.markdown("### 🔍 Data Explorer")
        
        # Column information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Column Information")
            column_info = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                unique_vals = df[col].nunique()
                missing_vals = df[col].isnull().sum()
                
                column_info.append({
                    "Column": col,
                    "Type": col_type,
                    "Unique": unique_vals,
                    "Missing": missing_vals,
                    "Missing %": f"{(missing_vals/len(df)*100):.2f}%"
                })
            
            column_df = pd.DataFrame(column_info)
            # Fix Arrow compatibility for display
            self.config.safe_dataframe_display(column_df, width="stretch")
        
        with col2:
            st.markdown("#### 📊 Statistical Summary")
            selected_column = st.selectbox("Select column for analysis:", df.columns)
            
            if selected_column:
                col_data = df[selected_column]
                
                if col_data.dtype in ['int64', 'float64']:
                    # Numerical column analysis
                    stats = {
                        "Count": col_data.count(),
                        "Mean": f"{col_data.mean():.2f}",
                        "Median": f"{col_data.median():.2f}",
                        "Std Dev": f"{col_data.std():.2f}",
                        "Min": f"{col_data.min():.2f}",
                        "Max": f"{col_data.max():.2f}"
                    }
                    
                    for key, value in stats.items():
                        st.metric(key, value)
                else:
                    # Categorical column analysis
                    value_counts = col_data.value_counts().head(10)
                    st.bar_chart(value_counts)
    
    def render_preprocessing_interface(self):
        """Render data preprocessing interface"""
        if st.session_state.current_dataset is None:
            self.ui.create_alert("⚠️ No dataset loaded. Please upload a dataset first.", "warning")
            return
        
        df = st.session_state.current_dataset.copy()
        st.markdown("### 🧹 Data Preprocessing")
        
        # Preprocessing options
        st.markdown("#### 🛠️ Preprocessing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Missing Value Handling:**")
            missing_strategy = st.selectbox(
                "Strategy:",
                ["Keep as is", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode"]
            )
            
            st.markdown("**Scaling Options:**")
            scaling_method = st.selectbox(
                "Method:",
                ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
            )
        
        with col2:
            st.markdown("**Feature Selection:**")
            columns_to_keep = st.multiselect(
                "Select columns to keep:",
                df.columns.tolist(),
                default=df.columns.tolist()
            )
            
            st.markdown("**Data Types:**")
            if st.checkbox("Auto-detect data types"):
                df = df.infer_objects()
        
        # Apply preprocessing
        if st.button("🔧 Apply Preprocessing"):
            try:
                processed_df = self.apply_preprocessing(
                    df, missing_strategy, scaling_method, columns_to_keep
                )
                
                st.session_state.current_dataset = processed_df
                self.ui.create_alert("✅ Preprocessing applied successfully!", "success")
                
                # Show before/after comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Before:**")
                    st.write(f"Shape: {df.shape}")
                    st.write(f"Missing values: {df.isnull().sum().sum()}")
                
                with col2:
                    st.markdown("**After:**")
                    st.write(f"Shape: {processed_df.shape}")
                    st.write(f"Missing values: {processed_df.isnull().sum().sum()}")
                    
            except Exception as e:
                st.error(f"Error in preprocessing: {str(e)}")
    
    def apply_preprocessing(self, df, missing_strategy, scaling_method, columns_to_keep):
        """Apply selected preprocessing steps"""
        # Select columns
        df = df[columns_to_keep]
        
        # Handle missing values
        if missing_strategy == "Drop rows":
            df = df.dropna()
        elif missing_strategy == "Fill with mean":
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        elif missing_strategy == "Fill with median":
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        elif missing_strategy == "Fill with mode":
            for col in df.columns:
                if df[col].isnull().any():
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col] = df[col].fillna(mode_value[0])
        
        # Apply scaling
        if scaling_method != "None":
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                if scaling_method == "StandardScaler":
                    scaler = StandardScaler()
                elif scaling_method == "MinMaxScaler":
                    scaler = MinMaxScaler()
                
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        return df
    
    def render_data_quality_interface(self):
        """Render data quality assessment interface"""
        if st.session_state.current_dataset is None:
            self.ui.create_alert("⚠️ No dataset loaded. Please upload a dataset first.", "warning")
            return
        
        df = st.session_state.current_dataset
        st.markdown("### 📊 Data Quality Assessment")
        
        # Quality metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            self.ui.create_metric_card("Completeness", f"{completeness:.1f}%")
        
        with col2:
            duplicates = df.duplicated().sum()
            self.ui.create_metric_card("Duplicate Rows", f"{duplicates:,}")
        
        with col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            self.ui.create_metric_card("Numeric Columns", f"{numeric_cols}")
    
    def render_export_interface(self):
        """Render data export interface"""
        if st.session_state.current_dataset is None:
            self.ui.create_alert("⚠️ No dataset loaded. Please upload a dataset first.", "warning")
            return
        
        df = st.session_state.current_dataset
        st.markdown("### 💾 Export Data")
        
        # Export options
        export_format = st.selectbox(
            "Select export format:",
            ["CSV", "Excel", "JSON", "Parquet"]
        )
        
        filename = st.text_input(
            "Filename:",
            value=f"processed_data.{export_format.lower()}"
        )
        
        if st.button("📥 Download Data"):
            self.download_data(df, export_format, filename)
    
    def download_data(self, df, format_type, filename):
        """Generate download for processed data"""
        try:
            if format_type == "CSV":
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=filename,
                    mime='text/csv'
                )
            elif format_type == "Excel":
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Data', index=False)
                st.download_button(
                    label="📥 Download Excel",
                    data=output.getvalue(),
                    file_name=filename,
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            elif format_type == "JSON":
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=filename,
                    mime='application/json'
                )
        except Exception as e:
            st.error(f"Error generating download: {str(e)}")
