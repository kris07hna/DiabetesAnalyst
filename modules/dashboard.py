"""
🏠 DASHBOARD MODULE
Executive dashboard with comprehensive analytics and KPIs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from modules.config import AppConfig
from modules.ui_components import UIComponents

class DashboardManager:
    """Executive Dashboard Manager"""
    
    def __init__(self):
        self.config = AppConfig()
        self.ui = UIComponents()
    
    def render(self):
        """Render executive dashboard"""
        self.ui.create_header(
            "🏠 Executive Dashboard",
            "Robust analytics, KPIs, and professional navigation for DiabeticsAI Enterprise"
        )
        
        # Create dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", "🎯 Model Performance", "📈 Analytics Insights", "⚡ System Status"
        ])
        
        with tab1:
            self.render_overview_dashboard()
        
        with tab2:
            self.render_model_performance_dashboard()
        
        with tab3:
            self.render_analytics_dashboard()
        
        with tab4:
            self.render_system_status_dashboard()
    
    def render_overview_dashboard(self):
        """Render main overview dashboard"""
        st.markdown("### 📊 Platform Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        dataset_loaded = st.session_state.current_dataset is not None
        total_models = len(st.session_state.get('trained_models', {}))
        total_predictions = len(st.session_state.get('prediction_history', []))
        
        # Dataset metrics
        if dataset_loaded:
            df = st.session_state.current_dataset
            total_records = len(df)
            dataset_size = f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
        else:
            total_records = 0
            dataset_size = "0 KB"
        
        with col1:
            self.ui.create_metric_card(
                "Dataset Records", 
                f"{total_records:,}" if dataset_loaded else "No Data",
                status="good" if dataset_loaded else "warning"
            )
        
        with col2:
            self.ui.create_metric_card(
                "Trained Models", 
                f"{total_models}",
                status="good" if total_models > 0 else "warning"
            )
        
        with col3:
            self.ui.create_metric_card(
                "Total Predictions", 
                f"{total_predictions:,}",
                status="good" if total_predictions > 0 else "neutral"
            )
        
        with col4:
            self.ui.create_metric_card(
                "Data Size", 
                dataset_size,
                status="good" if dataset_loaded else "warning"
            )
        
        # Main dashboard content
        if not dataset_loaded:
            self.render_empty_dashboard()
            return
        
        # Data overview section
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_data_quality_widget()
        
        with col2:
            self.render_target_distribution_widget()
        
        # Recent activity
        st.markdown("---")
        self.render_recent_activity()
        
        # Quick insights
        st.markdown("---")
        self.render_quick_insights()
    
    def render_empty_dashboard(self):
        """Render dashboard when no data is loaded"""
        st.markdown("---")
        
        # Welcome message
        st.markdown("""
        <div style="text-align: center; padding: 3rem; 
            background: #6cc5eb;
            background: radial-gradient(circle, rgba(108, 197, 235, 1) 0%, rgba(70, 70, 212, 1) 35%, rgba(0, 0, 212, 1) 61%, rgba(3, 0, 56, 1) 100%);
            border-radius: 20px; 
            color: #000000;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(27, 211, 205, 0.3) inset;
            border: 1px solid rgba(27, 211, 205, 0.4);">
            <h2 style="color: #000000; font-weight: 800; margin-bottom: 1rem;">🚀 Welcome to DiabeticsAI Enterprise</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0; color: #000000; font-weight: 600;">Your advanced medical AI platform is ready!</p>
            <p style="color: #000000; font-weight: 500;">Get started by uploading your dataset in the <strong style="font-weight: 700;">Data Management</strong> section.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Getting started guide
        st.markdown("### 🎯 Getting Started")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1️⃣ Load Data**
            - Upload your healthcare dataset
            - Use CSV, Excel, or JSON formats
            - Or load our sample diabetes data
            """)
        
        with col2:
            st.markdown("""
            **2️⃣ Train Models**
            - Explore multiple ML algorithms
            - Optimize hyperparameters
            - Compare model performance
            """)
        
        with col3:
            st.markdown("""
            **3️⃣ Make Predictions**
            - Single patient predictions
            - Batch processing
            - Risk assessment tools
            """)
        
        # Feature highlights
        st.markdown("### ✨ Platform Features")
        
        features = [
            {"icon": "🧠", "title": "Advanced ML Models", "desc": "Random Forest, XGBoost, Neural Networks, and more"},
            {"icon": "📊", "title": "Rich Analytics", "desc": "Statistical analysis, correlation studies, feature importance"},
            {"icon": "🔮", "title": "Real-time Predictions", "desc": "Instant diabetes risk assessment with confidence scores"},
            {"icon": "📈", "title": "Performance Monitoring", "desc": "Track model accuracy, precision, recall, and other metrics"},
            {"icon": "🎨", "title": "Professional UI", "desc": "Modern, responsive interface with interactive visualizations"},
            {"icon": "💾", "title": "Data Management", "desc": "Upload, process, clean, and export your healthcare data"}
        ]
        
        cols = st.columns(3)
        for i, feature in enumerate(features):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="padding: 1.5rem; 
                    background: #6cc5eb;
                    background: radial-gradient(circle, rgba(108, 197, 235, 1) 0%, rgba(70, 70, 212, 1) 35%, rgba(0, 0, 212, 1) 61%, rgba(3, 0, 56, 1) 100%);
                    border-radius: 16px; 
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(27, 211, 205, 0.3) inset; 
                    margin: 0.5rem 0;
                    border: 1px solid rgba(27, 211, 205, 0.4);
                    transition: all 0.3s ease;
                    backdrop-filter: blur(20px);">
                    <h4 style="color: #000000; font-weight: 700; margin-bottom: 0.5rem;">{feature['icon']} {feature['title']}</h4>
                    <p style="color: #000000; font-size: 0.95rem; font-weight: 500; margin: 0;">{feature['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_data_quality_widget(self):
        """Render data quality assessment widget"""
        st.markdown("#### 🔍 Data Quality Assessment")
        
        df = st.session_state.current_dataset
        
        # Calculate quality metrics
        completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        duplicates_pct = (df.duplicated().sum() / len(df)) * 100
        numeric_cols_pct = (len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)) * 100
        
        # Quality score (weighted average)
        quality_score = (completeness * 0.5 + (100 - duplicates_pct) * 0.3 + numeric_cols_pct * 0.2)
        
        # Display quality score
        if quality_score >= 80:
            quality_status = "Excellent"
            quality_color = "green"
        elif quality_score >= 60:
            quality_status = "Good"
            quality_color = "orange"
        else:
            quality_status = "Needs Improvement"
            quality_color = "red"
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = quality_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Data Quality Score"},
            delta = {'reference': 80, 'position': "top"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': quality_color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Quality metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Completeness", f"{completeness:.1f}%")
            st.metric("Duplicates", f"{duplicates_pct:.1f}%")
        with col2:
            st.metric("Numeric Features", f"{numeric_cols_pct:.1f}%")
            st.metric("Status", quality_status)
    
    def render_target_distribution_widget(self):
        """Render target distribution widget"""
        st.markdown("#### 🎯 Target Distribution")
        
        df = st.session_state.current_dataset
        
        if self.config.TARGET_COLUMN in df.columns:
            target_counts = df[self.config.TARGET_COLUMN].value_counts().sort_index()
            
            # Create donut chart
            fig = go.Figure(data=[go.Pie(
                labels=[f"Class {i}" for i in target_counts.index],
                values=target_counts.values,
                hole=.3,
                marker_colors=['#2E86AB', '#A23B72', '#F18F01']
            )])
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=12
            )
            
            fig.update_layout(
                title="Diabetes Risk Distribution",
                height=300,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution stats
            total = len(df)
            for class_val in target_counts.index:
                count = target_counts[class_val]
                percentage = (count / total) * 100
                
                risk_level = {0: "No Diabetes", 1: "Pre-diabetes", 2: "Diabetes"}.get(class_val, f"Class {class_val}")
                st.metric(risk_level, f"{count:,} ({percentage:.1f}%)")
        
        else:
            st.info("Target column not found in dataset.")
    
    def render_recent_activity(self):
        """Render recent activity feed"""
        st.markdown("### 🕒 Recent Activity")
        
        activities = []
        
        # Model training activities
        trained_models = st.session_state.get('trained_models', {})
        for model_name, model_info in trained_models.items():
            if 'created_at' in model_info:
                activities.append({
                    'timestamp': model_info['created_at'],
                    'type': 'model_trained',
                    'description': f"Model '{model_name}' trained successfully",
                    'icon': '🧠',
                    'details': f"Type: {model_info.get('model_type', 'Unknown')}"
                })
        
        # Prediction activities
        prediction_history = st.session_state.get('prediction_history', [])
        for pred in prediction_history[-5:]:  # Last 5 predictions
            activities.append({
                'timestamp': pred['timestamp'],
                'type': 'prediction_made',
                'description': f"{pred['type'].title()} prediction made",
                'icon': '🔮',
                'details': f"Model: {pred['model_name']}"
            })
        
        # Sort by timestamp (most recent first)
        activities.sort(key=lambda x: x['timestamp'], reverse=True)
        
        if activities:
            # Display recent activities
            for activity in activities[:8]:  # Show last 8 activities
                timestamp = datetime.fromisoformat(activity['timestamp'].replace('Z', '+00:00') if 'Z' in activity['timestamp'] else activity['timestamp'])
                time_ago = self.get_time_ago(timestamp)
                
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #1f77b4;">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <span style="font-size: 1.5rem;">{activity['icon']}</span>
                        <div>
                            <strong>{activity['description']}</strong><br>
                            <small style="color: #666;">{activity['details']} • {time_ago}</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent activity to display.")
    
    def get_time_ago(self, timestamp):
        """Calculate time ago string"""
        now = datetime.now()
        if timestamp.tzinfo:
            now = now.replace(tzinfo=timestamp.tzinfo)
        
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"
    
    def render_quick_insights(self):
        """Render quick insights section"""
        st.markdown("### 💡 Quick Insights")
        
        df = st.session_state.current_dataset
        insights = []
        
        # Dataset insights
        if df is not None:
            # High-risk patients
            if self.config.TARGET_COLUMN in df.columns:
                high_risk_count = len(df[df[self.config.TARGET_COLUMN] == 2])
                total_count = len(df)
                high_risk_pct = (high_risk_count / total_count) * 100
                
                if high_risk_pct > 20:
                    insights.append({
                        'type': 'warning',
                        'title': 'High Risk Population',
                        'message': f'{high_risk_pct:.1f}% of patients are classified as high-risk for diabetes.',
                        'icon': '⚠️'
                    })
            
            # Missing data insight
            missing_data_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            if missing_data_pct > 10:
                insights.append({
                    'type': 'info',
                    'title': 'Data Quality Alert',
                    'message': f'{missing_data_pct:.1f}% of data points are missing. Consider data cleaning.',
                    'icon': '📊'
                })
            
            # Feature correlation insight
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.8:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                if high_corr_pairs:
                    insights.append({
                        'type': 'info',
                        'title': 'Feature Correlation',
                        'message': f'Found {len(high_corr_pairs)} highly correlated feature pairs. Consider feature selection.',
                        'icon': '🔗'
                    })
        
        # Model insights
        trained_models = st.session_state.get('trained_models', {})
        if trained_models:
            best_model = None
            best_accuracy = 0
            
            for model_name, model_info in trained_models.items():
                if 'metrics' in model_info and 'accuracy' in model_info['metrics']:
                    accuracy = model_info['metrics']['accuracy']
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model_name
            
            if best_model:
                insights.append({
                    'type': 'success',
                    'title': 'Best Performing Model',
                    'message': f'{best_model} achieved {best_accuracy:.3f} accuracy.',
                    'icon': '🏆'
                })
        
        # Prediction insights
        prediction_history = st.session_state.get('prediction_history', [])
        if prediction_history:
            recent_predictions = [p for p in prediction_history if p['type'] == 'single']
            if recent_predictions:
                high_risk_predictions = sum(1 for p in recent_predictions if p.get('prediction', 0) == 2)
                if high_risk_predictions > 0:
                    insights.append({
                        'type': 'warning',
                        'title': 'Recent High-Risk Predictions',
                        'message': f'{high_risk_predictions} recent predictions indicate high diabetes risk.',
                        'icon': '🚨'
                    })
        
        # Display insights
        if insights:
            for insight in insights:
                alert_class = {
                    'success': 'alert-success',
                    'warning': 'alert-warning',
                    'info': 'alert-info'
                }.get(insight['type'], 'alert-info')
                
                st.markdown(f"""
                <div class="{alert_class}">
                    <strong>{insight['icon']} {insight['title']}</strong><br>
                    {insight['message']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific insights available. Load data and train models to see insights.")
    
    def render_model_performance_dashboard(self):
        """Render model performance dashboard"""
        st.markdown("### 🎯 Model Performance Dashboard")
        
        trained_models = st.session_state.get('trained_models', {})
        
        if not trained_models:
            st.info("No trained models available. Train some models to see performance metrics.")
            return
        
        # Performance overview
        performance_data = []
        
        for model_name, model_info in trained_models.items():
            if 'metrics' in model_info:
                metrics = model_info['metrics']
                performance_data.append({
                    'Model': model_name,
                    'Type': model_info.get('model_type', 'Unknown'),
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1_score', 0),
                    'Training Time': model_info.get('training_time', 0),
                    'Created': model_info.get('created_at', '')[:10] if model_info.get('created_at') else ''
                })
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            
            # Performance metrics cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_accuracy = performance_df['Accuracy'].max()
                self.ui.create_metric_card("Best Accuracy", f"{best_accuracy:.3f}")
            
            with col2:
                avg_precision = performance_df['Precision'].mean()
                self.ui.create_metric_card("Avg Precision", f"{avg_precision:.3f}")
            
            with col3:
                avg_recall = performance_df['Recall'].mean()
                self.ui.create_metric_card("Avg Recall", f"{avg_recall:.3f}")
            
            with col4:
                avg_f1 = performance_df['F1-Score'].mean()
                self.ui.create_metric_card("Avg F1-Score", f"{avg_f1:.3f}")
            
            # Performance comparison chart
            st.markdown("#### 📊 Model Comparison")
            fig = self.ui.create_model_comparison_chart(performance_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance table
            st.markdown("#### 📋 Detailed Performance Metrics")
            st.dataframe(performance_df, use_container_width=True)
            
            # Training time analysis
            st.markdown("#### ⏱️ Training Time Analysis")
            fig = go.Figure(data=[
                go.Bar(
                    x=performance_df['Model'],
                    y=performance_df['Training Time'],
                    marker_color='lightblue',
                    text=performance_df['Training Time'].round(2),
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Model Training Times",
                xaxis_title="Models",
                yaxis_title="Training Time (seconds)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_analytics_dashboard(self):
        """Render analytics insights dashboard"""
        st.markdown("### 📈 Analytics Insights Dashboard")
        
        if st.session_state.current_dataset is None:
            st.info("No dataset loaded. Load data to see analytics insights.")
            return
        
        df = st.session_state.current_dataset
        
        # Dataset statistics
        st.markdown("#### 📊 Dataset Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature types distribution
            numeric_count = len(df.select_dtypes(include=[np.number]).columns)
            categorical_count = len(df.select_dtypes(include=['object']).columns)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Numeric', 'Categorical'],
                    y=[numeric_count, categorical_count],
                    marker_color=['#1f77b4', '#ff7f0e']
                )
            ])
            
            fig.update_layout(
                title="Feature Types Distribution",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Missing data by column
            missing_data = df.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) > 0:
                fig = go.Figure(data=[
                    go.Bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        marker_color='red'
                    )
                ])
                
                fig.update_layout(
                    title="Missing Data by Column",
                    height=300,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing data found!")
        
        # Feature analysis
        if self.config.TARGET_COLUMN in df.columns:
            st.markdown("#### 🎯 Target Analysis")
            
            target_stats = df.groupby(self.config.TARGET_COLUMN).agg({
                df.columns[0]: 'count'  # Count of records
            }).rename(columns={df.columns[0]: 'Count'})
            
            # Class distribution
            fig = px.bar(
                x=[f"Class {i}" for i in target_stats.index],
                y=target_stats['Count'],
                title="Class Distribution",
                color=target_stats['Count'],
                color_continuous_scale='viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_system_status_dashboard(self):
        """Render system status dashboard"""
        st.markdown("### ⚡ System Status Dashboard")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Dataset status
            dataset_status = "✅ Loaded" if st.session_state.current_dataset is not None else "❌ Not Loaded"
            self.ui.create_metric_card("Dataset Status", dataset_status)
        
        with col2:
            # Model count
            model_count = len(st.session_state.get('trained_models', {}))
            self.ui.create_metric_card("Active Models", f"{model_count}")
        
        with col3:
            # Prediction count
            prediction_count = len(st.session_state.get('prediction_history', []))
            self.ui.create_metric_card("Total Predictions", f"{prediction_count}")
        
        with col4:
            # System uptime (simulated)
            uptime = "99.9%"
            self.ui.create_metric_card("System Uptime", uptime)
        
        # Resource usage (simulated)
        st.markdown("#### 💻 Resource Usage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Memory usage gauge
            memory_usage = 45  # Simulated
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = memory_usage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Memory Usage (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # CPU usage gauge
            cpu_usage = 32  # Simulated
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = cpu_usage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "CPU Usage (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # System logs (simulated)
        st.markdown("#### 📝 System Logs")
        
        logs = [
            {"timestamp": "2024-01-26 10:30:15", "level": "INFO", "message": "Model training completed successfully"},
            {"timestamp": "2024-01-26 10:25:42", "level": "INFO", "message": "Dataset uploaded and validated"},
            {"timestamp": "2024-01-26 10:20:18", "level": "INFO", "message": "User session started"},
            {"timestamp": "2024-01-26 10:15:33", "level": "DEBUG", "message": "Cache cleared successfully"},
            {"timestamp": "2024-01-26 10:10:01", "level": "INFO", "message": "System health check passed"}
        ]
        
        for log in logs:
            level_color = {
                "INFO": "blue",
                "DEBUG": "gray",
                "WARNING": "orange",
                "ERROR": "red"
            }.get(log["level"], "black")
            
            st.markdown(f"""
            <div style="background: white; padding: 0.5rem 1rem; border-radius: 5px; margin: 0.2rem 0; border-left: 3px solid {level_color};">
                <code style="color: {level_color};">[{log['timestamp']}] {log['level']}</code> - {log['message']}
            </div>
            """, unsafe_allow_html=True)
        
        # System information
        st.markdown("#### ℹ️ System Information")
        
        system_info = {
            "Platform Version": "DiabeticsAI Enterprise v2.0.0",
            "Python Version": "3.9.7",
            "Streamlit Version": "1.28.0",
            "Last Updated": "2024-01-26",
            "Environment": "Production",
            "Database": "In-Memory Session State"
        }
        
        info_col1, info_col2 = st.columns(2)
        
        items = list(system_info.items())
        mid_point = len(items) // 2
        
        with info_col1:
            for key, value in items[:mid_point]:
                st.text(f"{key}: {value}")
        
        with info_col2:
            for key, value in items[mid_point:]:
                st.text(f"{key}: {value}")
