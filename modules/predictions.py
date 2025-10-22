"""
🔮 PREDICTIONS ENGINE MODULE
Advanced prediction system with real-time inference and batch processing
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from modules.config import AppConfig
from modules.ui_components import UIComponents

class PredictionEngine:
    """Advanced Prediction Engine"""
    
    def __init__(self):
        self.config = AppConfig()
        self.ui = UIComponents()
    
    def render(self):
        """Render prediction interface"""
        self.ui.create_header(
            "🔮 AI Prediction Engine",
            "Make real-time predictions and batch processing with trained models"
        )
        
        # Check if models are available
        if not st.session_state.get('trained_models'):
            self.ui.create_alert("⚠️ No trained models found. Please train a model first.", "warning")
            return
        
        # Create prediction tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Single Prediction", "📊 Batch Predictions", "📈 Prediction Analysis", "🔄 Model Inference"
        ])
        
        with tab1:
            self.render_single_prediction()
        
        with tab2:
            self.render_batch_predictions()
        
        with tab3:
            self.render_prediction_analysis()
        
        with tab4:
            self.render_model_inference()
    
    def render_single_prediction(self):
        """Render single prediction interface"""
        st.markdown("### 🎯 Single Patient Prediction")
        
        # Model selection
        available_models = list(st.session_state.trained_models.keys())
        selected_model = st.selectbox("Select prediction model:", available_models)
        
        if not selected_model:
            return
        
        model_info = st.session_state.trained_models[selected_model]
        features = model_info.get('features', [])
        
        st.markdown("#### 📝 Patient Information")
        st.markdown("*Enter patient details for diabetes risk prediction*")
        
        # Create input form
        input_data = {}
        
        # Organize features in columns for better layout
        col1, col2, col3 = st.columns(3)
        
        feature_groups = [features[i:i+len(features)//3+1] for i in range(0, len(features), len(features)//3+1)]
        
        columns = [col1, col2, col3]
        
        for i, feature_group in enumerate(feature_groups[:3]):
            with columns[i]:
                for feature in feature_group:
                    input_data[feature] = self.create_feature_input(feature)
        
        # Prediction button
        if st.button("🔮 Predict Diabetes Risk", type="primary"):
            prediction_result = self.make_single_prediction(
                model_info, input_data, selected_model
            )
            
            if prediction_result:
                self.display_prediction_result(prediction_result, input_data)
    
    def create_feature_input(self, feature):
        """Create appropriate input widget for each feature"""
        # Define feature types and ranges based on medical knowledge
        feature_configs = {
            'BMI': {'type': 'slider', 'min': 10.0, 'max': 60.0, 'default': 25.0, 'step': 0.1},
            'Age': {'type': 'slider', 'min': 18, 'max': 100, 'default': 45, 'step': 1},
            'MentHlth': {'type': 'slider', 'min': 0, 'max': 30, 'default': 0, 'step': 1},
            'PhysHlth': {'type': 'slider', 'min': 0, 'max': 30, 'default': 0, 'step': 1},
            'GenHlth': {'type': 'slider', 'min': 1, 'max': 5, 'default': 3, 'step': 1},
            'Education': {'type': 'slider', 'min': 1, 'max': 6, 'default': 4, 'step': 1},
            'Income': {'type': 'slider', 'min': 1, 'max': 8, 'default': 5, 'step': 1},
        }
        
        # Binary features (0 or 1)
        binary_features = [
            'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
            'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
            'NoDocbcCost', 'DiffWalk', 'Sex'
        ]
        
        if feature in binary_features:
            return st.selectbox(
                f"{feature.replace('_', ' ').title()}:",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key=f"input_{feature}"
            )
        elif feature in feature_configs:
            config = feature_configs[feature]
            return st.slider(
                f"{feature.replace('_', ' ').title()}:",
                min_value=config['min'],
                max_value=config['max'],
                value=config['default'],
                step=config['step'],
                key=f"input_{feature}"
            )
        else:
            # Default numeric input
            return st.number_input(
                f"{feature.replace('_', ' ').title()}:",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                key=f"input_{feature}"
            )
    
    def make_single_prediction(self, model_info, input_data, model_name):
        """Make a single prediction"""
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Apply scaling if model was trained with scaling
            if 'scaler' in model_info and model_info['scaler'] is not None:
                input_df = pd.DataFrame(
                    model_info['scaler'].transform(input_df),
                    columns=input_df.columns
                )
            
            # Make prediction
            model = model_info['model']
            prediction = model.predict(input_df)[0]
            
            # Get prediction probability if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_df)[0]
            else:
                probabilities = None
            
            # Store prediction in history
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'prediction': int(prediction),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'input_data': input_data,
                'type': 'single'
            }
            
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            st.session_state.prediction_history.append(prediction_record)
            
            return prediction_record
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None
    
    def display_prediction_result(self, prediction_result, input_data):
        """Display prediction results with rich visualization"""
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        prediction = prediction_result['prediction']
        probabilities = prediction_result['probabilities']
        
        # Main prediction display
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if prediction == 0:
                st.success("✅ **LOW DIABETES RISK**")
                st.markdown("*The model predicts this patient has a low risk of diabetes.*")
                risk_level = "Low"
                risk_color = "green"
            elif prediction == 1:
                st.warning("⚠️ **MODERATE DIABETES RISK**")
                st.markdown("*The model predicts this patient has a moderate risk of diabetes.*")
                risk_level = "Moderate"
                risk_color = "orange"
            else:
                st.error("🚨 **HIGH DIABETES RISK**")
                st.markdown("*The model predicts this patient has a high risk of diabetes.*")
                risk_level = "High"
                risk_color = "red"
        
        with col2:
            st.metric("Risk Level", risk_level)
        
        with col3:
            st.metric("Prediction", f"Class {prediction}")
        
        # Probability visualization
        if probabilities is not None:
            st.markdown("#### 📊 Risk Probability Distribution")
            
            # Create probability chart
            classes = [f"Class {i}" for i in range(len(probabilities))]
            risk_labels = ["No Diabetes", "Pre-diabetes", "Diabetes"][:len(probabilities)]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=risk_labels,
                    y=probabilities,
                    marker_color=['green', 'orange', 'red'][:len(probabilities)],
                    text=[f"{p:.1%}" for p in probabilities],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Prediction Confidence",
                xaxis_title="Risk Category",
                yaxis_title="Probability",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence metrics
            probabilities_array = np.array(probabilities)
            max_prob = np.max(probabilities_array)
            max_idx = np.argmax(probabilities_array)
            confidence = max_prob - max([p for i, p in enumerate(probabilities_array) if i != max_idx], default=0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{max_prob:.1%}")
            with col2:
                st.metric("Certainty", f"{confidence:.1%}")
        
        # Risk factors analysis
        self.display_risk_factors(input_data, prediction)
        
        # Recommendations
        self.display_recommendations(prediction, input_data)
    
    def display_risk_factors(self, input_data, prediction):
        """Display risk factors analysis"""
        st.markdown("#### 🔍 Risk Factors Analysis")
        
        # Define risk factor interpretations
        risk_factors = []
        
        if input_data.get('BMI', 0) > 30:
            risk_factors.append("🔴 High BMI (Obesity)")
        elif input_data.get('BMI', 0) > 25:
            risk_factors.append("🟡 Elevated BMI (Overweight)")
        
        if input_data.get('HighBP', 0) == 1:
            risk_factors.append("🔴 High Blood Pressure")
        
        if input_data.get('HighChol', 0) == 1:
            risk_factors.append("🔴 High Cholesterol")
        
        if input_data.get('Age', 0) > 65:
            risk_factors.append("🟡 Advanced Age")
        elif input_data.get('Age', 0) > 45:
            risk_factors.append("🟡 Middle Age")
        
        if input_data.get('PhysActivity', 0) == 0:
            risk_factors.append("🟡 Lack of Physical Activity")
        
        if input_data.get('Smoker', 0) == 1:
            risk_factors.append("🔴 Smoking")
        
        # Protective factors
        protective_factors = []
        
        if input_data.get('PhysActivity', 0) == 1:
            protective_factors.append("🟢 Regular Physical Activity")
        
        if input_data.get('Fruits', 0) == 1 and input_data.get('Veggies', 0) == 1:
            protective_factors.append("🟢 Healthy Diet (Fruits & Vegetables)")
        
        if input_data.get('BMI', 0) < 25:
            protective_factors.append("🟢 Normal BMI")
        
        # Display factors
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Factors:**")
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("- 🟢 No major risk factors identified")
        
        with col2:
            st.markdown("**Protective Factors:**")
            if protective_factors:
                for factor in protective_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("- 🟡 Limited protective factors")
    
    def display_recommendations(self, prediction, input_data):
        """Display personalized recommendations"""
        st.markdown("#### 💡 Personalized Recommendations")
        
        recommendations = []
        
        # BMI-based recommendations
        bmi = input_data.get('BMI', 0)
        if bmi > 30:
            recommendations.append("🏃‍♂️ **Weight Management**: Consider a structured weight loss program with medical supervision")
        elif bmi > 25:
            recommendations.append("🍎 **Healthy Weight**: Aim for gradual weight loss through balanced diet and exercise")
        
        # Physical activity recommendations
        if input_data.get('PhysActivity', 0) == 0:
            recommendations.append("🏋️‍♀️ **Exercise**: Start with 150 minutes of moderate exercise per week")
        
        # Diet recommendations
        if input_data.get('Fruits', 0) == 0 or input_data.get('Veggies', 0) == 0:
            recommendations.append("🥗 **Nutrition**: Include at least 5 servings of fruits and vegetables daily")
        
        # Medical monitoring
        if prediction >= 1:
            recommendations.append("🩺 **Medical Care**: Schedule regular check-ups and HbA1c testing")
            recommendations.append("📊 **Blood Sugar Monitoring**: Consider regular blood glucose monitoring")
        
        # Lifestyle recommendations
        if input_data.get('Smoker', 0) == 1:
            recommendations.append("🚭 **Smoking Cessation**: Seek support to quit smoking")
        
        if input_data.get('HighBP', 0) == 1:
            recommendations.append("💊 **Blood Pressure**: Monitor and manage blood pressure with healthcare provider")
        
        # Display recommendations
        for i, recommendation in enumerate(recommendations, 1):
            st.markdown(f"{i}. {recommendation}")
        
        if not recommendations:
            st.markdown("🎉 **Great job!** Continue maintaining your healthy lifestyle.")
    
    def render_batch_predictions(self):
        """Render batch prediction interface"""
        st.markdown("### 📊 Batch Predictions")
        st.markdown("*Upload a dataset to make predictions for multiple patients*")
        
        # Model selection
        available_models = list(st.session_state.trained_models.keys())
        selected_model = st.selectbox("Select model for batch predictions:", available_models, key="batch_model")
        
        if not selected_model:
            return
        
        model_info = st.session_state.trained_models[selected_model]
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch predictions:",
            type=['csv'],
            help="Upload a CSV file with the same features used for training"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                batch_df = pd.read_csv(uploaded_file)
                
                st.markdown("#### 📋 Data Preview")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                # Validate features
                required_features = model_info.get('features', [])
                missing_features = [f for f in required_features if f not in batch_df.columns]
                
                if missing_features:
                    st.error(f"Missing required features: {missing_features}")
                    return
                
                # Make batch predictions
                if st.button("🚀 Run Batch Predictions"):
                    predictions_df = self.make_batch_predictions(
                        model_info, batch_df, selected_model
                    )
                    
                    if predictions_df is not None:
                        self.display_batch_results(predictions_df)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    def make_batch_predictions(self, model_info, batch_df, model_name):
        """Make batch predictions"""
        try:
            with st.spinner("Making batch predictions..."):
                # Prepare data
                features = model_info.get('features', [])
                X = batch_df[features].copy()
                
                # Apply scaling if needed
                if 'scaler' in model_info and model_info['scaler'] is not None:
                    X = pd.DataFrame(
                        model_info['scaler'].transform(X),
                        columns=X.columns,
                        index=X.index
                    )
                
                # Make predictions
                model = model_info['model']
                predictions = model.predict(X)
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
                else:
                    probabilities = None
                
                # Create results dataframe
                results_df = batch_df.copy()
                results_df['Prediction'] = predictions
                results_df['Risk_Level'] = results_df['Prediction'].map({
                    0: 'Low Risk',
                    1: 'Moderate Risk', 
                    2: 'High Risk'
                })
                
                if probabilities is not None:
                    for i in range(probabilities.shape[1]):
                        results_df[f'Prob_Class_{i}'] = probabilities[:, i]
                
                # Store in session state
                batch_record = {
                    'timestamp': datetime.now().isoformat(),
                    'model_name': model_name,
                    'results': results_df,
                    'type': 'batch'
                }
                
                if 'prediction_history' not in st.session_state:
                    st.session_state.prediction_history = []
                st.session_state.prediction_history.append(batch_record)
                
                return results_df
                
        except Exception as e:
            st.error(f"Error making batch predictions: {str(e)}")
            return None
    
    def display_batch_results(self, predictions_df):
        """Display batch prediction results"""
        st.markdown("### 🎯 Batch Prediction Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_predictions = len(predictions_df)
        low_risk = len(predictions_df[predictions_df['Prediction'] == 0])
        moderate_risk = len(predictions_df[predictions_df['Prediction'] == 1])
        high_risk = len(predictions_df[predictions_df['Prediction'] == 2])
        
        with col1:
            self.ui.create_metric_card("Total Patients", f"{total_predictions:,}")
        with col2:
            self.ui.create_metric_card("Low Risk", f"{low_risk:,}")
        with col3:
            self.ui.create_metric_card("Moderate Risk", f"{moderate_risk:,}")
        with col4:
            self.ui.create_metric_card("High Risk", f"{high_risk:,}")
        
        # Risk distribution chart
        risk_counts = predictions_df['Risk_Level'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Distribution",
                color_discrete_map={
                    'Low Risk': 'green',
                    'Moderate Risk': 'orange',
                    'High Risk': 'red'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Risk Counts",
                color=risk_counts.index,
                color_discrete_map={
                    'Low Risk': 'green',
                    'Moderate Risk': 'orange',
                    'High Risk': 'red'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Results table
        st.markdown("#### 📋 Detailed Results")
        st.dataframe(predictions_df, use_container_width=True)
        
        # Download results
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
    
    def render_prediction_analysis(self):
        """Render prediction analysis interface"""
        st.markdown("### 📈 Prediction Analysis")
        
        if not st.session_state.get('prediction_history'):
            st.info("No prediction history available.")
            return
        
        # Filter predictions
        prediction_types = st.multiselect(
            "Filter by prediction type:",
            ["single", "batch"],
            default=["single", "batch"]
        )
        
        filtered_history = [
            p for p in st.session_state.prediction_history 
            if p['type'] in prediction_types
        ]
        
        if not filtered_history:
            st.info("No predictions match the selected filters.")
            return
        
        # Summary statistics
        st.markdown("#### 📊 Prediction Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self.ui.create_metric_card("Total Predictions", f"{len(filtered_history):,}")
        
        with col2:
            single_count = len([p for p in filtered_history if p['type'] == 'single'])
            self.ui.create_metric_card("Single Predictions", f"{single_count:,}")
        
        with col3:
            batch_count = len([p for p in filtered_history if p['type'] == 'batch'])
            self.ui.create_metric_card("Batch Predictions", f"{batch_count:,}")
        
        # Prediction timeline
        if len(filtered_history) > 1:
            st.markdown("#### 📈 Prediction Timeline")
            
            timeline_data = []
            for pred in filtered_history:
                timeline_data.append({
                    'timestamp': pred['timestamp'],
                    'type': pred['type'],
                    'model': pred['model_name']
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])
            timeline_df = timeline_df.sort_values('timestamp')
            
            fig = px.scatter(
                timeline_df,
                x='timestamp',
                y='type',
                color='model',
                title="Prediction Activity Timeline"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent predictions
        st.markdown("#### 🕒 Recent Predictions")
        
        recent_predictions = sorted(
            filtered_history, 
            key=lambda x: x['timestamp'], 
            reverse=True
        )[:10]
        
        for pred in recent_predictions:
            with st.expander(f"{pred['type'].title()} - {pred['model_name']} - {pred['timestamp'][:19]}"):
                if pred['type'] == 'single':
                    st.write(f"**Prediction:** {pred['prediction']}")
                    if pred['probabilities']:
                        st.write(f"**Probabilities:** {[f'{p:.3f}' for p in pred['probabilities']]}")
                else:
                    st.write("**Batch prediction results stored**")
    
    def render_model_inference(self):
        """Render model inference and comparison interface"""
        st.markdown("### 🔄 Model Inference & Comparison")
        
        # Model comparison
        available_models = list(st.session_state.trained_models.keys())
        
        if len(available_models) < 2:
            st.info("Need at least 2 models for comparison.")
            return
        
        selected_models = st.multiselect(
            "Select models to compare:",
            available_models,
            default=available_models[:2]
        )
        
        if not selected_models:
            return
        
        st.markdown("#### 🔍 Model Comparison")
        
        # Create sample input for comparison
        if st.session_state.current_dataset is not None:
            sample_data = st.session_state.current_dataset.sample(1).iloc[0]
            
            st.markdown("**Sample Patient Data:**")
            st.json(sample_data.to_dict())
            
            if st.button("🔄 Compare Model Predictions"):
                comparison_results = []
                
                for model_name in selected_models:
                    model_info = st.session_state.trained_models[model_name]
                    
                    try:
                        # Prepare sample data
                        features = model_info.get('features', [])
                        input_data = {feature: sample_data.get(feature, 0) for feature in features}
                        input_df = pd.DataFrame([input_data])
                        
                        # Apply scaling if needed
                        if 'scaler' in model_info and model_info['scaler'] is not None:
                            input_df = pd.DataFrame(
                                model_info['scaler'].transform(input_df),
                                columns=input_df.columns
                            )
                        
                        # Make prediction
                        model = model_info['model']
                        prediction = model.predict(input_df)[0]
                        
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(input_df)[0]
                        else:
                            probabilities = None
                        
                        comparison_results.append({
                            'Model': model_name,
                            'Prediction': int(prediction),
                            'Confidence': f"{max(probabilities):.3f}" if probabilities is not None else "N/A"
                        })
                        
                    except Exception as e:
                        st.error(f"Error with model {model_name}: {str(e)}")
                
                # Display comparison
                if comparison_results:
                    comparison_df = pd.DataFrame(comparison_results)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visualization
                    if len(comparison_results) > 1:
                        fig = px.bar(
                            comparison_df,
                            x='Model',
                            y='Prediction',
                            title="Model Prediction Comparison"
                        )
                        st.plotly_chart(fig, use_container_width=True)
