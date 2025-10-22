"""
🔬 ANALYTICS ENGINE - UNIFIED PROFESSIONAL INTERFACE
Analytics and visualization engine with comprehensive explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.ui_components import UIComponents
from modules.config import AppConfig


class AnalyticsEngine:
    """Unified Professional Analytics Engine"""
    
    def __init__(self):
        self.ui = UIComponents()
        self.config = AppConfig()
    
    def render(self):
        """Render unified analytics interface"""
        self.ui.apply_enterprise_css()
        
        self.ui.create_header(
            "📊 Professional Analytics Dashboard",
            "Comprehensive data insights and visualizations"
        )
        
        if st.session_state.current_dataset is None:
            self.ui.create_alert("⚠️ No dataset loaded. Please load a dataset first.", "warning")
            return
        
        df = st.session_state.current_dataset
        
        # Create unified analytics tabs
        tab1, tab2 = st.tabs([
            "📊 Data Overview", "🔍 Visual Analytics"
        ])
        
        with tab1:
            self.render_unified_overview(df)
        
        with tab2:
            self.render_visual_analytics(df)
    
    def render_unified_overview(self, df):
        """Render unified data overview with Arrow compatibility"""
        st.markdown("### 📊 Dataset Overview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.ui.create_metric_card("Total Records", f"{len(df):,}")
        with col2:
            self.ui.create_metric_card("Features", f"{len(df.columns):,}")
        with col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            self.ui.create_metric_card("Numeric Features", f"{numeric_cols}")
        with col4:
            missing_count = df.isnull().sum().sum()
            self.ui.create_metric_card("Missing Values", f"{missing_count:,}")
        
        st.markdown("---")
        
        # Data quality overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 Data Quality")
            
            # Create a simplified data info table with Arrow compatibility
            data_info = []
            for col in df.columns:
                col_type = str(df[col].dtype).replace('64', '').replace('object', 'text')
                unique_vals = df[col].nunique()
                missing_vals = df[col].isnull().sum()
                missing_pct = (missing_vals / len(df) * 100)
                
                data_info.append({
                    "Column": str(col)[:20] + "..." if len(str(col)) > 20 else str(col),
                    "Type": col_type,
                    "Unique": unique_vals,
                    "Missing": missing_vals,
                    "Missing %": f"{missing_pct:.1f}%"
                })
            
            # Convert to DataFrame without forcing problematic types
            info_df = pd.DataFrame(data_info)
            
            # Safe display with Arrow compatibility
            self.config.safe_dataframe_display(info_df, width="stretch")
        
        with col2:
            st.markdown("#### 📈 Statistical Summary")
            
            # Get numeric columns only for statistical summary
            numeric_df = df.select_dtypes(include=[np.number])
            
            if not numeric_df.empty:
                # Create simplified stats with proper formatting
                stats_summary = numeric_df.describe().round(2)
                # Reset index to make stats row names a column
                stats_summary = stats_summary.reset_index()
                stats_summary.columns = ['Statistic'] + [str(col)[:15] + "..." if len(str(col)) > 15 else str(col) for col in stats_summary.columns[1:]]
                
                self.config.safe_dataframe_display(stats_summary, width="stretch")
            else:
                st.info("No numeric columns found for statistical summary")
        
        # Target distribution (if exists)
        if self.config.TARGET_COLUMN in df.columns:
            st.markdown("---")
            st.markdown("#### 🎯 Target Variable Analysis")
            
            target_data = df[self.config.TARGET_COLUMN].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Target Distribution (Pie Chart)")
                st.info("📊 **What this shows:** This pie chart displays the proportion of each diabetes risk category in your dataset. It helps you understand the class balance - whether you have equal representation of all risk levels or if some categories are more common than others.")
                
                fig = px.pie(
                    values=target_data.values,
                    names=[str(name) for name in target_data.index],
                    title="Target Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                st.markdown("#### 📊 Target Counts (Bar Chart)")
                st.info("📊 **What this shows:** This bar chart shows the exact number of records in each diabetes risk category. Compare the heights to see which categories have more or fewer samples - this is crucial for understanding potential model bias.")
                
                fig = px.bar(
                    x=[str(name) for name in target_data.index],
                    y=target_data.values,
                    title="Target Counts",
                    labels={'x': 'Class', 'y': 'Count'},
                    color=target_data.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")
    
    def render_visual_analytics(self, df):
        """Render unified visual analytics"""
        st.markdown("### 🔍 Visual Data Analysis")
        
        # Get numeric and categorical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not numeric_columns:
            st.warning("No numeric columns found for visualization.")
            return
        
        # Visualization selection
        col1, col2 = st.columns(2)
        
        with col1:
            viz_type = st.selectbox(
                "📊 Select Visualization:",
                ["Distribution Analysis", "Correlation Analysis", "Relationship Analysis", "Missing Values Pattern"]
            )
        
        with col2:
            if viz_type in ["Distribution Analysis", "Relationship Analysis"]:
                selected_column = st.selectbox("Select Column:", numeric_columns)
        
        st.markdown("---")
        
        if viz_type == "Distribution Analysis" and selected_column:
            st.markdown(f"#### 📈 Distribution Analysis for '{selected_column}'")
            st.info("📊 **Purpose:** Understanding how your data is distributed helps identify patterns, outliers, and data quality issues that could affect model performance.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📉 Histogram with Box Plot**")
                st.caption("🔍 **What this shows:** The histogram reveals the frequency distribution of values, while the box plot above shows quartiles and outliers. Look for normal distribution, skewness, or multiple peaks.")
                
                # Histogram
                fig = px.histogram(
                    df, x=selected_column,
                    title=f"Distribution of {selected_column}",
                    marginal="box",
                    color_discrete_sequence=['#3498db']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                st.markdown("**📦 Detailed Box Plot**")
                st.caption("🔍 **What this shows:** Box plots highlight the median, quartiles (25%, 75%), and outliers. The 'box' contains 50% of your data, while points outside the whiskers are potential outliers that may need attention.")
                
                # Box plot
                fig = px.box(
                    df, y=selected_column,
                    title=f"Box Plot of {selected_column}",
                    color_discrete_sequence=['#e74c3c']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")
        
        elif viz_type == "Correlation Analysis":
            st.markdown("#### 🌡️ Correlation Analysis")
            st.info("📊 **Purpose:** Correlation analysis reveals relationships between features. Strong correlations (close to +1 or -1) indicate features that move together, which can affect model performance and feature selection.")
            
            # Correlation heatmap
            if len(numeric_columns) > 1:
                st.markdown("**🗒️ Correlation Heatmap**")
                st.caption("🔍 **How to read:** Red colors show negative correlations, blue shows positive. Darker colors mean stronger correlations. Values near +1/-1 are strong, near 0 are weak.")
                
                corr_matrix = df[numeric_columns].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu',
                    aspect="auto"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, width="stretch")
                
                # Top correlations
                st.markdown("#### 🔗 Strongest Correlations")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': abs(corr_matrix.iloc[i, j])
                        })
                
                corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False).head(10)
                self.config.safe_dataframe_display(corr_df, width="stretch")
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
        
        elif viz_type == "Relationship Analysis" and selected_column:
            st.markdown(f"#### 🔗 Relationship Analysis: '{selected_column}' vs Target")
            st.info("📊 **Purpose:** Understanding how each feature relates to the target variable helps identify which features are most predictive for diabetes risk classification.")
            
            if self.config.TARGET_COLUMN in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎯 Scatter Plot with Marginal Distributions**")
                    st.caption("🔍 **What this shows:** Each point represents a patient. Colors show different diabetes risk levels. Marginal histograms show distributions. Look for clear separation between colors.")
                    
                    # Scatter plot with target
                    fig = px.scatter(
                        df, x=selected_column, y=self.config.TARGET_COLUMN,
                        title=f"{selected_column} vs Target",
                        color=self.config.TARGET_COLUMN,
                        marginal_x="histogram",
                        marginal_y="histogram"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width="stretch")
                
                with col2:
                    st.markdown("**📦 Distribution by Risk Category**")
                    st.caption("🔍 **What this shows:** Compare how the selected feature values differ across diabetes risk categories. Different box positions indicate the feature helps distinguish between risk levels.")
                    
                    # Box plot by target
                    fig = px.box(
                        df, x=self.config.TARGET_COLUMN, y=selected_column,
                        title=f"{selected_column} Distribution by Target",
                        color=self.config.TARGET_COLUMN
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Target column not found for relationship analysis.")
        
        elif viz_type == "Missing Values Pattern":
            st.markdown("#### 🕳️ Missing Values Pattern Analysis")
            st.info("📊 **Purpose:** Understanding missing data patterns is crucial for data quality assessment and choosing appropriate preprocessing strategies.")
            
            # Missing values heatmap
            missing_data = df.isnull()
            
            if missing_data.sum().sum() > 0:
                st.markdown("**🔥 Missing Values Heatmap**")
                st.caption("🔍 **How to read:** Red areas indicate missing values, white areas indicate present data. Vertical patterns suggest systematic missing data that may need special handling.")
                
                fig = px.imshow(
                    missing_data.astype(int),
                    title="Missing Values Pattern",
                    color_continuous_scale=['white', 'red'],
                    aspect="auto"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")
                
                # Missing values summary
                missing_summary = pd.DataFrame({
                    'Column': df.columns,
                    'Missing Count': df.isnull().sum(),
                    'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
                }).sort_values('Missing Count', ascending=False)
                
                missing_summary = missing_summary[missing_summary['Missing Count'] > 0]
                if not missing_summary.empty:
                    self.config.safe_dataframe_display(missing_summary, width="stretch")
                else:
                    st.success("✅ No missing values found in the dataset!")
            else:
                st.success("✅ No missing values found in the dataset!")