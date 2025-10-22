"""
🏥 DIABETICSAI ENTERPRISE - MAIN APPLICATION
Advanced Medical AI Platform with Professional Dashboard Analytics
"""

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", message=".*version.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*Arrow.*")
warnings.filterwarnings("ignore", message=".*Serialization.*")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Import all modules
from modules.dashboard import DashboardManager
from modules.model_trainer import ModelTrainer
from modules.data_manager import DataManager
from modules.analytics import AnalyticsEngine
from modules.predictions import PredictionEngine
from modules.ui_components import UIComponents
from modules.config import AppConfig
# Debug tools are now integrated directly into the main app class

# Configure Streamlit page
st.set_page_config(
    page_title="🏥 DiabeticsAI Enterprise - Medical AI Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/diabeticsai/help',
        'Report a bug': 'https://github.com/diabeticsai/issues',
        'About': '''
        # 🏥 DiabeticsAI Enterprise
        ## Professional Medical AI Analytics Platform
        
        **⚡ Enterprise Features:**
        - 🎯 Advanced ML Model Training & Management
        - 📊 Professional Dashboard Analytics
        - 📁 Dataset Upload & Management
        - 🔬 Model Selection & Parameter Tuning
        - 📈 Real-time Performance Monitoring
        - 💾 Model Repository & Export
        
        **Built for medical professionals, researchers, and healthcare organizations.**
        '''
    }
)

class DiabeticsAIEnterprise:
    """Main Enterprise Application Class"""
    
    def __init__(self):
        self.config = AppConfig()
        self.ui = UIComponents()
        self.data_manager = DataManager()
        self.model_trainer = ModelTrainer()
        self.analytics = AnalyticsEngine()
        self.predictions = PredictionEngine()
        self.dashboard = DashboardManager()
        
        # Initialize session state
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'current_dataset' not in st.session_state:
            st.session_state.current_dataset = None
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        if 'model_history' not in st.session_state:
            st.session_state.model_history = []
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
    
    def render_sidebar(self):
        """Render professional sidebar navigation"""
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <h1 style="color: #1f77b4; font-size: 2rem; margin: 0;">🏥 DiabeticsAI</h1>
                <p style="color: #666; margin: 0.5rem 0;">Enterprise Medical AI</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation menu
            st.markdown("### 📋 Navigation")
            page = st.selectbox(
                "Select Module:",
                ["🏠 Dashboard", "📊 Analytics", "🧠 Model Training", 
                 "🔮 Predictions", "📁 Data Management", "⚙️ Settings", "🛠️ UI Debug Tools"],
                label_visibility="collapsed"
            )
            
            # Add debug mode toggle
            if st.checkbox("🔍 Enable UI Inspector", help="Activate real-time UI element inspector"):
                self.inject_simple_inspector()
            
            # System status
            st.markdown("---")
            st.markdown("### 📊 System Status")
            
            # Dataset status
            dataset_status = "🟢 Loaded" if st.session_state.current_dataset is not None else "🔴 None"
            st.metric("Dataset", dataset_status)
            
            # Models status
            models_count = len(st.session_state.trained_models)
            st.metric("Trained Models", models_count)
            
            # Memory usage (simulated)
            st.metric("Memory Usage", "234 MB")
            
            return page
    
    def run(self):
        """Main application runner"""
        # Apply custom CSS
        self.ui.apply_enterprise_css()
        
        # Render sidebar and get selected page
        selected_page = self.render_sidebar()
        
        # Route to appropriate module
        if selected_page == "🏠 Dashboard":
            self.dashboard.render()
            
        elif selected_page == "📊 Analytics":
            self.analytics.render()
            
        elif selected_page == "🧠 Model Training":
            self.model_trainer.render()
            
        elif selected_page == "🔮 Predictions":
            self.predictions.render()
            
        elif selected_page == "📁 Data Management":
            self.data_manager.render()
            
        elif selected_page == "⚙️ Settings":
            self.render_settings()
            
        elif selected_page == "🛠️ UI Debug Tools":
            self.render_debug_tools()
    
    def render_settings(self):
        """Render application settings"""
        st.markdown("# ⚙️ System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎨 Interface Settings")
            theme = st.selectbox("Theme", ["Professional", "Dark", "Light"])
            animation = st.checkbox("Enable Animations", value=True)
            
            st.markdown("### 🔧 Model Settings")
            default_test_size = st.slider("Default Test Size", 0.1, 0.4, 0.2)
            auto_save = st.checkbox("Auto-save Models", value=True)
        
        with col2:
            st.markdown("### 📊 Analytics Settings")
            chart_style = st.selectbox("Chart Style", ["Professional", "Colorful", "Minimal"])
            show_confidence = st.checkbox("Show Confidence Intervals", value=True)
            
            st.markdown("### 💾 Data Settings")
            max_history = st.slider("Max History Records", 50, 500, 100)
            cache_timeout = st.slider("Cache Timeout (minutes)", 5, 60, 15)
    
    def render_debug_tools(self):
        """Render UI debug tools"""
        st.markdown("# 🛠️ UI Debug Tools")
        st.markdown("Professional tools for customizing and debugging the user interface.")
        
        # Tab layout for different debug tools
        tab1, tab2, tab3 = st.tabs(["🎨 Color Inspector", "🔍 CSS Finder", "🛠️ Debug Panel"])
        
        with tab1:
            st.markdown("### 🎨 Advanced Color Inspector")
            st.info("💡 **Tip:** Enable 'UI Inspector' in the sidebar to get real-time element inspection with a floating inspector panel.")
            
            # Show current color palette
            st.markdown("#### Current Application Color Palette")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Primary Colors:**")
                st.color_picker("Primary Cyan", "#1BD3CD", disabled=True)
                st.color_picker("Dark Cyan", "#09757A", disabled=True)
                
            with col2:
                st.markdown("**Background Colors:**")
                st.color_picker("Main Background", "#020024", disabled=True)
                st.color_picker("White Overlay", "#FFFFFF", disabled=True)
                
            with col3:
                st.markdown("**Text Colors:**")
                st.color_picker("White Text", "#FFFFFF", disabled=True)
                st.color_picker("Black Text", "#000000", disabled=True)
        
        with tab2:
            self.render_css_finder()
        
        with tab3:
            self.render_advanced_debug_panel()
        
        # Instructions
        st.markdown("---")
        st.markdown("""
        ### 📖 How to Use the Debug Tools
        
        1. **🔍 UI Inspector (Sidebar)**: 
           - Enable the checkbox in sidebar to activate floating inspector
           - Click the 🔍 button that appears in top-right
           - Hover over any element to highlight it
           - Click to inspect and get CSS details
           - Use the color pickers to modify colors in real-time
           - Generate CSS code for your changes
        
        2. **🎯 CSS Rule Finder (Tab 2)**:
           - Search for specific CSS selectors by element name
           - Get exact CSS rules for Streamlit components
           - Copy sample CSS code for customization
        
        3. **🛠️ Debug Panel (Tab 3)**:
           - Advanced debugging tools
           - Theme generator
           - CSS export functionality
           - Color palette management
        
        ### 💡 Pro Tips:
        - Use browser developer tools alongside these tools for advanced debugging
        - Test color changes with the real-time inspector before applying them permanently
        - Export your custom CSS when you find a design you like
        - The floating inspector shows computed styles, which are the actual applied styles
        """)
    
    def render_css_finder(self):
        """Render CSS rule finder"""
        st.markdown("### 🎯 CSS Rule Finder")
        
        # Search input
        search_query = st.text_input(
            "Search for CSS rules:",
            placeholder="Enter element name, class, or property (e.g., button, stTextInput, background-color)"
        )
        
        if search_query:
            # Common Streamlit CSS selectors
            streamlit_selectors = {
                "Main App": {
                    "selector": ".stApp",
                    "common_props": ["background", "color", "font-family"],
                    "description": "Main application container"
                },
                "Sidebar": {
                    "selector": "[data-testid='stSidebar']",
                    "common_props": ["background", "color", "width"],
                    "description": "Application sidebar"
                },
                "Text Input": {
                    "selector": ".stTextInput > div > div > input",
                    "common_props": ["background", "color", "border", "border-radius"],
                    "description": "Text input fields"
                },
                "Button Primary": {
                    "selector": ".stButton > button[kind='primary']",
                    "common_props": ["background", "color", "border"],
                    "description": "Primary buttons"
                },
                "Tab Button": {
                    "selector": ".stTabs [data-baseweb='tab-list'] button",
                    "common_props": ["background", "color", "border"],
                    "description": "Individual tab buttons"
                },
                "Active Tab": {
                    "selector": ".stTabs [data-baseweb='tab-list'] button[aria-selected='true']",
                    "common_props": ["background", "color", "border"],
                    "description": "Currently selected tab"
                },
                "Metric Container": {
                    "selector": "[data-testid='metric-container']",
                    "common_props": ["background", "color", "border", "padding"],
                    "description": "Metric display containers"
                },
                "DataFrame": {
                    "selector": ".stDataFrame",
                    "common_props": ["background", "color", "border"],
                    "description": "Data tables"
                }
            }
            
            # Filter selectors based on search
            matches = []
            for name, info in streamlit_selectors.items():
                if (search_query.lower() in name.lower() or 
                    search_query.lower() in info['selector'].lower() or
                    any(search_query.lower() in prop.lower() for prop in info['common_props'])):
                    matches.append((name, info))
            
            if matches:
                st.markdown(f"#### Found {len(matches)} matches:")
                
                for name, info in matches:
                    with st.expander(f"🎯 {name}"):
                        st.markdown(f"**Description:** {info['description']}")
                        st.code(info['selector'], language="css")
                        
                        st.markdown("**Common Properties:**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            for prop in info['common_props']:
                                st.markdown(f"• `{prop}`")
                        
                        with col2:
                            # Generate sample CSS
                            sample_css = f"""{info['selector']} {{
    {info['common_props'][0]}: #1BD3CD !important;
    {info['common_props'][1] if len(info['common_props']) > 1 else 'color'}: #ffffff !important;
}}"""
                            st.code(sample_css, language="css")
            else:
                st.warning("No matches found. Try searching for 'button', 'input', 'tab', etc.")
    
    def render_advanced_debug_panel(self):
        """Render advanced debug panel"""
        st.markdown("### 🛠️ Advanced Debug Panel")
        
        # Color palette editor
        st.markdown("#### 🎨 Theme Generator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            primary_color = st.color_picker("Primary Color", "#1BD3CD")
            background_color = st.color_picker("Background Color", "#020024")
            
        with col2:
            text_color = st.color_picker("Text Color", "#FFFFFF")
            accent_color = st.color_picker("Accent Color", "#6cc5eb")
        
        # Generate theme CSS
        if st.button("🎭 Generate Custom Theme"):
            theme_css = f"""
/* Custom Generated Theme */
:root {{
    --primary-color: {primary_color};
    --background-color: {background_color};
    --text-color: {text_color};
    --accent-color: {accent_color};
}}

/* Main App */
.stApp {{
    background: var(--background-color) !important;
    background: radial-gradient(circle, {background_color}, {accent_color}) !important;
    color: var(--text-color) !important;
}}

/* Buttons */
.stButton > button[kind='primary'] {{
    background: var(--primary-color) !important;
    color: var(--text-color) !important;
    border: 2px solid var(--primary-color) !important;
    border-radius: 12px !important;
}}

/* Text Inputs */
.stTextInput > div > div > input {{
    background: rgba(255, 255, 255, 0.95) !important;
    color: #000000 !important;
    border: 2px solid var(--accent-color) !important;
    border-radius: 12px !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 12px !important;
}}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
    background: var(--primary-color) !important;
    color: var(--text-color) !important;
}}
"""
            
            st.markdown("#### Generated CSS:")
            st.code(theme_css, language="css")
            
            # Apply theme button
            if st.button("🚀 Apply Theme (Live Preview)"):
                st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)
                st.success("✅ Theme applied! Refresh to see full effects.")
        
        # CSS Export
        st.markdown("---")
        st.markdown("#### 📤 Export Tools")
        
        if st.button("📋 Copy Current CSS"):
            current_css = """
/* Current DiabeticsAI Enterprise CSS */
.stApp {
    background: radial-gradient(circle, rgba(2,0,36,1) 0%, rgba(9,9,121,1) 35%, rgba(2,154,217,1) 84%, rgba(0,212,255,1) 100%) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 12px !important;
}

.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.95) !important;
    color: #000000 !important;
    border-radius: 12px !important;
}
"""
            st.code(current_css, language="css")
            st.info("💡 Copy this CSS and modify as needed!")
    
    def inject_simple_inspector(self):
        """Inject a simple working UI inspector"""
        inspector_html = """
        <div id="simple-inspector" style="
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 9999;
            background: #1BD3CD;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 12px;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            user-select: none;
        ">🔍 Click to Inspect Elements</div>
        
        <div id="inspector-panel" style="
            position: fixed;
            top: 50px;
            right: 10px;
            width: 300px;
            max-height: 400px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 8px;
            padding: 15px;
            z-index: 9998;
            font-family: monospace;
            font-size: 11px;
            overflow-y: auto;
            display: none;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5);
        ">
            <div style="border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 10px;">
                <strong style="color: #1BD3CD;">🔍 Element Inspector</strong>
                <button onclick="document.getElementById('inspector-panel').style.display='none'" 
                        style="float: right; background: #ff4444; color: white; border: none; padding: 2px 6px; border-radius: 3px; cursor: pointer;">×</button>
            </div>
            <div id="element-details">Click on any element to inspect it</div>
        </div>
        
        <script>
        (function() {
            let inspectorActive = false;
            
            document.getElementById('simple-inspector').onclick = function() {
                inspectorActive = !inspectorActive;
                const panel = document.getElementById('inspector-panel');
                const button = document.getElementById('simple-inspector');
                
                if (inspectorActive) {
                    panel.style.display = 'block';
                    button.innerHTML = '🔍 Inspecting... (Click elements)';
                    button.style.background = '#ff4444';
                    document.body.style.cursor = 'crosshair';
                    
                    // Add click handler to all elements
                    document.addEventListener('click', inspectElement);
                } else {
                    panel.style.display = 'none';
                    button.innerHTML = '🔍 Click to Inspect Elements';
                    button.style.background = '#1BD3CD';
                    document.body.style.cursor = 'default';
                    
                    // Remove click handler
                    document.removeEventListener('click', inspectElement);
                }
            };
            
            function inspectElement(e) {
                if (e.target.closest('#simple-inspector') || e.target.closest('#inspector-panel')) {
                    return;
                }
                
                e.preventDefault();
                e.stopPropagation();
                
                const element = e.target;
                const computedStyle = window.getComputedStyle(element);
                
                // Get element information
                const tagName = element.tagName.toLowerCase();
                const className = element.className || 'none';
                const id = element.id || 'none';
                const backgroundColor = computedStyle.backgroundColor;
                const color = computedStyle.color;
                const border = computedStyle.border;
                
                // Find data attributes
                const dataAttrs = Array.from(element.attributes)
                    .filter(attr => attr.name.startsWith('data-'))
                    .map(attr => `${attr.name}="${attr.value}"`)
                    .join('<br>') || 'none';
                
                // Generate CSS selector
                let selector = tagName;
                if (id !== 'none') selector += '#' + id;
                if (className !== 'none') {
                    const classes = className.split(' ').filter(c => c.trim()).map(c => '.' + c).join('');
                    selector += classes;
                }
                
                // Check for Streamlit-specific selectors
                const testId = element.getAttribute('data-testid');
                if (testId) {
                    selector = `[data-testid="${testId}"]`;
                }
                
                const details = `
                    <h4 style="color: #1BD3CD; margin: 10px 0 5px 0;">Element Info</h4>
                    <p><strong>Tag:</strong> ${tagName}</p>
                    <p><strong>Classes:</strong> ${className}</p>
                    <p><strong>ID:</strong> ${id}</p>
                    <p><strong>Selector:</strong> <code style="background: #333; padding: 2px 4px; border-radius: 3px;">${selector}</code></p>
                    
                    <h4 style="color: #1BD3CD; margin: 15px 0 5px 0;">Styling</h4>
                    <p><strong>Background:</strong> <span style="background: ${backgroundColor}; padding: 2px 6px; border-radius: 3px; color: black;">${backgroundColor}</span></p>
                    <p><strong>Text Color:</strong> <span style="color: ${color};">${color}</span></p>
                    <p><strong>Border:</strong> ${border}</p>
                    
                    <h4 style="color: #1BD3CD; margin: 15px 0 5px 0;">Sample CSS</h4>
                    <textarea readonly style="width: 100%; height: 80px; background: #222; color: #1BD3CD; border: 1px solid #555; padding: 5px; font-family: monospace; font-size: 10px; border-radius: 3px;">${selector} {
    background: ${backgroundColor} !important;
    color: ${color} !important;
    border: ${border} !important;
}</textarea>
                    
                    <h4 style="color: #1BD3CD; margin: 15px 0 5px 0;">Data Attributes</h4>
                    <div style="font-size: 10px; color: #ccc;">${dataAttrs}</div>
                `;
                
                document.getElementById('element-details').innerHTML = details;
            }
        })();
        </script>
        """
        
        # Use st.components to inject the HTML
        import streamlit.components.v1 as components
        components.html(inspector_html, height=0)

def main():
    """Application entry point"""
    try:
        app = DiabeticsAIEnterprise()
        app.run()
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.info("🔄 Please refresh the page to restart the application.")

if __name__ == "__main__":
    main()
