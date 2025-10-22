"""
🎨 UI COMPONENTS MODULE
Professional UI components and styling for DiabeticsAI Enterprise
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

class UIComponents:
    """Professional UI Components Class"""
    
    def get_base64_of_bin_file(self, bin_file):
        """Convert binary file to base64 string"""
        import base64
        try:
            with open(bin_file, 'rb') as f:
                data = f.read()
            return base64.b64encode(data).decode()
        except Exception as e:
            print(f"Error encoding file {bin_file}: {e}")
            return None
    
    def apply_enterprise_css(self):
        """Apply world-class UI/UX with premium gradient background and modern design"""
        
        # Premium gradient background - no image loading needed
        background_css = f"""
        [data-testid="stAppViewContainer"] {{
            background: #020024;
            background: radial-gradient(circle, rgba(2,0,36,1) 0%, rgba(9,9,121,1) 35%, rgba(2,154,217,1) 84%, rgba(0,212,255,1) 100%);
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        
        [data-testid="stAppViewContainer"] > div:first-child {{
            background: transparent;
        }}
        
        /* Premium gradient background */
        [data-testid="stAppViewContainer"] {{
            background: #020024;
            background: radial-gradient(circle, rgba(2,0,36,1) 0%, rgba(9,9,121,1) 35%, rgba(2,154,217,1) 84%, rgba(0,212,255,1) 100%);
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        
        [data-testid="stHeader"] {{
            background: rgba(2, 0, 36, 0.8);
            backdrop-filter: blur(15px);
        }}
        
        .main .block-container {{
            position: relative;
            z-index: 1;
        }}"""
        
        print("✅ Premium gradient background configured")
        
        css_content = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Premium Radial Gradient Background */
        {background_css}
        
        .stApp {{
            font-family: 'Poppins', sans-serif;
            position: relative;
        }}
        
        /* Ensure content is visible above background */
        [data-testid="stAppViewContainer"] {{
            position: relative;
        }}
        
        section[data-testid="stSidebar"],
        .main,
        [data-testid="stHeader"] {{
            position: relative;
            z-index: 999;
        }}
        
        /* Glassmorphism Effect */
        .glass-card {{
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
        }}
        /* Floating Header with Glassmorphism */
        .enterprise-header {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(30px) saturate(180%);
            -webkit-backdrop-filter: blur(30px) saturate(180%);
            padding: 3rem 2.5rem;
            border-radius: 24px;
            margin-bottom: 2.5rem;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.1) inset;
            border: 1px solid rgba(255, 255, 255, 0.18);
            position: relative;
            overflow: hidden;
            animation: floatHeader 6s ease-in-out infinite;
        }}
        .enterprise-header::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.05), transparent);
            transform: rotate(45deg);
            animation: shimmer 3s infinite;
        }}
        .enterprise-header h1 {{
            color: #ffffff;
            font-family: 'Poppins', sans-serif;
            font-weight: 800;
            font-size: 3.2rem;
            margin: 0;
            text-shadow: 0 4px 12px rgba(0, 0, 0, 0.5), 0 0 30px rgba(138, 180, 248, 0.3);
            letter-spacing: -0.5px;
            position: relative;
            z-index: 1;
        }}
        .enterprise-header p {{
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.3rem;
            font-weight: 400;
            margin: 0.8rem 0 0 0;
            position: relative;
            z-index: 1;
            letter-spacing: 0.3px;
        }}
        @keyframes floatHeader {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-8px); }}
        }}
        @keyframes shimmer {{
            0% {{ transform: translateX(-100%) rotate(45deg); }}
            100% {{ transform: translateX(100%) rotate(45deg); }}
        }}
        /* Floating Metric Cards with Glassmorphism */
        .metric-card {{
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(25px) saturate(180%);
            -webkit-backdrop-filter: blur(25px) saturate(180%);
            color: #fff;
            padding: 2.5rem 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(255, 255, 255, 0.12) inset;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-left: 5px solid transparent;
            border-image: linear-gradient(180deg, #1BD3CD 0%, #097579 100%) 1;
            margin: 1rem 0;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
        }}
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.5s;
        }}
        .metric-card:hover {{
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(255, 255, 255, 0.18) inset;
            border-image: linear-gradient(180deg, #00D4FF 0%, #1BD3CD 100%) 1;
        }}
        .metric-card:hover::before {{
            left: 100%;
        }}
        .metric-value {{
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00D4FF 0%, #1BD3CD 50%, #097579 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            font-family: 'Poppins', sans-serif;
        }}
        .metric-label {{
            font-size: 1.1rem;
            color: rgba(255, 255, 255, 0.8);
            margin: 0.8rem 0 0 0;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: 500;
        }}
        .status-good {{
            color: #00e676;
            background: rgba(0, 230, 118, 0.12);
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: 600;
        }}
        .status-warning {{
            color: #ffb300;
            background: rgba(255, 179, 0, 0.12);
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: 600;
        }}
        .status-danger {{
            color: #e53935;
            background: rgba(229, 57, 53, 0.12);
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: 600;
        }}
        .progress-container {{
            background: #333;
            border-radius: 10px;
            overflow: hidden;
            height: 12px;
            margin: 1rem 0;
        }}
        .progress-bar {{
            height: 100%;
            background: linear-gradient(90deg, #097579, #1BD3CD);
            transition: width 0.3s ease;
        }}
        /* Premium Floating Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, #1BD3CD 0%, #097579 100%);
            color: #000000;
            border: none;
            border-radius: 16px;
            padding: 1rem 2.5rem;
            font-weight: 600;
            font-size: 1.05rem;
            font-family: 'Poppins', sans-serif;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 8px 24px rgba(27, 211, 205, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.1) inset;
            position: relative;
            overflow: hidden;
            letter-spacing: 0.5px;
        }}
        .stButton > button::before {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }}
        .stButton > button:hover {{
            transform: translateY(-4px) scale(1.05);
            box-shadow: 0 16px 40px rgba(27, 211, 205, 0.5), 0 0 30px rgba(0, 212, 255, 0.3);
            background: linear-gradient(135deg, #00D4FF 0%, #1BD3CD 100%);
        }}
        .stButton > button:hover::before {{
            width: 300px;
            height: 300px;
        }}
        .stButton > button:active {{
            transform: translateY(-2px) scale(1.02);
        }}
        /* Floating Tables with Glassmorphism */
        .dataframe {{
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }}
        
        /* Floating Carousel Container */
        .carousel-container {{
            background: #6cc5eb;
            background: radial-gradient(circle, rgba(108, 197, 235, 1) 0%, rgba(70, 70, 212, 1) 35%, rgba(0, 0, 212, 1) 61%, rgba(3, 0, 56, 1) 100%);
            backdrop-filter: blur(30px) saturate(180%);
            -webkit-backdrop-filter: blur(30px) saturate(180%);
            border-radius: 24px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(27, 211, 205, 0.3) inset;
            border: 1px solid rgba(27, 211, 205, 0.4);
            position: relative;
            animation: floatCarousel 8s ease-in-out infinite;
            color: #000000;
        }}
        @keyframes floatCarousel {{
            0%, 100% {{ transform: translateY(0px) translateX(0px); }}
            25% {{ transform: translateY(-10px) translateX(5px); }}
            50% {{ transform: translateY(-5px) translateX(-5px); }}
            75% {{ transform: translateY(-12px) translateX(3px); }}
        }}
        /* Floating Alerts with Glassmorphism */
        .alert-success {{
            background: rgba(27, 211, 205, 0.15);
            backdrop-filter: blur(20px);
            border-left: 5px solid #1BD3CD;
            padding: 1.3rem 1.5rem;
            border-radius: 16px;
            margin: 1rem 0;
            color: #fff;
            box-shadow: 0 8px 24px rgba(27, 211, 205, 0.3), 0 0 0 1px rgba(27, 211, 205, 0.2) inset;
            border: 1px solid rgba(27, 211, 205, 0.3);
            font-weight: 500;
        }}
        .alert-info {{
            background: rgba(0, 212, 255, 0.15);
            backdrop-filter: blur(20px);
            border-left: 5px solid #00D4FF;
            padding: 1.3rem 1.5rem;
            border-radius: 16px;
            margin: 1rem 0;
            color: #fff;
            box-shadow: 0 8px 24px rgba(0, 212, 255, 0.3), 0 0 0 1px rgba(0, 212, 255, 0.2) inset;
            border: 1px solid rgba(0, 212, 255, 0.3);
            font-weight: 500;
        }}
        .alert-warning {{
            background: rgba(255, 226, 89, 0.15);
            backdrop-filter: blur(20px);
            border-left: 5px solid #ffe259;
            padding: 1.3rem 1.5rem;
            border-radius: 16px;
            margin: 1rem 0;
            color: #fff;
            box-shadow: 0 8px 24px rgba(255, 226, 89, 0.3), 0 0 0 1px rgba(255, 226, 89, 0.2) inset;
            border: 1px solid rgba(255, 226, 89, 0.3);
            font-weight: 500;
        }}
        /* Enhanced Animations */
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(40px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        @keyframes scaleIn {{
            from {{
                opacity: 0;
                transform: scale(0.9);
            }}
            to {{
                opacity: 1;
                transform: scale(1);
            }}
        }}
        .fade-in {{
            animation: fadeInUp 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }}
        .scale-in {{
            animation: scaleIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }}
        
        /* Clean, Visible Text - Enhanced */
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div,
        p, span, div, label, .stSelectbox label, .stRadio label {{
            color: #ffffff !important;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.7);
            font-weight: 400;
        }}
        
        /* Headings with Better Visibility */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
            color: #ffffff !important;
            font-family: 'Poppins', sans-serif !important;
            font-weight: 700 !important;
            text-shadow: 0 3px 12px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 0, 0, 0.4) !important;
        }}
        
        /* Stronger text for markdown bold */
        strong, b, .stMarkdown strong, .stMarkdown b {{
            color: #ffffff !important;
            font-weight: 700 !important;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.6);
        }}
        
        /* Tab labels */
        .stTabs [data-baseweb="tab-list"] {{
            background: #ffffff !important;
            border-radius: 12px !important;
            padding: 0.5rem !important;
            backdrop-filter: none !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
        }}
        
        .stTabs [data-baseweb="tab-list"] button {{
            background: #ffffff !important;
            color: #1a1a1a !important;
            font-weight: 600 !important;
            text-shadow: none !important;
            border: none !important;
            padding: 0.75rem 1.5rem !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }}
        
        .stTabs [data-baseweb="tab-list"] button:hover {{
            background: #f5f5f5 !important;
            color: #0d0d0d !important;
        }}
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            background: #808080 !important;
            color: #0a0a0a !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        }}
        
        /* Metric labels */
        [data-testid="stMetricLabel"] {{
            color: #ffffff !important;
            font-weight: 600 !important;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
        }}
        
        /* Metric values */
        [data-testid="stMetricValue"] {{
            color: #ffffff !important;
            font-weight: 700 !important;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.6);
        }}
        
        /* Input Fields with Clean White Background */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stMultiSelect > div > div,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea {{
            background: rgba(10, 83, 112, 0.95) !important;
            backdrop-filter: blur(15px) !important;
            border: 2px solid rgba(0, 0, 0, 0.2) !important;
            border-radius: 12px !important;
            color: #000000 !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(10, 83, 112, 0.8) inset !important;
            padding: 0.75rem !important;
            transition: all 0.3s ease !important;
        }}
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stMultiSelect > div > div:focus-within,
        .stNumberInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {{
            border: 2px solid #1BD3CD !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15), 0 0 0 3px rgba(27, 211, 205, 0.2) !important;
            outline: none !important;
        }}
        
        /* Input placeholder */
        .stTextInput > div > div > input::placeholder,
        .stTextArea > div > div > textarea::placeholder {{
            color: rgba(0, 0, 0, 0.5) !important;
            font-weight: 500 !important;
        }}
        
        /* Select dropdown options */
        .stSelectbox > div > div > select option {{
            background: rgba(10, 83, 112, 0.98) !important;
            color: #000000 !important;
            font-weight: 600 !important;
            padding: 0.5rem !important;
        }}
        
        .stSelectbox > div > div > select option:hover {{
            background: rgba(27, 211, 205, 0.1) !important;
            color: #000000 !important;
        }}
        
        /* Professional Navigation Dropdown Styling */
        .stSelectbox > div > div > select {{
            background: #f0f0f0 !important;
            color: #333333 !important;
            font-weight: 500 !important;
            border: 1px solid #cccccc !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
        }}
        
        /* Sidebar navigation dropdown */
        [data-testid="stSidebar"] .stSelectbox > div > div > select {{
            background: #f0f0f0 !important;
            color: #333333 !important;
            font-weight: 500 !important;
            border: 1px solid #cccccc !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
        }}
        
        /* Dropdown text color - comprehensive targeting */
        select,
        .stSelectbox select,
        [data-baseweb="select"] input,
        [role="combobox"],
        .stSelectbox > div > div > select,
        [data-testid="stSidebar"] select {{
            background: #f0f0f0 !important;
            color: #333333 !important;
            font-weight: 500 !important;
            border: 1px solid #cccccc !important;
        }}
        
        /* Dropdown option text color */
        select option,
        option {{
            background: rgba(255, 255, 255, 0.95) !important;
            color: #333333 !important;
            font-weight: 500 !important;
        }}
        
        /* Dropdown placeholder and input text */
        .stSelectbox input,
        [data-baseweb="select"] input,
        [class*="select"] input {{
            color: #000000 !important;
            font-weight: 500 !important;
        }}
        
        /* Universal dropdown text styling */
        [class*="select"]:not([class*="multiselect"]) *,
        [data-baseweb="select"] * {{
            color: #000000 !important;
        }}
        
        [data-testid="stSidebar"] .stSelectbox label {{
            color: #000000 !important;
            font-weight: 600 !important;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
        }}
        
        /* Multiselect tags */
        .stMultiSelect span {{
            color: #000000 !important;
            background: rgba(255, 255, 255, 0.9) !important;
            border: 1px solid rgba(0, 0, 0, 0.3) !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
            padding: 0.3rem 0.6rem !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }}
        
        .stMultiSelect span:hover {{
            background: rgba(27, 211, 205, 0.2) !important;
            border: 1px solid #1BD3CD !important;
        }}
        
        /* Checkboxes and Radio Buttons - Enhanced Visibility */
        .stCheckbox label, .stRadio label {{
            color: rgba(255, 255, 255, 0.95) !important;
            font-weight: 500 !important;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
        }}
        
        /* Radio button text */
        .stRadio > label > div[role="radiogroup"] > label {{
            color: #ffffff !important;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
        }}
        
        /* Slider labels */
        .stSlider > label {{
            color: rgba(255, 255, 255, 0.95) !important;
            font-weight: 600 !important;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
        }}
        
        /* Data editor / DataFrame */
        .stDataFrame, [data-testid="stDataFrame"] {{
            background: #ffffff !important;
            color: #0a0a0a !important;
            border-radius: 12px !important;
            border: 1px solid rgba(0, 0, 0, 0.15) !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
        }}
        
        .stDataFrame table, [data-testid="stDataFrame"] table {{
            background: #ffffff !important;
            color: #0a0a0a !important;
        }}
        
        .stDataFrame th, [data-testid="stDataFrame"] th {{
            background: #ffffff !important;
            color: #000000 !important;
            font-weight: 800 !important;
            border-bottom: 2px solid #e0e0e0 !important;
            padding: 12px 16px !important;
            text-align: left !important;
        }}
        
        .stDataFrame td, [data-testid="stDataFrame"] td {{
            background: #ffffff !important;
            color: #0a0a0a !important;
            padding: 12px 16px !important;
            border-bottom: 1px solid #e8e8e8 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }}
        
        .stDataFrame tr:hover td {{
            background: #f8f9fa !important;
            color: #000000 !important;
        }}
        
        .stDataFrame tr:nth-child(even) td {{
            background: #fcfcfc !important;
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(20px) !important;
            border-radius: 12px !important;
            color: #000000 !important;
            font-weight: 600 !important;
            text-shadow: none !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        }}
        
        .streamlit-expanderHeader:hover {{
            background: rgba(255, 255, 255, 1) !important;
            border: 1px solid #1BD3CD !important;
        }}
        
        /* Sidebar Glassmorphism */
        [data-testid="stSidebar"] {{
            background: rgba(13, 27, 42, 0.85) !important;
            backdrop-filter: blur(25px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        }}
        
        /* Sidebar headers and navigation */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: #ffffff !important;
            text-shadow: 0 2px 6px rgba(0, 0, 0, 0.6);
        }}
        
        /* Sidebar text */
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {{
            color: #ffffff !important;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
        }}
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        ::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: linear-gradient(180deg, #43e97b 0%, #38f9d7 100%);
        }}
        
        /* List items (for features, etc.) */
        ul, ol {{
            color: rgba(255, 255, 255, 0.95) !important;
        }}
        
        li {{
            color: rgba(255, 255, 255, 0.95) !important;
            margin: 0.5rem 0;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
        }}
        
        /* Links */
        a {{
            color: #1BD3CD !important;
            text-decoration: none;
            font-weight: 600;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
        }}
        
        a:hover {{
            color: #00D4FF !important;
            text-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
        }}
        
        /* Code blocks */
        code {{
            background: rgba(255, 255, 255, 0.15) !important;
            color: #1BD3CD !important;
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-weight: 600;
        }}
        
        pre {{
            background: rgba(13, 27, 42, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px;
            padding: 1rem;
        }}
        
        pre code {{
            color: rgba(255, 255, 255, 0.95) !important;
        }}
        
        /* Info/Warning/Error boxes */
        .stAlert {{
            background: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(20px) !important;
            border-radius: 12px !important;
            color: #ffffff !important;
        }}
        
        /* Progress bars */
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, #00D4FF 0%, #1BD3CD 50%, #097579 100%) !important;
        }}
        </style>
        """
        
        st.markdown(css_content, unsafe_allow_html=True)
    
    def create_header(self, title, subtitle=None):
        """Create professional header"""
        st.markdown(f"""
        <div class="enterprise-header fade-in">
            <h1>{title}</h1>
            {f'<p>{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)
    
    def create_metric_card(self, title, value, change=None, status="neutral"):
        """Create animated metric card"""
        status_class = f"status-{status}" if status != "neutral" else ""
        change_html = f'<span class="{status_class}">({change})</span>' if change else ''
        
        st.markdown(f"""
        <div class="metric-card fade-in">
            <p class="metric-value">{value} {change_html}</p>
            <p class="metric-label">{title}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def create_progress_bar(self, value, max_value=100, label="Progress"):
        """Create animated progress bar"""
        percentage = (value / max_value) * 100
        st.markdown(f"""
        <div class="fade-in">
            <p style="margin: 0.5rem 0; font-weight: 600;">{label}: {value}/{max_value}</p>
            <div class="progress-container">
                <div class="progress-bar" style="width: {percentage}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_floating_carousel(self, content_html, title=""):
        """Create a floating carousel container with glassmorphism effect"""
        st.markdown(f"""
        <div class="carousel-container fade-in">
            {f'<h2 style="margin-top: 0; color: white; font-family: Poppins, sans-serif;">{title}</h2>' if title else ''}
            {content_html}
        </div>
        """, unsafe_allow_html=True)
    
    def create_alert(self, message, alert_type="info"):
        """Create styled alert box with glassmorphism"""
        st.markdown(f"""
        <div class="alert-{alert_type} fade-in">
            <strong>{message}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    def create_feature_grid(self, features, selected_features=None):
        """Create interactive feature selection grid"""
        if selected_features is None:
            selected_features = features
            
        st.markdown("### 🎯 Feature Selection")
        
        # Create grid layout
        cols = st.columns(4)
        selected = []
        
        for i, feature in enumerate(features):
            with cols[i % 4]:
                is_selected = st.checkbox(
                    feature.replace('_', ' ').title(),
                    value=feature in selected_features,
                    key=f"feature_{feature}"
                )
                if is_selected:
                    selected.append(feature)
        
        return selected
    
    def create_model_comparison_chart(self, results_df):
        """Create model comparison visualization"""
        fig = go.Figure()
        
        # Add bars for each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                fig.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=results_df['model'],
                    y=results_df[metric],
                    marker_color=colors[i % len(colors)],
                    text=results_df[metric].round(3),
                    textposition='auto',
                ))
        
        fig.update_layout(
            title="🏆 Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            template='plotly_white',
            height=500,
            font=dict(family="Inter, sans-serif", size=12),
            title_font_size=16
        )
        
        return fig
    
    def create_confusion_matrix_heatmap(self, cm, classes):
        """Create confusion matrix heatmap"""
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=classes,
            y=classes,
            color_continuous_scale='Blues',
            title="🎯 Confusion Matrix"
        )
        
        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(cm[i, j]),
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                )
        
        fig.update_layout(
            template='plotly_white',
            font=dict(family="Inter, sans-serif", size=12),
            title_font_size=16
        )
        
        return fig
    
    def create_feature_importance_chart(self, importance_df):
        """Create feature importance visualization"""
        fig = px.bar(
            importance_df.head(15),
            x='importance',
            y='feature',
            orientation='h',
            title="🎯 Top 15 Feature Importance",
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            template='plotly_white',
            height=600,
            font=dict(family="Inter, sans-serif", size=12),
            title_font_size=16,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_training_progress_chart(self, history):
        """Create training progress visualization"""
        fig = go.Figure()
        
        epochs = list(range(1, len(history['loss']) + 1))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history['loss'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#1f77b4', width=3)
        ))
        
        if 'val_loss' in history:
            fig.add_trace(go.Scatter(
                x=epochs,
                y=history['val_loss'],
                mode='lines+markers',
                name='Validation Loss',
                line=dict(color='#ff7f0e', width=3)
            ))
        
        fig.update_layout(
            title="📈 Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template='plotly_white',
            height=400,
            font=dict(family="Inter, sans-serif", size=12),
            title_font_size=16
        )
        
        return fig
    
    def display_timestamp(self):
        """Display current timestamp"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"*Last updated: {current_time}*")
