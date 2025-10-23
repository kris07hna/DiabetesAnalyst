#!/usr/bin/env python3
"""
Health check script for Streamlit Cloud deployment
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def health_check():
    """Simple health check function"""
    try:
        # Test basic imports
        import streamlit as st
        import pandas as pd
        import numpy as np
        
        # Test if data file exists
        data_file = "diabetes_012_health_indicators_BRFSS2015.csv"
        if not os.path.exists(data_file):
            return False, f"Data file {data_file} not found"
        
        # Test if modules can be imported
        from modules import dashboard, model_trainer, data_manager
        
        return True, "All checks passed"
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    success, message = health_check()
    if success:
        print("OK")
        sys.exit(0)
    else:
        print(f"FAIL: {message}")
        sys.exit(1)