"""
DiabeticsAI Enterprise Modules
Professional medical AI platform modules
"""

__version__ = "2.0.0"
__author__ = "DiabeticsAI Team"

# Module imports
from .config import AppConfig
from .ui_components import UIComponents
from .data_manager import DataManager
from .model_trainer import ModelTrainer
from .analytics import AnalyticsEngine
from .predictions import PredictionEngine
from .dashboard import DashboardManager

__all__ = [
    'AppConfig',
    'UIComponents', 
    'DataManager',
    'ModelTrainer',
    'AnalyticsEngine',
    'PredictionEngine',
    'DashboardManager'
]
