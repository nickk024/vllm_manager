import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class TestFrontend:
    """Test suite for the frontend application."""
    
    def test_frontend_exists(self):
        """Test that the frontend app.py file exists."""
        frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'frontend', 'app.py')
        assert os.path.exists(frontend_path)
        
    def test_frontend_templates_exist(self):
        """Test that the frontend templates exist."""
        templates_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'frontend', 'templates')
        assert os.path.exists(templates_dir)
        assert os.path.exists(os.path.join(templates_dir, 'index.html'))