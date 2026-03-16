"""Validation tests to ensure the testing infrastructure is set up correctly."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock


class TestInfrastructureValidation:
    """Tests to validate that the testing infrastructure is properly configured."""
    
    def test_pytest_is_importable(self):
        """Test that pytest can be imported successfully."""
        import pytest
        assert pytest is not None
        assert hasattr(pytest, 'main')
    
    def test_pytest_fixtures_available(self):
        """Test that custom fixtures from conftest.py are available."""
        from tests.conftest import (
            temp_dir, mock_config, mock_environment,
            mock_transformer_model, mock_tokenizer
        )
        assert temp_dir is not None
        assert mock_config is not None
        assert mock_environment is not None
        assert mock_transformer_model is not None
        assert mock_tokenizer is not None
    
    def test_project_structure_exists(self):
        """Test that the expected project structure exists."""
        project_root = Path(__file__).parent.parent
        
        # Check main directories
        assert (project_root / "src").exists()
        assert (project_root / "demo").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        
        # Check configuration files
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / ".gitignore").exists()
    
    def test_coverage_is_working(self):
        """Test that coverage tracking is functional."""
        try:
            import coverage
            cov = coverage.Coverage()
            assert cov is not None
        except ImportError:
            pytest.fail("Coverage module is not installed")
    
    @pytest.mark.unit
    def test_unit_marker_works(self):
        """Test that the unit test marker is recognized."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker_works(self):
        """Test that the integration test marker is recognized."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker_works(self):
        """Test that the slow test marker is recognized."""
        assert True
    
    def test_temp_dir_fixture(self, temp_dir):
        """Test that the temp_dir fixture works correctly."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Create a test file in the temp directory
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    def test_mock_config_fixture(self, mock_config):
        """Test that the mock_config fixture provides expected data."""
        assert isinstance(mock_config, dict)
        assert "debug" in mock_config
        assert "api_key" in mock_config
        assert mock_config["debug"] is True
        assert mock_config["api_key"] == "test-key-123"
    
    def test_mock_environment_fixture(self, mock_environment):
        """Test that the mock_environment fixture sets environment variables."""
        import os
        assert os.environ.get("API_KEY") == "test-api-key"
        assert os.environ.get("DEBUG") == "true"
        assert os.environ.get("LOG_LEVEL") == "DEBUG"
    
    def test_sample_data_fixtures(self, sample_text_data, sample_json_data):
        """Test that sample data fixtures provide expected data."""
        assert isinstance(sample_text_data, list)
        assert len(sample_text_data) > 0
        assert all(isinstance(text, str) for text in sample_text_data)
        
        assert isinstance(sample_json_data, dict)
        assert "items" in sample_json_data
        assert isinstance(sample_json_data["items"], list)
    
    def test_mock_model_fixtures(self, mock_transformer_model, mock_tokenizer):
        """Test that ML model mock fixtures work correctly."""
        # Test transformer model
        assert hasattr(mock_transformer_model, 'generate')
        assert hasattr(mock_transformer_model, 'config')
        result = mock_transformer_model.generate()
        assert result.sequences == [[1, 2, 3, 4, 5]]
        
        # Test tokenizer
        assert hasattr(mock_tokenizer, 'encode')
        assert hasattr(mock_tokenizer, 'decode')
        encoded = mock_tokenizer.encode("test")
        assert encoded == [101, 2003, 2049, 102]
        decoded = mock_tokenizer.decode([1, 2, 3])
        assert decoded == "This is test output"
    
    def test_pytest_mock_is_available(self):
        """Test that pytest-mock functionality is available."""
        from unittest.mock import Mock, MagicMock, patch
        
        mock_obj = Mock()
        mock_obj.method.return_value = "mocked"
        assert mock_obj.method() == "mocked"
        
        magic_mock = MagicMock()
        magic_mock.__len__.return_value = 5
        assert len(magic_mock) == 5
    
    def test_capture_logs_fixture(self, capture_logs):
        """Test that the capture_logs fixture works correctly."""
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info("Test log message")
        logger.warning("Test warning message")
        
        log_contents = capture_logs.getvalue()
        assert "Test log message" in log_contents
        assert "Test warning message" in log_contents
        assert "INFO" in log_contents
        assert "WARNING" in log_contents
    
    def test_parametrized_device_fixture(self, device_type):
        """Test that the parametrized device fixture provides different values."""
        assert device_type in ["cpu", "cuda", "mps"]
    
    def test_http_response_mock(self, mock_http_response):
        """Test that the mock HTTP response fixture works correctly."""
        assert mock_http_response.status_code == 200
        assert mock_http_response.json() == {"status": "success", "data": {"result": "test"}}
        assert "Content-Type" in mock_http_response.headers
        
        # Should not raise an exception
        mock_http_response.raise_for_status()
    
    def test_poetry_scripts_configured(self):
        """Test that Poetry scripts are properly configured in pyproject.toml."""
        from pathlib import Path
        
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists()
        
        # Read the file content to check for test scripts
        content = pyproject_path.read_text()
        assert "[tool.poetry.scripts]" in content
        assert "test = \"pytest:main\"" in content
        assert "tests = \"pytest:main\"" in content