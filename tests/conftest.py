import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is automatically cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file within a temporary directory."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("test content")
    yield temp_file_path


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing."""
    return {
        "debug": True,
        "verbose": False,
        "max_retries": 3,
        "timeout": 30,
        "api_key": "test-key-123",
        "base_url": "https://api.example.com",
        "model_name": "test-model",
        "temperature": 0.7,
        "max_tokens": 1000,
    }


@pytest.fixture
def mock_environment(monkeypatch) -> Dict[str, str]:
    """Set up mock environment variables for testing."""
    env_vars = {
        "API_KEY": "test-api-key",
        "DATABASE_URL": "sqlite:///test.db",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


@pytest.fixture
def mock_transformer_model():
    """Create a mock transformer model for testing."""
    model = MagicMock()
    model.generate.return_value = MagicMock(sequences=[[1, 2, 3, 4, 5]])
    model.config = MagicMock(
        max_length=512,
        vocab_size=50000,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
    )
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [101, 2003, 2049, 102]
    tokenizer.decode.return_value = "This is test output"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 102
    tokenizer.vocab_size = 50000
    return tokenizer


@pytest.fixture
def sample_text_data() -> list[str]:
    """Provide sample text data for testing."""
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "Machine learning models process text data.",
        "Testing is important for code quality.",
        "Python is a versatile programming language.",
    ]


@pytest.fixture
def sample_json_data() -> Dict[str, Any]:
    """Provide sample JSON data structure for testing."""
    return {
        "id": "test-123",
        "name": "Test Dataset",
        "description": "A sample dataset for testing purposes",
        "items": [
            {"id": 1, "text": "First item", "label": "positive"},
            {"id": 2, "text": "Second item", "label": "negative"},
            {"id": 3, "text": "Third item", "label": "neutral"},
        ],
        "metadata": {
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00Z",
            "author": "Test Author",
        }
    }


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset module-level state between tests."""
    yield
    import sys
    modules_to_reset = [
        module for module in sys.modules 
        if module.startswith(("src.", "demo.", "clc."))
    ]
    for module in modules_to_reset:
        sys.modules.pop(module, None)


@pytest.fixture
def capture_logs():
    """Capture log messages for testing."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    root_logger.removeHandler(handler)
    handler.close()


@pytest.fixture
def mock_http_response():
    """Create a mock HTTP response for testing."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"status": "success", "data": {"result": "test"}}
    response.text = '{"status": "success", "data": {"result": "test"}}'
    response.headers = {"Content-Type": "application/json"}
    response.raise_for_status = Mock()
    return response


@pytest.fixture(params=["cpu", "cuda", "mps"])
def device_type(request):
    """Parametrized fixture for testing with different device types."""
    return request.param


@pytest.fixture
def mock_torch_device(device_type):
    """Create a mock torch device based on device type."""
    device = Mock()
    device.type = device_type
    device.index = 0 if device_type != "cpu" else None
    return device