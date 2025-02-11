import os
import pytest

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Set up mock environment variables for testing"""
    os.environ["OPENAI_API_KEY"] = "test_key"
    os.environ["OPENAI_BASE_URL"] = "https://test.api/v1"
    os.environ["OPENAI_MODEL"] = "test-model"
