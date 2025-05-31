"""
Tests for the LLMHandler class.
"""

import pytest
from unittest.mock import patch, MagicMock

from mlflow_assist.llm.llm_handler import LLMHandler, LLMConfig

@pytest.fixture
def llm_handler():
    """Create an LLMHandler instance for testing."""
    config = LLMConfig(
        model_name="gpt-3.5-turbo",
        max_length=100,
        temperature=0.7
    )
    return LLMHandler(config)

def test_llm_initialization(llm_handler):
    """Test LLMHandler initialization."""
    assert llm_handler.config.model_name == "gpt-3.5-turbo"
    assert llm_handler.config.max_length == 100
    assert llm_handler.config.temperature == 0.7

@patch('requests.post')
def test_generate_api(mock_post, llm_handler):
    """Test API-based text generation."""
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Generated text"}}]
    }
    mock_post.return_value = mock_response
    
    response = llm_handler.generate("Test prompt")
    assert response == "Generated text"
    assert mock_post.called

def test_load_model(llm_handler):
    """Test local model loading."""
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            llm_handler.load_model("local-model")
            assert mock_tokenizer.called
            assert mock_model.called

