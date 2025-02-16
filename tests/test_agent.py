import pytest
from unittest.mock import Mock, patch
from joao import Agent
from joao import AsyncAgent

@pytest.fixture
def mock_openai():
    with patch('joao.agent.OpenAI') as mock:
        # Create a mock client instance
        mock_client = Mock()
        mock.return_value = mock_client
        
        # Create mock response
        mock_message = Mock()
        mock_message.content = "Hello, I'm a mock response"
        mock_message.tool_calls = None
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        # Set up the mock chat completions
        mock_client.chat.completions.create.return_value = mock_response
        yield mock_client

@pytest.fixture
def mock_async_openai():
    with patch('joao.async_agent.AsyncOpenAI') as mock:
        # Create a mock client instance
        mock_client = Mock()
        mock.return_value = mock_client
        
        # Create mock response
        mock_message = Mock()
        mock_message.content = "Hello, I'm a mock response"
        mock_message.tool_calls = None
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        # Set up the mock chat completions to return an awaitable
        async def async_create(**kwargs):
            return mock_response
        
        mock_client.chat.completions.create = async_create
        yield mock_client

# Synchronous Agent Tests
def test_agent_initialization():
    agent = Agent("test prompt")
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"
    assert agent.messages[0]["content"] == "test prompt"

def test_agent_request_without_tools(mock_openai):
    agent = Agent("test prompt")
    response = agent.request("test message")
    
    assert response == "Hello, I'm a mock response"
    assert len(agent.messages) == 3  # system + user + assistant
    assert agent.messages[1]["role"] == "user"
    assert agent.messages[1]["content"] == "test message"

def test_agent_request_with_tools(mock_openai):
    def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    agent = Agent("test prompt")
    response = agent.request("test message", tools=[test_tool])
    
    # Verify tool definitions were passed
    call_kwargs = mock_openai.chat.completions.create.call_args[1]
    assert "tools" in call_kwargs
    assert len(call_kwargs["tools"]) == 1
    assert call_kwargs["tools"][0]["function"]["name"] == "test_tool"

def test_agent_request_with_auto_tool_execution(mock_openai):
    def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    # Set up mock response with tool calls
    mock_tool_call = Mock()
    mock_tool_call.id = "call_1"
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"param": "test value"}'
    
    # First response with tool call
    mock_message1 = Mock()
    mock_message1.content = "Using tool..."
    mock_message1.tool_calls = [mock_tool_call]
    
    mock_choice1 = Mock()
    mock_choice1.message = mock_message1
    
    mock_response1 = Mock()
    mock_response1.choices = [mock_choice1]
    
    # Second response after tool execution
    mock_message2 = Mock()
    mock_message2.content = "Tool execution complete"
    mock_message2.tool_calls = None
    
    mock_choice2 = Mock()
    mock_choice2.message = mock_message2
    
    mock_response2 = Mock()
    mock_response2.choices = [mock_choice2]
    
    # Set up sequence of responses
    mock_openai.chat.completions.create.side_effect = [mock_response1, mock_response2]
    
    agent = Agent("test prompt")
    response = agent.request("test message", tools=[test_tool], use_tools=True)
    
    assert response == "Tool execution complete"
    assert mock_openai.chat.completions.create.call_count == 2

def test_agent_request_no_auto_tool_execution(mock_openai):
    def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    # Set up mock response with tool calls
    mock_tool_call = Mock()
    mock_tool_call.id = "call_1"
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"param": "test value"}'
    
    mock_message = Mock()
    mock_message.content = "Using tool..."
    mock_message.tool_calls = [mock_tool_call]
    
    mock_choice = Mock()
    mock_choice.message = mock_message
    
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    
    mock_openai.chat.completions.create.return_value = mock_response
    
    agent = Agent("test prompt")
    response = agent.request("test message", tools=[test_tool], use_tools=False)
    
    assert response == "Using tool..."  # Original response without tool execution
    assert mock_openai.chat.completions.create.call_count == 1

def test_use_tools_without_auto_update(mock_openai):
    def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    agent = Agent("test prompt")
    agent.tools_handler.set_tools([test_tool])
    
    # Set up a mock tool call
    mock_tool_call = Mock()
    mock_tool_call.id = "call_1"
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"param": "test value"}'
    
    agent.tools_handler.set_tool_calls([mock_tool_call])
    result = agent.use_tools(auto_update=False)
    
    assert result is None  # No update requested
    assert mock_openai.chat.completions.create.call_count == 0

def test_use_tools_with_auto_update(mock_openai):
    def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    agent = Agent("test prompt")
    agent.tools_handler.set_tools([test_tool])
    
    # Set up a mock tool call
    mock_tool_call = Mock()
    mock_tool_call.id = "call_1"
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"param": "test value"}'
    
    agent.tools_handler.set_tool_calls([mock_tool_call])
    result = agent.use_tools(auto_update=True)
    
    assert result == "Hello, I'm a mock response"  # From the completion after tool execution
    assert mock_openai.chat.completions.create.call_count == 1

# Async Agent Tests
@pytest.mark.asyncio
async def test_async_agent_initialization():
    agent = AsyncAgent("test prompt")
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"
    assert agent.messages[0]["content"] == "test prompt"

@pytest.mark.asyncio
async def test_async_agent_request_without_tools(mock_async_openai):
    agent = AsyncAgent("test prompt")
    response = await agent.request("test message")
    
    assert response == "Hello, I'm a mock response"
    assert len(agent.messages) == 3  # system + user + assistant
    assert agent.messages[1]["role"] == "user"
    assert agent.messages[1]["content"] == "test message"

@pytest.mark.asyncio
async def test_async_agent_request_with_tools(mock_async_openai):
    async def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    # Set up mock response
    mock_message = Mock()
    mock_message.content = "Using tool..."
    mock_message.tool_calls = None
    
    mock_choice = Mock()
    mock_choice.message = mock_message
    
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    
    async def async_create(**kwargs):
        return mock_response
    
    mock_async_openai.chat.completions.create = async_create
    
    agent = AsyncAgent("test prompt")
    response = await agent.request("test message", tools=[test_tool])
    
    assert response == "Using tool..."

@pytest.mark.asyncio
async def test_async_agent_request_with_auto_tool_execution(mock_async_openai):
    async def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    # Set up mock response with tool calls
    mock_tool_call = Mock()
    mock_tool_call.id = "call_1"
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"param": "test value"}'
    
    # First response with tool call
    mock_message1 = Mock()
    mock_message1.content = "Using tool..."
    mock_message1.tool_calls = [mock_tool_call]
    
    mock_choice1 = Mock()
    mock_choice1.message = mock_message1
    
    mock_response1 = Mock()
    mock_response1.choices = [mock_choice1]
    
    # Second response after tool execution
    mock_message2 = Mock()
    mock_message2.content = "Tool execution complete"
    mock_message2.tool_calls = None
    
    mock_choice2 = Mock()
    mock_choice2.message = mock_message2
    
    mock_response2 = Mock()
    mock_response2.choices = [mock_choice2]
    
    # Set up sequence of responses
    responses = [mock_response1, mock_response2]
    async def async_create(**kwargs):
        return responses.pop(0)
    
    mock_async_openai.chat.completions.create = async_create
    
    agent = AsyncAgent("test prompt")
    response = await agent.request("test message", tools=[test_tool], use_tools=True)
    
    assert response == "Tool execution complete"

@pytest.mark.asyncio
async def test_async_use_tools_without_auto_update(mock_async_openai):
    async def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    agent = AsyncAgent("test prompt")
    agent.tools_handler.set_tools([test_tool])
    
    # Set up a mock tool call
    mock_tool_call = Mock()
    mock_tool_call.id = "call_1"
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"param": "test value"}'
    
    agent.tools_handler.set_tool_calls([mock_tool_call])
    result = await agent.use_tools(auto_update=False)
    
    assert result is None  # No update requested

@pytest.mark.asyncio
async def test_async_use_tools_with_auto_update(mock_async_openai):
    async def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    # Set up mock response
    mock_message = Mock()
    mock_message.content = "Tool execution complete"
    mock_message.tool_calls = None
    
    mock_choice = Mock()
    mock_choice.message = mock_message
    
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    
    async def async_create(**kwargs):
        return mock_response
    
    mock_async_openai.chat.completions.create = async_create
    
    agent = AsyncAgent("test prompt")
    agent.tools_handler.set_tools([test_tool])
    
    # Set up a mock tool call
    mock_tool_call = Mock()
    mock_tool_call.id = "call_1"
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"param": "test value"}'
    
    agent.tools_handler.set_tool_calls([mock_tool_call])
    result = await agent.use_tools(auto_update=True)
    
    assert result == "Tool execution complete"
