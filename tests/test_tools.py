import pytest
from unittest.mock import Mock
from joao.tools import ToolsHandler, AsyncToolsHandler

def test_tools_handler_initialization():
    handler = ToolsHandler()
    assert handler.tools is None
    assert handler.tool_calls == []

def test_async_tools_handler_initialization():
    handler = AsyncToolsHandler()
    assert handler.tools is None
    assert handler.tool_calls == []

def test_set_tools():
    def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    handler = ToolsHandler()
    handler.set_tools([test_tool])
    assert handler.tools == [test_tool]

def test_async_set_tools():
    def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    handler = AsyncToolsHandler()
    handler.set_tools([test_tool])
    assert handler.tools == [test_tool]

def test_create_tool_def():
    def test_tool(param: str, optional: str = "default"):
        """Test tool description"""
        return f"Tool called with {param}"
    
    handler = ToolsHandler()
    tool_def = handler.create_tool_def(test_tool)
    
    assert tool_def["function"]["name"] == "test_tool"
    assert tool_def["function"]["description"] == "Test tool description"
    assert "param" in tool_def["function"]["parameters"]["properties"]
    assert "optional" in tool_def["function"]["parameters"]["properties"]
    assert tool_def["function"]["parameters"]["required"] == ["param"]

def test_async_create_tool_def():
    def test_tool(param: str, optional: str = "default"):
        """Test tool description"""
        return f"Tool called with {param}"
    
    handler = AsyncToolsHandler()
    tool_def = handler.create_tool_def(test_tool)
    
    assert tool_def["function"]["name"] == "test_tool"
    assert tool_def["function"]["description"] == "Test tool description"
    assert "param" in tool_def["function"]["parameters"]["properties"]
    assert "optional" in tool_def["function"]["parameters"]["properties"]
    assert tool_def["function"]["parameters"]["required"] == ["param"]

def test_execute_tool_calls():
    def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    handler = ToolsHandler()
    handler.set_tools([test_tool])
    
    mock_call = Mock()
    mock_call.function.name = "test_tool"
    mock_call.function.arguments = '{"param": "test value"}'
    
    handler.set_tool_calls([mock_call])
    results = handler.execute_tool_calls()
    
    # Tool calls should be cleared
    assert handler.tool_calls == []
    assert len(results) == 1
    assert results[0] == "Tool called with test value"

@pytest.mark.asyncio
async def test_async_execute_tool_calls():
    async def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    handler = AsyncToolsHandler()
    handler.set_tools([test_tool])
    
    mock_call = Mock()
    mock_call.function.name = "test_tool"
    mock_call.function.arguments = '{"param": "test value"}'
    
    handler.set_tool_calls([mock_call])
    results = await handler.execute_tool_calls()
    
    # Tool calls should be cleared
    assert handler.tool_calls == []
    assert len(results) == 1
    assert results[0] == "Tool called with test value"

def test_sync_handler_with_async_tool():
    async def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    handler = ToolsHandler()
    handler.set_tools([test_tool])
    
    mock_call = Mock()
    mock_call.function.name = "test_tool"
    mock_call.function.arguments = '{"param": "test value"}'
    
    handler.set_tool_calls([mock_call])
    
    # Should raise TypeError when trying to execute async tool in sync handler
    with pytest.raises(TypeError):
        handler.execute_tool_calls()

@pytest.mark.asyncio
async def test_async_handler_with_sync_tool():
    def test_tool(param: str):
        """Test tool"""
        return f"Tool called with {param}"
    
    handler = AsyncToolsHandler()
    handler.set_tools([test_tool])
    
    mock_call = Mock()
    mock_call.function.name = "test_tool"
    mock_call.function.arguments = '{"param": "test value"}'
    
    handler.set_tool_calls([mock_call])
    
    # Should raise TypeError when trying to execute sync tool in async handler
    with pytest.raises(TypeError):
        await handler.execute_tool_calls()

def test_has_pending_calls():
    handler = ToolsHandler()
    
    # Test with no calls
    assert not handler.has_pending_calls()
    
    # Test with a call
    mock_call = Mock()
    mock_call.function.name = "test_tool"
    mock_call.function.arguments = '{"param": "test"}'
    
    handler.set_tool_calls([mock_call])
    assert handler.has_pending_calls()
    
    # Test after executing calls
    handler.clear_tool_calls()
    assert not handler.has_pending_calls()

@pytest.mark.asyncio
async def test_async_has_pending_calls():
    handler = AsyncToolsHandler()
    
    # Test with no calls
    assert not handler.has_pending_calls()
    
    # Test with a call
    mock_call = Mock()
    mock_call.function.name = "test_tool"
    mock_call.function.arguments = '{"param": "test"}'
    
    handler.set_tool_calls([mock_call])
    assert handler.has_pending_calls()
    
    # Test after executing calls
    handler.clear_tool_calls()
    assert not handler.has_pending_calls()
