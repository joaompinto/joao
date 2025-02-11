import inspect
from inspect import Parameter
import json
import asyncio
from typing import Optional, List, Callable, Any, Dict, Union

class ToolsHandler:
    def __init__(self):
        self.tools = None
        self.tool_calls = []
        self._last_tool_calls = []

    def set_tools(self, tools: Optional[List[Callable]]) -> None:
        """Set the available tools for the handler"""
        self.tools = tools

    def clear_tool_calls(self) -> None:
        """Clear all pending tool calls"""
        self.tool_calls = []

    def set_tool_calls(self, tool_calls: Optional[List[Any]]) -> None:
        """Set the pending tool calls"""
        self._last_tool_calls = self.tool_calls
        self.tool_calls = tool_calls or []

    def has_pending_calls(self) -> bool:
        """Check if there are any pending tool calls"""
        return bool(self.tool_calls)

    def get_last_tool_calls(self) -> List[Any]:
        """Get the tool calls from the last request"""
        return self._last_tool_calls

    def create_tool_def(self, callable: Callable) -> Dict[str, Any]:
        """
        Create a tool definition based on a function's signature and docstring.
        The definition follows the JSONSchema format used for tool descriptions.
        
        Args:
            callable: The function or method to create a tool definition for
            
        Returns:
            dict: A tool definition containing name, description, and parameters schema
        """
        # Get function signature
        sig = inspect.signature(callable)
        
        # Create parameters schema
        properties = {}
        required = []
        
        for name, param in sig.parameters.items():
            param_def = {"type": "string"}  # Default to string type
            
            # Handle different parameter kinds
            if param.kind == Parameter.VAR_POSITIONAL:
                continue  # Skip *args
            if param.kind == Parameter.VAR_KEYWORD:
                continue  # Skip **kwargs
                
            # Add parameter to required list if it has no default value
            if param.default == Parameter.empty:
                required.append(name)
                
            properties[name] = param_def
            
        # Create the complete tool definition
        tool_def = {
            "type": "function",
            "function": {
                "name": callable.__name__,
                "description": callable.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                }
            }
        }
        
        if required:
            tool_def["function"]["parameters"]["required"] = required
            
        return tool_def

    def get_tools_definitions(self) -> List[Dict[str, Any]]:
        """Get the tool definitions for all available tools"""
        if not self.tools:
            return []
        return [self.create_tool_def(tool) for tool in self.tools]

    def execute_tool_calls(self) -> Optional[str]:
        """Execute all pending tool calls"""
        if not self.has_pending_calls():
            return None

        results = []
        for tool_call in self.tool_calls:
            result = self._call_tool(tool_call)
            if result is not None:
                results.append(str(result))

        return "\n".join(results)

    def _call_tool(self, tool_call: Any) -> Any:
        """
        Execute a tool call from the model response
        
        Args:
            tool_call: The tool call object from the model
        """
        if not self.tools:
            return None
            
        tool_name = tool_call.function.name
        tool = next((t for t in self.tools if t.__name__ == tool_name), None)
        
        if not tool:
            return None
            
        if asyncio.iscoroutinefunction(tool):
            raise TypeError("Cannot execute async tool in sync handler")
            
        try:
            args = json.loads(tool_call.function.arguments)
            return tool(**args)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

class AsyncToolsHandler:
    def __init__(self):
        self.tools = None
        self.tool_calls = []
        self._last_tool_calls = []

    def set_tools(self, tools: Optional[List[Callable]]) -> None:
        """Set the available tools for the handler"""
        self.tools = tools

    def clear_tool_calls(self) -> None:
        """Clear all pending tool calls"""
        self.tool_calls = []

    def set_tool_calls(self, tool_calls: Optional[List[Any]]) -> None:
        """Set the pending tool calls"""
        self._last_tool_calls = self.tool_calls
        self.tool_calls = tool_calls or []

    def has_pending_calls(self) -> bool:
        """Check if there are any pending tool calls"""
        return bool(self.tool_calls)

    def get_last_tool_calls(self) -> List[Any]:
        """Get the tool calls from the last request"""
        return self._last_tool_calls

    def create_tool_def(self, callable: Callable) -> Dict[str, Any]:
        """
        Create a tool definition based on a function's signature and docstring.
        The definition follows the JSONSchema format used for tool descriptions.
        
        Args:
            callable: The function or method to create a tool definition for
            
        Returns:
            dict: A tool definition containing name, description, and parameters schema
        """
        # Get function signature
        sig = inspect.signature(callable)
        
        # Create parameters schema
        properties = {}
        required = []
        
        for name, param in sig.parameters.items():
            param_def = {"type": "string"}  # Default to string type
            
            # Handle different parameter kinds
            if param.kind == Parameter.VAR_POSITIONAL:
                continue  # Skip *args
            if param.kind == Parameter.VAR_KEYWORD:
                continue  # Skip **kwargs
                
            # Add parameter to required list if it has no default value
            if param.default == Parameter.empty:
                required.append(name)
                
            properties[name] = param_def
            
        # Create the complete tool definition
        tool_def = {
            "type": "function",
            "function": {
                "name": callable.__name__,
                "description": callable.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                }
            }
        }
        
        if required:
            tool_def["function"]["parameters"]["required"] = required
            
        return tool_def

    def get_tools_definitions(self) -> List[Dict[str, Any]]:
        """Get the tool definitions for all available tools"""
        if not self.tools:
            return []
        return [self.create_tool_def(tool) for tool in self.tools]

    async def execute_tool_calls(self) -> Optional[str]:
        """Execute all pending tool calls"""
        if not self.has_pending_calls():
            return None

        results = []
        for tool_call in self.tool_calls:
            result = await self._call_tool(tool_call)
            if result is not None:
                results.append(str(result))

        return "\n".join(results)

    async def _call_tool(self, tool_call: Any) -> Any:
        """
        Execute a tool call from the model response
        
        Args:
            tool_call: The tool call object from the model
        """
        if not self.tools:
            return None
            
        tool_name = tool_call.function.name
        tool = next((t for t in self.tools if t.__name__ == tool_name), None)
        
        if not tool:
            return None
            
        if not asyncio.iscoroutinefunction(tool):
            raise TypeError("Cannot execute sync tool in async handler")
            
        try:
            args = json.loads(tool_call.function.arguments)
            return await tool(**args)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"