import inspect
from inspect import Parameter
import json
import asyncio
from typing import Optional, List, Callable, Any, Dict, Union
import os
from .debug import debug_print, is_debug_enabled

class ToolsHandler:
    """Handler for synchronous tool calls."""

    def __init__(self):
        self.tools = None
        self.tool_calls = []
        self._last_tool_calls = []

    def debug_print(self, *args, **kwargs):
        """Print debug information if debug is enabled for tools component"""
        debug_print('tools', *args, **kwargs)

    def set_tools(self, tools: Optional[List[Callable]]) -> None:
        """Set the available tools for the handler"""
        self.tools = tools
        if is_debug_enabled('tools'):
            self.debug_print("Available tools:")
            if tools:
                for tool in tools:
                    self.debug_print(f"- {tool.__name__}: {tool.__doc__}")
            else:
                self.debug_print("No tools available")

    def clear_tool_calls(self) -> None:
        """Clear all pending tool calls"""
        if is_debug_enabled('tools'):
            self.debug_print("Clearing tool calls")
        self.tool_calls = []

    def set_tool_calls(self, tool_calls: Optional[List[Any]]) -> None:
        """Set the pending tool calls"""
        self._last_tool_calls = self.tool_calls
        self.tool_calls = tool_calls or []
        if is_debug_enabled('tools'):
            self.debug_print(f"Set {len(self.tool_calls)} new tool calls")
            for call in self.tool_calls:
                self.debug_print(f"- Tool call: {call.function.name} with args: {call.function.arguments}")

    def has_pending_calls(self) -> bool:
        """Check if there are any pending tool calls"""
        has_calls = bool(self.tool_calls)
        if is_debug_enabled('tools'):
            self.debug_print(f"Has pending calls: {has_calls}")
        return has_calls

    def get_last_tool_calls(self) -> List[Any]:
        """Get the tool calls from the last request"""
        if is_debug_enabled('tools'):
            self.debug_print(f"Getting last {len(self._last_tool_calls)} tool calls")
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
            
        if is_debug_enabled('tools'):
            self.debug_print(f"Created tool definition for {callable.__name__}:")
            self.debug_print(json.dumps(tool_def, indent=2))
            
        return tool_def

    def get_tools_definitions(self) -> List[Dict[str, Any]]:
        """Get the tool definitions for all available tools"""
        if not self.tools:
            if is_debug_enabled('tools'):
                self.debug_print("No tools available")
            return []
        defs = [self.create_tool_def(tool) for tool in self.tools]
        if is_debug_enabled('tools'):
            self.debug_print(f"Generated {len(defs)} tool definitions")
        return defs

    def get_tool_schemas(self) -> Optional[List[Dict]]:
        """Get the OpenAI function schemas for all available tools
        
        Returns:
            Optional[List[Dict]]: List of tool schemas or None if no tools
        """
        if not self.tools:
            return None
            
        schemas = []
        for tool in self.tools:
            schema = self._get_tool_schema(tool)
            if schema:
                schemas.append(schema)
                
        return schemas if schemas else None

    def _get_tool_schema(self, tool: Callable) -> Optional[Dict]:
        """Get OpenAI function schema for a tool
        
        Args:
            tool: The tool function to get schema for
            
        Returns:
            Dict: OpenAI function schema
        """
        # Get function signature
        sig = inspect.signature(tool)
        
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
            
        # Create the complete tool schema
        tool_schema = {
            "type": "function",
            "function": {
                "name": tool.__name__,
                "description": tool.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        
        return tool_schema

    def execute_tool_call(self, tool_call: Any) -> str:
        """Execute a single tool call
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            str: The tool response
        """
        if not self.tools:
            if is_debug_enabled('tools'):
                self.debug_print("No tools available")
            return None
            
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        if is_debug_enabled('tools'):
            self.debug_print(f"Executing tool: {tool_name}")
            self.debug_print(f"Arguments: {tool_args}")
            
        tool = next((t for t in self.tools if t.__name__ == tool_name), None)
        if not tool:
            if is_debug_enabled('tools'):
                self.debug_print(f"Tool not found: {tool_name}")
            return str(None)
            
        try:
            response = tool(**tool_args)
            if is_debug_enabled('tools'):
                self.debug_print(f"Tool response: {response}")
            return str(response) if response is not None else str(None)
        except Exception as e:
            if is_debug_enabled('tools'):
                self.debug_print(f"Error executing tool: {e}")
            return str(None)

    def execute_tool_calls(self) -> Optional[str]:
        """Execute all pending tool calls
        
        Returns:
            str: Combined tool responses
        """
        if not self.has_pending_calls():
            if is_debug_enabled('tools'):
                self.debug_print("No pending tool calls to execute")
            return None

        results = []
        for tool_call in self.tool_calls:
            response = self.execute_tool_call(tool_call)
            if response is not None:
                results.append(response)

        return "\n".join(results)

    def get_pending_calls(self) -> List[Any]:
        """Get the list of pending tool calls"""
        return self.tool_calls

    def clear_pending_calls(self) -> None:
        """Clear the list of pending tool calls"""
        self.clear_tool_calls()

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
            if is_debug_enabled('tools'):
                self.debug_print(f"Calling tool {tool_name} with args: {args}")
            return tool(**args)
        except Exception as e:
            if is_debug_enabled('tools'):
                self.debug_print(f"Error executing {tool_name}: {str(e)}")
            return f"Error executing {tool_name}: {str(e)}"

class AsyncToolsHandler:
    """Handler for asynchronous tool calls."""

    def __init__(self):
        self.tools = None
        self.tool_calls = []
        self._last_tool_calls = []

    def debug_print(self, *args, **kwargs):
        """Print debug information if debug is enabled for tools component"""
        debug_print('tools', *args, **kwargs)

    def set_tools(self, tools: Optional[List[Callable]]) -> None:
        """Set the available tools for the handler"""
        self.tools = tools
        if is_debug_enabled('tools'):
            self.debug_print("Available tools:")
            if tools:
                for tool in tools:
                    self.debug_print(f"- {tool.__name__}: {tool.__doc__}")
            else:
                self.debug_print("No tools available")

    def clear_tool_calls(self) -> None:
        """Clear all pending tool calls"""
        if is_debug_enabled('tools'):
            self.debug_print("Clearing tool calls")
        self.tool_calls = []

    def set_tool_calls(self, tool_calls: Optional[List[Any]]) -> None:
        """Set the pending tool calls"""
        self._last_tool_calls = self.tool_calls
        self.tool_calls = tool_calls or []
        if is_debug_enabled('tools'):
            self.debug_print(f"Set {len(self.tool_calls)} new tool calls")
            for call in self.tool_calls:
                self.debug_print(f"- Tool call: {call.function.name} with args: {call.function.arguments}")

    def has_pending_calls(self) -> bool:
        """Check if there are any pending tool calls"""
        has_calls = bool(self.tool_calls)
        if is_debug_enabled('tools'):
            self.debug_print(f"Has pending calls: {has_calls}")
        return has_calls

    def get_last_tool_calls(self) -> List[Any]:
        """Get the tool calls from the last request"""
        if is_debug_enabled('tools'):
            self.debug_print(f"Getting last {len(self._last_tool_calls)} tool calls")
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
            
        if is_debug_enabled('tools'):
            self.debug_print(f"Created tool definition for {callable.__name__}:")
            self.debug_print(json.dumps(tool_def, indent=2))
            
        return tool_def

    def get_tools_definitions(self) -> List[Dict[str, Any]]:
        """Get the tool definitions for all available tools"""
        if not self.tools:
            if is_debug_enabled('tools'):
                self.debug_print("No tools available")
            return []
        defs = [self.create_tool_def(tool) for tool in self.tools]
        if is_debug_enabled('tools'):
            self.debug_print(f"Generated {len(defs)} tool definitions")
        return defs

    async def execute_tool_call(self, tool_call: Any) -> str:
        """Execute a single tool call
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            str: The tool response
        """
        if not self.tools:
            if is_debug_enabled('tools'):
                self.debug_print("No tools available")
            return None
            
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        if is_debug_enabled('tools'):
            self.debug_print(f"Executing tool: {tool_name}")
            self.debug_print(f"Arguments: {tool_args}")
            
        tool = next((t for t in self.tools if t.__name__ == tool_name), None)
        if not tool:
            if is_debug_enabled('tools'):
                self.debug_print(f"Tool not found: {tool_name}")
            return str(None)
            
        try:
            response = await tool(**tool_args)
            if is_debug_enabled('tools'):
                self.debug_print(f"Tool response: {response}")
            return str(response) if response is not None else str(None)
        except Exception as e:
            if is_debug_enabled('tools'):
                self.debug_print(f"Error executing tool: {e}")
            return str(None)

    async def execute_tool_calls(self) -> Optional[str]:
        """Execute all pending tool calls
        
        Returns:
            str: Combined tool responses
        """
        if not self.has_pending_calls():
            if is_debug_enabled('tools'):
                self.debug_print("No pending tool calls to execute")
            return None

        results = []
        for tool_call in self.tool_calls:
            response = await self.execute_tool_call(tool_call)
            if response is not None:
                results.append(response)

        return "\n".join(results)

    def get_pending_calls(self) -> List[Any]:
        """Get the list of pending tool calls"""
        return self.tool_calls

    def clear_pending_calls(self) -> None:
        """Clear the list of pending tool calls"""
        self.clear_tool_calls()

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
            if is_debug_enabled('tools'):
                self.debug_print(f"Calling tool {tool_name} with args: {args}")
            return await tool(**args)
        except Exception as e:
            if is_debug_enabled('tools'):
                self.debug_print(f"Error executing {tool_name}: {str(e)}")
            return f"Error executing {tool_name}: {str(e)}"