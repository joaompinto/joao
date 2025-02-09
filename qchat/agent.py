from openai import OpenAI
from os import getenv

GEMINI_API_KEY=getenv("GEMINI_API_KEY")
OPENAI_BASE_URL=getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")


class QuickChat:

    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
        self.client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        self.tool_calls = []
        self.tools = None
        
    def request(self, message, tools=None):
        """
        Send a message to the chat model with optional function calling tools.
        
        Args:
            message: The message to send
            tools: Optional list of callable functions to be used as tools
        
        Returns:
            str: The model's response
        """
        self.tools = tools
        kwargs = {
            "model": "gemini-2.0-flash",
            "n": 1,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message}
            ]
        }
        
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": create_tool_def(tool)
                }
                for tool in tools
            ]
            
        response = self.client.chat.completions.create(**kwargs)
        answer = response.choices[0].message
        self.tool_calls = answer.tool_calls or []
        return response.choices[0].message.content

    def use_tools(self):
        """Execute all pending tool calls"""
        for tool_call in self.tool_calls:
            self._call_tool(tool_call, self.tools)
        self.tool_calls = []  # Clear the tool calls after executing them

    def check_last_request(self):
        """View any pending tool calls from the last request"""
        if not self.tool_calls:
            print("No pending tool calls")
            return
            
        for call in self.tool_calls:
            print(f"Function: {call.function.name}")
            print(f"Arguments: {call.function.arguments}")
            print()

    def _call_tool(self, tool_call, tools):
        """
        Execute a tool call from the model response
        
        Args:
            tool_call: The tool call object from the model
            tools: List of available tool functions
        """
        import json
        
        # Find the matching tool function
        tool_name = tool_call.function.name
        tool_fn = next((t for t in tools if t.__name__ == tool_name), None)
        
        if tool_fn is None:
            print(f"Warning: Tool {tool_name} not found")
            return
            
        # Parse arguments and call the function
        try:
            args = json.loads(tool_call.function.arguments)
            result = tool_fn(**args)
            #print(f"Tool {tool_name} result: {result}")
        except Exception as e:
            print(f"Error calling tool {tool_name}: {str(e)}")

def create_tool_def(callable):
    """
    Create a tool definition based on a function's signature and docstring.
    The definition follows the JSONSchema format used for tool descriptions.
    
    Args:
        callable: The function or method to create a tool definition for
        
    Returns:
        dict: A tool definition containing name, description, and parameters schema
    """
    import inspect
    from inspect import Parameter
    
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
        "name": callable.__name__,
        "description": callable.__doc__ or "No description available",
        "parameters": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    }
    
    return tool_def
