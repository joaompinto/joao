from os import getenv
from openai import OpenAI
from typing import Optional, List, Callable, Union, Iterator, Any
from .tools import ToolsHandler
from .debug import debug_print, is_debug_enabled
import json
from rich.console import Console

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

console = Console()

class Agent:
    def __init__(self, system_prompt: str = None, temperature: float = 0, tenant_prefix: str = None, debug: bool = False, api_key: str = None):
        """Initialize the agent with optional system prompt and tenant prefix.
        
        Args:
            system_prompt: Initial system prompt for the conversation
            temperature: Sampling temperature (0.0-2.0)
            tenant_prefix: Optional prefix for environment variables
            debug: Enable debug output
            api_key: Optional API key (if not provided, will look in environment variables)
        """
        prefix = f"{tenant_prefix}_" if tenant_prefix else ""
        
        self.api_key = api_key or getenv(f"{prefix}OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(f"Missing API key: must be provided via constructor or {prefix}OPENAI_API_KEY environment variable")
            
        self.base_url = getenv(f"{prefix}OPENAI_BASE_URL", DEFAULT_BASE_URL)
        self.model = getenv(f"{prefix}OPENAI_MODEL", DEFAULT_MODEL)
        self.debug = debug
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.system_prompt = system_prompt  # Store the system prompt
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self.tools_handler = ToolsHandler()
        self.temperature = temperature

    def debug_print(self, *args, component='agent'):
        """Print debug information with the appropriate prefix"""
        if component == 'sent' and is_debug_enabled('sent'):
            print("[DEBUG SENT]", *args)
        elif component == 'agent' and is_debug_enabled('agent'):
            print("[DEBUG AGENT]", *args)

    def request(
        self, 
        message: str, 
        tools: Optional[List[Callable]] = None, 
        auto_use_tools: bool = True,
        stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """Send a message to the chat model with optional function calling tools.
        
        Args:
            message: The message to send
            tools: Optional list of callable functions to be used as tools
            auto_use_tools: If True, automatically execute any tool calls
            stream: If True, stream the response token by token
        
        Returns:
            Union[str, Iterator[str]]: The model's response as a string or token iterator
        """
        if is_debug_enabled('sent'):
            self.debug_print("======= Sending message to model", component='sent')
            if self.system_prompt:
                self.debug_print("System prompt:", component='sent')
                self.debug_print(self.system_prompt, component='sent')
            self.debug_print("Message:", component='sent')
            self.debug_print(message, component='sent')
            if tools:
                self.debug_print("Tools:", component='sent')
                for tool in tools:
                    self.debug_print(f"- {tool.__name__}", component='sent')
            self.debug_print("======= End of message", component='sent')

        if is_debug_enabled('agent'):
            self.debug_print("Making API call to:", self.base_url)
            self.debug_print("Model:", self.model)
            self.debug_print("Message:", message)
            if tools:
                self.debug_print("Tools enabled:", [t.__name__ for t in tools])
            self.debug_print("Stream mode:", stream)
            self.debug_print("Auto use tools:", auto_use_tools)
            self.debug_print("Current conversation length:", len(self.messages))
            
        # Set up tools if provided
        self.tools_handler.set_tools(tools)
        tools_definitions = self.tools_handler.get_tools_definitions()
        
        # Add user message
        self.messages.append({"role": "user", "content": message})
        
        try:
            if is_debug_enabled('agent'):
                self.debug_print("Sending request to model...")
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=tools_definitions if tools else None,
                stream=stream,
                n=1,
                temperature=self.temperature,
            )
            
            if is_debug_enabled('agent'):
                self.debug_print("Received response from model")
            
            if stream:
                return self._stream_response(response)
                
            answer = response.choices[0].message
            self.messages.append(answer)
            
            has_content = answer.content is not None and answer.content.strip() != ""
            has_tool_calls = hasattr(answer, 'tool_calls') and answer.tool_calls
            
            if is_debug_enabled('sent'):
                self.debug_print("======= Model Response =======", component='sent')
                if has_content:
                    self.debug_print(f"Content: {answer.content}", component='sent')
                if has_tool_calls:
                    self.debug_print(f"Tool calls requested: {len(answer.tool_calls)}", component='sent')
                    for tc in answer.tool_calls:
                        self.debug_print(f"Tool: {tc.function.name}", component='sent')
                        self.debug_print(f"Arguments: {tc.function.arguments}", component='sent')
                self.debug_print("======= End Model Response =======", component='sent')
            
            if has_tool_calls:
                if is_debug_enabled('agent'):
                    self.debug_print("Model requested tool calls:", len(answer.tool_calls))
                self.tools_handler.set_tool_calls(answer.tool_calls)
                if auto_use_tools:
                    if is_debug_enabled('agent'):
                        self.debug_print("Auto-executing tool calls...")
                    tool_response = self.use_tools(auto_update=True)
                    if has_content:
                        return f"{answer.content}\n\n{tool_response}"
                    return tool_response
            
            return answer.content
            
        except Exception as e:
            if is_debug_enabled('agent'):
                self.debug_print(f"Error in request: {str(e)}")
            return f"Error: {str(e)}"

    def _stream_response(self, response_stream):
        """Process streaming response and yield tokens."""
        if is_debug_enabled('agent'):
            self.debug_print("Starting to stream response...")
            
        for chunk in response_stream:
            if chunk.choices[0].delta.content is not None:
                if is_debug_enabled('agent'):
                    self.debug_print("Received chunk:", chunk.choices[0].delta.content)
                yield chunk.choices[0].delta.content
                
        if is_debug_enabled('agent'):
            self.debug_print("Finished streaming response")

    def use_tools(self, auto_update=True):
        """Execute all pending tool calls and optionally get model's response
        
        Args:
            auto_update (bool): If True, adds tool responses to messages and gets model's response
        
        Returns:
            str: The model's response if auto_update=True, otherwise the tool response
        """
        if not self.tools_handler.has_pending_calls():
            return None

        if is_debug_enabled('agent'):
            self.debug_print("Executing tool calls...")

        # Always execute all pending tool calls
        responses = []
        tool_calls = self.tools_handler.get_pending_calls()
        for tool_call in tool_calls:
            response = self.tools_handler.execute_tool_call(tool_call)
            responses.append(response)
            
            if auto_update:
                # Add assistant's tool call to conversation
                self.messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call]
                })
                # Add tool response to conversation
                self.messages.append({
                    "role": "tool",
                    "content": response,
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name
                })

        # Clear pending calls after execution
        self.tools_handler.clear_pending_calls()
        
        if auto_update:
            # Get model's response to tool results
            if is_debug_enabled('agent'):
                self.debug_print("Getting model's response to tool results...")
                self.debug_print("Messages to send:")
                for msg in self.messages:
                    self.debug_print(f"[{msg['role']}]: {msg.get('content', '')}")
                    if msg.get('tool_calls'):
                        self.debug_print(f"  Tool calls: {len(msg['tool_calls'])}")
                        for tc in msg['tool_calls']:
                            self.debug_print(f"  - {tc.function.name}: {tc.function.arguments}")
                    if msg.get('tool_call_id'):
                        self.debug_print(f"  Tool response for: {msg['tool_call_id']}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system_prompt}] + self.messages,
                n=1,
                temperature=self.temperature,
            )
            
            answer = response.choices[0].message
            if is_debug_enabled('sent'):
                self.debug_print("======= Model's Response to Tools =======", component='sent')
                if answer.content:
                    self.debug_print(f"Content: {answer.content}", component='sent')
                if hasattr(answer, 'tool_calls') and answer.tool_calls:
                    self.debug_print(f"Tool calls requested: {len(answer.tool_calls)}", component='sent')
                    for tc in answer.tool_calls:
                        self.debug_print(f"Tool: {tc.function.name}", component='sent')
                        self.debug_print(f"Arguments: {tc.function.arguments}", component='sent')
                self.debug_print("======= End Response =======", component='sent')
            
            self.messages.append({
                "role": "assistant",
                "content": answer.content or ""
            })
            
            # If model requests more tool calls, handle them recursively
            if hasattr(answer, 'tool_calls') and answer.tool_calls:
                if is_debug_enabled('agent'):
                    self.debug_print("Model requested more tool calls, handling recursively...")
                self.tools_handler.set_tool_calls(answer.tool_calls)
                return self.use_tools(auto_update=True)
            
            return answer.content or "\n".join(responses)
        
        return "\n".join(responses)

    def prepare_call(self, tool_call: Any) -> Optional[Callable]:
        """Prepare a tool call by finding the matching function from supported tools.
        
        Args:
            tool_call: The tool call object from the model response
            
        Returns:
            Optional[Callable]: The matching tool function if found, None otherwise
        """
        if not self.tools_handler.tools:
            return None
            
        tool_name = tool_call.function.name
        
        # Find the matching tool function
        for tool in self.tools_handler.tools:
            if tool.__name__ == tool_name:
                return tool
        return None
        
    def tool_exec(self) -> Any:
        """Execute the current tool call with its arguments.
        
        Returns:
            Any: The result of the tool execution
        """
        if not self.tools_handler.has_pending_calls():
            return None
            
        tool_call = self.tools_handler.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        
        # Get the tool function we prepared
        tool = self.tools_handler.tools[0]  # Should be set by prepare_call
        
        # Execute the tool and add the result to messages
        result = tool(**args)
        
        # Add the result to conversation history
        self.messages.append({
            "role": "assistant",
            "content": str(result)
        })
        
        return result

    def check_last_request(self):
        """View any pending tool calls from the last request"""
        return self.tools_handler.get_last_tool_calls()
