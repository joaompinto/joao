from os import getenv
from openai import AsyncOpenAI
from typing import Optional, List, Callable, Union, AsyncIterator
from .tools import AsyncToolsHandler

DEFAULT_BASE_URL = "https://api.openai.com/v1/"
DEFAULT_MODEL = "gpt-3.5-turbo"

class AsyncAgent:
    def __init__(self, system_prompt=None, temperature=0.7, tenant_prefix=None):
        """Initialize the agent with optional system prompt and tenant prefix."""
        prefix = f"{tenant_prefix}_" if tenant_prefix else ""
        
        self.api_key = getenv(f"{prefix}OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(f"Missing {prefix}OPENAI_API_KEY environment variable")
            
        self.base_url = getenv(f"{prefix}OPENAI_BASE_URL", DEFAULT_BASE_URL)
        self.model = getenv(f"{prefix}OPENAI_MODEL", DEFAULT_MODEL)
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.system_prompt = system_prompt  # Store the system prompt
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self.tools_handler = AsyncToolsHandler()
        self.temperature = temperature

    async def request(
        self, 
        message: str, 
        tools: Optional[List[Callable]] = None, 
        use_tools: bool = False,
        stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """Send a message to the chat model with optional function calling tools.
        
        Args:
            message: The message to send
            tools: Optional list of callable functions to be used as tools
            use_tools: If True, automatically execute any tool calls in the response
            stream: If True, stream the response token by token
        
        Returns:
            Union[str, AsyncIterator[str]]: The model's response as a string or token iterator
        """
        self.messages.append({"role": "user", "content": message})
        
        if tools:
            self.tools_handler.set_tools(tools)
            tools_definitions = self.tools_handler.get_tools_definitions()
            kwargs = {
                "model": self.model,
                "n": 1,
                "messages": self.messages,
                "temperature": self.temperature,
                "stream": stream,
                "tools": tools_definitions
            }
        else:
            kwargs = {
                "model": self.model,
                "n": 1,
                "messages": self.messages,
                "temperature": self.temperature,
                "stream": stream
            }
            
        if stream:
            return self._stream_response(await self.client.chat.completions.create(**kwargs))
            
        response = await self.client.chat.completions.create(**kwargs)
        answer = response.choices[0].message
        self.messages.append(answer)
        self.tools_handler.set_tool_calls(answer.tool_calls)
        
        if use_tools and answer.tool_calls:
            return await self.use_tools(autoupdate=True)
            
        return answer.content
    
    async def _stream_response(self, response_stream) -> AsyncIterator[str]:
        """Process streaming response and yield tokens."""
        collected_message = {"role": "assistant", "content": ""}
        async for chunk in response_stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                collected_message["content"] += content
                yield content
        
        # After streaming is done, add the complete message to history
        self.messages.append(collected_message)
    
    async def use_tools(self, autoupdate=True):
        """Execute all pending tool calls
        
        Args:
            autoupdate (bool): If True, adds tool responses to messages and gets a new completion
        
        Returns:
            str: The model's new response if autoupdate is True, otherwise None
        """
        if not self.tools_handler.has_pending_calls():
            return None
            
        try:
            tool_response = await self.tools_handler.execute_tool_calls()
        except Exception as e:
            print(f"Error executing tool calls: {e}")
            return None
            
        if not autoupdate:
            return None
            
        # Add tool response and get new completion
        last_tool_calls = self.tools_handler.get_last_tool_calls()
        if not last_tool_calls:
            return None
            
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": last_tool_calls
        })
        self.messages.append({
            "role": "tool",
            "content": tool_response,
            "tool_call_id": last_tool_calls[0].id
        })
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                n=1,
                temperature=self.temperature,
            )
        except Exception as e:
            print(f"Error getting new completion: {e}")
            return None
            
        answer = response.choices[0].message
        self.messages.append(answer)
        return answer.content
    
    def check_last_request(self):
        """View any pending tool calls from the last request"""
        return self.tools_handler.get_last_tool_calls()
