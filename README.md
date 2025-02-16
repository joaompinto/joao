# joao

A dual-purpose tool providing both a Python library for building chat agents and a ready-to-use CLI chat interface. Built with function calling capabilities and compatible with OpenAI-style APIs.

## Features

- **As a Library**:
  - Simple and clean API for chat interactions
  - Support for function calling (tools)
  - Both synchronous and asynchronous APIs
  - Streaming support for real-time responses
  - Compatible with any OpenAI-style API (tested with Gemini 2.0 Flash)

- **As a CLI**:
  - Interactive chat interface
  - Single prompt mode
  - Rich markdown formatting for output
  - Temperature control for response randomness
  - Configurable system prompts
  - Multiple API provider support

- Minimal dependencies

## Requirements

1. Python 3.7 or higher
2. An API key from one of these providers:
   - **Gemini (Recommended)**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **OpenAI**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

## Installation

1. Install the package:
   ```bash
   pip install joao
   ```

2. Set up environment variables:

   For Gemini (default):
   ```bash
   export OPENAI_API_KEY="your_gemini_api_key"
   ```

   For OpenAI:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export OPENAI_BASE_URL="https://api.openai.com/v1"
   export OPENAI_MODEL="gpt-3.5-turbo"  # or your preferred model
   ```

## Usage

### Command Line Options

```bash
# Start interactive chat mode (with system prompt)
joao -s "You are a helpful assistant"

# Start chat mode without system prompt
joao

# Single prompt mode with system prompt
joao -s "You are a helpful assistant" "What is the capital of France?"

# Single prompt mode without system prompt
joao "What is the capital of France?"

# Enable streaming output
joao -s "You are a storyteller" --stream "Tell me a story"

# Set temperature (0.0-2.0, default: 0)
joao -s "You are a creative writer" -t 0.7 "Tell me a creative story"

# Show raw output without markdown formatting
joao -s "You are a helpful assistant" --raw "Hello"

# Use different API provider
joao -s "You are a helpful assistant" -e OPENAI "Using OpenAI API"

# Show debug info
joao -s "You are a helpful assistant" --debug "Test message"
```

### Interactive Chat Commands

When in chat mode, you have access to these commands:
- `/reset` - Clear conversation history
- `/reset <prompt>` - Clear history and set new system prompt
- `Ctrl+C` - Exit chat session

### Configuration Examples

```python
from joao import Agent, AsyncAgent

# Basic configuration with API key
agent = Agent(
    system_prompt="You are a helpful assistant",
    api_key="your-api-key-here"
)

# Full configuration
agent = Agent(
    system_prompt="You are a helpful assistant",
    api_key="your-api-key-here",
    temperature=0.7,  # Control response randomness (0.0-2.0)
    tenant_prefix="AZURE",  # Use AZURE_OPENAI_BASE_URL, AZURE_OPENAI_MODEL etc.
    debug=True  # Enable debug output
)

# Async agent configuration
async_agent = AsyncAgent(
    system_prompt="You are a helpful assistant",
    api_key="your-api-key-here",
    temperature=0.7,
    tenant_prefix="OPENAI"  # Use OPENAI_BASE_URL, OPENAI_MODEL etc.
)
```

The API key can be provided in three ways (in order of precedence):
1. Directly in the constructor via `api_key` parameter
2. Through environment variable `OPENAI_API_KEY`
3. Through prefixed environment variable (e.g., `AZURE_OPENAI_API_KEY` if tenant_prefix="AZURE")

### Simple Chat API

```python
from joao import Agent

# Create an agent with a personality
agent = Agent(
    "You are Snoopy, the beloved Peanuts character",
    api_key="your-api-key-here"  # Optional - can also use environment variable
)

# Get a response
response = agent.request("Who are your friends?")
print(response)

# Get a streamed response
for token in agent.request("Tell me a story", stream=True):
    print(token, end="", flush=True)
print()  # Final newline
```

### Using Function Calling (Tools)

```python
from joao import Agent

def drive_to(location: str):
    """Drive to the specified location"""
    print(f"Driving to {location}!")

# Create an agent with tools
agent = Agent(
    "You are a helpful driver",
    api_key="your-api-key-here"  # Optional - can also use environment variable
)
response = agent.request(
    "Can you drive me to San Francisco?", 
    tools=[drive_to],
    auto_use_tools=True  # Auto-execute any tool calls
)
```

### Async Support

```python
from joao import AsyncAgent
import asyncio

async def main():
    agent = AsyncAgent(
        "You are a helpful assistant",
        api_key="your-api-key-here"  # Optional - can also use environment variable
    )
    response = await agent.request("Hello!")
    print(response)

    # With streaming
    async for token in agent.request("Tell me a story", stream=True):
        print(token, end="", flush=True)
    print()

asyncio.run(main())
```

## Environment Variables

- `OPENAI_API_KEY` - Your API key
- `OPENAI_BASE_URL` - API endpoint (default: Gemini endpoint)
- `OPENAI_MODEL` - Model to use (default: gemini-2.0-flash)

You can prefix these with any string by using the `-e` flag, e.g., `-e AZURE` will look for `AZURE_OPENAI_API_KEY`
