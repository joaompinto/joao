from joao import Agent
from typing import List, Dict, Any
import os

def search_web(query: str) -> List[Dict[str, Any]]:
    """Search the web for the given query and return a list of results.
    
    Args:
        query: The search query string
        
    Returns:
        List of dictionaries containing search results with 'title' and 'url' keys
    """
    # This is a mock implementation
    return [
        {"title": "Example Result 1", "url": "https://example.com/1"},
        {"title": "Example Result 2", "url": "https://example.com/2"},
    ]

def main():
    # Define supported tools
    supported_tools = [search_web]
    
    # Create an agent that will use our search tool
    agent = Agent(
        "You are a helpful research assistant. When asked to search, use the search_web tool."
    )
    
    # First, let's see what the agent wants to do
    response = agent.request(
        "Can you search for information about Python type hints?",
        tools=supported_tools
    )
    print("\nInitial response from agent:")
    print(response or "No response")
    
    # Check if there are any pending tool calls
    if agent.tools_handler.has_pending_calls():
        print("\nAgent wants to use tools. Here are the calls:")
        
        # Process each tool call
        for tool_call in agent.tools_handler.tool_calls:
            print(f"\nTool: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")
            
            # Get the tool function
            tool = agent.prepare_call(tool_call)
            if tool:
                # Execute the tool
                result = agent.tool_exec()
                print(f"\nSearch results:")
                for item in result:
                    print(f"- {item['title']}: {item['url']}")
            
        # Get the final response
        response = agent.request(
            "Here are the search results. Please summarize them.",
            tools=supported_tools
        )
    
    print("\nFinal response from agent:")
    print(response or "No response")

if __name__ == "__main__":
    main()
