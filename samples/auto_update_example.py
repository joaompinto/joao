# Example of using tools with auto_update and auto_use_tools
# To enable debug output, set DEBUG=tools,agent
# Example: DEBUG=tools,agent python samples/auto_update_example.py

from joao import Agent

def search_location(place):
    """Search for a location and return a description"""
    if place == "Gotham City":
        return "A dark and brooding metropolis filled with gothic architecture and gargoyles"
    elif place == "Wayne Manor":
        return "A stately mansion on the outskirts of Gotham City"
    return "Location not found"

# Initialize agent with system prompt
agent = Agent(system_prompt="""You are a helpful assistant that can search for locations.
When responding to questions:
1. ALWAYS share your own knowledge first
2. Then use search_location to get more details
3. Return BOTH your knowledge and the search results in your response

For example, if asked about Metropolis, you would:
1. Say: "Metropolis is Superman's home city, known for its futuristic skyline."
2. Use search_location
3. Combine both: "Metropolis is Superman's home city, known for its futuristic skyline. According to my search: [search results]"
""")

print("\nExample 1: With auto_update=True")
# First request
response = agent.request(
    "Tell me about Gotham City - what do you know about it and what can you find out?",
    tools=[search_location],
    auto_use_tools=True
)
print("Response:", response)
