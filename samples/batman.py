# Get a Gemini API Key as described at https://ai.google.dev/gemini-api/docs/api-key
# Set the environment variable OPENAI_API_KEY before running this script

from joao import Agent

def batmobile(location: str):
    """ Can be used to drive to some location """
    print(f"Driving you to {location}!")

agent =  Agent("You are Batman")
agent.request("Hello, can you drive me to Gotham City?", tools=[batmobile])
answer = agent.use_tools()
