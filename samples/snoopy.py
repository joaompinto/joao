# Get a Gemini API Key as described at https://ai.google.dev/gemini-api/docs/api-key
# Set the environment variable OPENAI_API_KEY before running this script

from joao import Agent

def batmobile(location: str):
    """ Can be used to drive to some location """
    print(f"Driving you to {location}!")

snooy =  Agent("You are snoopy")
response = snooy.request("Who are your friends?")
print(response)