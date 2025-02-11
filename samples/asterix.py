# Get a Gemini API Key as described at https://ai.google.dev/gemini-api/docs/api-key
# Set the environment variable OPENAI_API_KEY before running this script

from pydoc import resolve
from joao import Agent

def search_for(person: str) -> str:
    """ Can be used to look for someone
    returns the location of the person
    """
    if person == "Obelix":
        return "Near the barbecue"
    return "I am unable to see it"


agent =  Agent("You are Asterix")
response = agent.request("Where is the IdeaFix and Obelix?", tools=[search_for], use_tools=True)
print(response)
