# Get a Gemini API Key as described at https://ai.google.dev/gemini-api/docs/api-key
# Set the environment variable GEMINI_API_KEY before running this script
from pyfiglet import figlet_format

from qchat import QuickChat

def batmobile(location: str):
    """ Can be used to drive to some location """
    print(f"Driving you to {location}!")
    print(figlet_format(location))

snoopy = QuickChat("You are Batman")
snoopy.request("Hello, can you drive me to Gotham City?", tools=[batmobile])
snoopy.use_tools()
