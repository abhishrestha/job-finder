import os
from dotenv import load_dotenv
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
import requests
load_dotenv() 

@tool 
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

@tool 
def get_weather(location: str) -> str:
    """Get weather information."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.weatherstack.com/current?access_key={api_key}&query={location}&units=m"

    response = requests.get(url).json()
    if (response.get("error")):
        return "Could not retrive weather data"

    temp = response["current"]["temperature"]
    description = response["current"]["weather_descriptions"][0]

    return f"Weather in {location}: {description}, {temp}°C"



def system_prompt() -> str:
    return """You are a helpful assistant that can search for information and provide weather updates.
Use the provided tools to answer user queries effectively.If someone ask other than weather imformation simple say sorry. """

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_agent(
    llm,
    [search, get_weather],
    system_prompt = system_prompt(),
)

def main():
    user_input = input("Enter your query: ")
    response = agent.invoke({"messages": [("user", user_input)]})
    print(response["messages"][-1].content)

if __name__ == "__main__":
    main()
