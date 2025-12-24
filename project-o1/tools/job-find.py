import os
import sys
from pathlib import Path

# Add parent directory to path to import memory module
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from memory.memory import save_jobs, save_preferences, get_preferences
import requests

load_dotenv()

# Initialize global variables
CURRENT_ROLE = None
CURRENT_LOCATION = None
NEXT_PAGE_TOKEN = None

# Search Jobs
@tool
def search_jobs(query: str) -> str:
    """Search for job listings."""
    return f"Job listings for: {query}"


@tool
def job_role(role: str, location: str) -> str:
    """Search for jobs by role and location."""
    global CURRENT_ROLE, CURRENT_LOCATION, NEXT_PAGE_TOKEN

    CURRENT_ROLE = role
    CURRENT_LOCATION = location

    save_preferences(role, location)

    api_key = os.getenv("SERPAPI_API_KEY")
    url = (
        f"https://www.searchapi.io/api/v1/search"
        f"?api_key={api_key}"
        f"&engine=google_jobs"
        f"&q={role}+in+{location}"
    )

    response = requests.get(url).json()
    jobs_raw = response.get("jobs", [])
    NEXT_PAGE_TOKEN = response.get("pagination", {}).get("next_page_token")

    if not jobs_raw:
        return "No jobs found."

    jobs = [format_job(job) for job in jobs_raw[:5]]

    # 🔥 SAVE TO MEMORY
    save_jobs(role, location, jobs)

    output = []
    for job in jobs:
        output.append(
            f"""
Title: {job['title']}
Company: {job['company']}
Location: {job['location']}
Apply Link: {job['apply_link']}
"""
        )

    return "\n---\n".join(output)


@tool
def more_jobs() -> str:
    """Fetch more jobs for the previously searched role."""
    global NEXT_PAGE_TOKEN, CURRENT_ROLE, CURRENT_LOCATION

    if not NEXT_PAGE_TOKEN:
        return "No more jobs available."

    api_key = os.getenv("SERPAPI_API_KEY")
    url = (
        f"https://www.searchapi.io/api/v1/search"
        f"?api_key={api_key}&engine=google_jobs"
        f"&q={CURRENT_ROLE}+in+{CURRENT_LOCATION}"
        f"&next_page_token={NEXT_PAGE_TOKEN}"
    )

    response = requests.get(url).json()

    jobs = response.get("jobs", [])
    NEXT_PAGE_TOKEN = response.get("pagination", {}).get("next_page_token")

    if not jobs:
        return "No more jobs available."

    results = [format_job(job) for job in jobs[:5]]

    if NEXT_PAGE_TOKEN:
        results.append("\nWould you like to see more jobs?")
    else:
        results.append("\nNo more jobs available.")

    return "\n---\n".join(results)


def format_job(job: dict) -> dict:
    return {
        "title": job.get("title", "N/A"),
        "company": job.get("company_name", "N/A"),
        "location": job.get("location", "N/A"),
        "apply_link": job.get("apply_link", "N/A"),
    }


    
def system_prompt() -> str:
    return """
You are a job search agent.

You must follow this reasoning process for EVERY user request:

PLAN:
- Identify the user's intent (new search, more jobs, or stop)

ACT:
- Decide which tool to call (job_role or more_jobs)
- If no tool is needed, say so

OBSERVATION:
- Summarize what the tool returned (internally)

FINAL:
- Present job results clearly to the user

Rules:
- Always think using PLAN → ACT → OBSERVATION → FINAL
- NEVER expose raw internal reasoning
- Only expose FINAL to the user
- Use job_role for first search
- Use more_jobs only when user asks for more
- If no more jobs exist, say "No more jobs available"
"""


llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_agent(
    llm,
    [search_jobs, job_role, more_jobs],
    system_prompt = system_prompt(),
)

def main():
    prefs = get_preferences()

    if prefs:
        print("I remember your previous preferences:")
        for p in prefs:
            print("-", p)

    print("Do you want to see the jobs according to your preferences? (yes/no)")
    choice = input().strip().lower()
    print(f"pref: {prefs}")
    if choice == "yes" and prefs:
        last_pref = prefs[-1]
        role = last_pref["role"]
        location = last_pref["location"]
    else:
        role = input("Enter job role: ")
        location = input("Enter job location: ")


    
    # ✅ ALWAYS build user message
    user_message = f"Find {role} jobs in {location}"
    messages = [("user", user_message)]

    # ✅ ALWAYS run agent
    while True:
        response = agent.invoke({"messages": messages})
        assistant_msg = response["messages"][-1].content
        print(assistant_msg)

        user_input = input("\nType 'yes' to see more jobs or 'no' to exit: ").strip().lower()

        if user_input in ["yes", "more", "show more"]:
            messages.append(("user", "show more jobs"))
        else:
            print("Okay, ending job search.")
            break


if __name__ == "__main__":
    main()