# memory.py
from mem0 import Memory
from dotenv import load_dotenv
import os
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "openai",
        "config": {"api_key": OPENAI_API_KEY, "model": "text-embedding-3-small"}
    },
    "llm": {
        "provider": "openai",
        "config": {"api_key": OPENAI_API_KEY, "model": "gpt-4o-mini"}
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333}
    }
}

mem_client = Memory.from_config(config)

USER_ID = "abhishrestha"


def save_preferences(role: str, location: str):
    response = mem_client.add(
        user_id=USER_ID,
        messages=[
            {
                "role": "user",
                "content": json.dumps({
                    "type": "job_preference",
                    "role": role,
                    "location": location
                })
            }
        ]
    )
    # print("\n=== Mem0 API Response (save_preferences) ===")
    # print(json.dumps(response, indent=2))
    # print("============================================\n")
    return response




def save_jobs(role: str, location: str, jobs: list):
    for job in jobs:
        response = mem_client.add(
            user_id=USER_ID,
            messages=[
                {
                    "role": "system",
                    "content": json.dumps({
                        "type": "job",
                        "role": role,
                        "location": location,
                        "title": job["title"],
                        "company": job["company"],
                        "apply_link": job["apply_link"]
                    })
                }
            ]
        )
        # print(f"\n=== Mem0 API Response (save_jobs - {job['title']}) ===")
        # print(json.dumps(response, indent=2))
        # print("=================================================\n")

def get_preferences():
    results = mem_client.search(
        user_id=USER_ID,
        query="preferred job role location"
    )

    role = None
    location = None

    for r in results.get("results", []):
        memory = r.get("memory", "").lower()

        if "preferred job role" in memory:
            role = r["memory"].replace("Preferred job role is", "").strip()

        if "preferred location" in memory:
            location = r["memory"].replace("Preferred location is", "").strip()

    if role and location:
        return [{"role": role, "location": location}]

    return []


if __name__ == "__main__":
    save_preferences("Backend Developer", "Berlin")
    print(get_preferences())
