from crewai import Agent
from src.tools import yt_tool

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "gpt-4-0125-preview"

yt_researcher = Agent(
    role = "Detailed Aricle Researcher for Youtube Videos",
    goal = "Find the relevent video transcription for the topic {topic} for provided youtube channel",
    verbose = True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI Data Science , MAchine Learning And GEN AI and providing suggestion"
    ),
    tools = [yt_tool],
    allow_delegation=True
)

yt_writer = Agent(
    role = "Detailed Aricle Writer for Youtube Videos",
    goal = "Write a detailed article based on the video transcription for the topic {topic} from YT video",
    verbose = True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools = [yt_tool],
    allow_delegation=False
)