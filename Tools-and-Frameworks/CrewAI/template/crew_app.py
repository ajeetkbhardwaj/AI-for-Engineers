from src.agents import yt_researcher, yt_writer
from src.tasks import research_task, write_task

from crewai import Crew, Process

# adding all the agents and tasks to the crew

crew = Crew(
    agents=[yt_researcher, yt_writer],
    tasks=[research_task, write_task],
    process=Process.sequential,  # Sequential task execution is default
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

# Start the task execution process
result = crew.kickoff(inputs={'topic': 'AI VS ML VS DL vs Data Science'})