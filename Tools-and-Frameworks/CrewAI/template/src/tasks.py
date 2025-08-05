from crewai import Task
from src.tools import yt_tool

from src.agents import yt_researcher, yt_writer

# task-1

research_task = Task(
    description=(
        "Identify the video {topic}."
        "Get detailed information about the video from the channel video."
    ),
    expected_output='A comprehensive 3 paragraphs long report based on the {topic} of video content.',
    tools=[yt_tool],
    agent=yt_researcher,
)

# task-2
write_task = Task(
    description=(
        "get the info from the youtube channel on the topic {topic}."
    ),
    expected_output='Summarize the info from the youtube channel video on the topic {topic} and create the content for the blog',
    tools=[yt_tool],
    agent=yt_writer,
    async_execution=False,
    output_file='article_post.md'  # Example of output customization
)