"""
Multi-Agentic System for AI Agents 
@codebase : https://microsoft.github.io/autogen/0.2/docs/Use-Cases/agent_chat
"""

import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import DockerCommandLineCodeExecutor
from autogen import OpenAIChatCompletionClient

load_dotenv()

model_client = OpenAIChatCompletionClient(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key= os.getenv("DEEPSEEK_API_KEY"),
        )

# create an AssistantAgent instance named "assistant" with the LLM configuration.
assistant = AssistantAgent(name="assistant", llm_configs=model_client)

# create a UserProxyAgent instance named "user_proxy" with code execution on docker.
code_executor = DockerCommandLineCodeExecutor()
user_proxy = UserProxyAgent(name="user_proxy", code_execution_config={"executor": code_executor})

# two agent conversation 
# the assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""What date is today? Which big tech stock has the largest year-to-date gain this year? How much is the gain?""",
)