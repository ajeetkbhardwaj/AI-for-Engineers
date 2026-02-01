<h1 align="center">LangGraph</h1>

## Table of Contents



What is the main differences between generative ai and agentic ai ?

What agentic ai is ?

Why langraph needed if langchain already there for building llm based applications ?

What is the main differences between the langchain and langgraph ?


### Ch-1 :

What is LangGraph ?

> 1. An orchestration framework for building intelligent, statefull and multi-step llm based agentic workflow.
> 2. Features like parallelism, loops, branching, memory and resumability makes it ideal for agentic and production-grade AI applications.
> 3. It models our llm based agentic workflow logic as a graph of nodes(tasks) and edges(routing) instead of a linear chain.

What is LLM workflow ?

> * **A step-by-step process** used to build complex LLM applications.
> * Each step performs a  **distinct task** , such as prompting, reasoning, tool calling, memory access, or decision-making.
> * Workflows can be  **linear, parallel, branched, or looped** , enabling complex behaviors like retries, multi-agent communication, or tool-augmented reasoning.

Design a llm-based agentic workflow for Essay Evaluation System that utilizes a graph based workflow such as LangGraph ?

The system aims to generate an essay topic, collect a student's submission, evaluate it, and then either provide feedback for improvement or approve the essay.

### Ch-2 : 
How to make sequential workflow in langgraph ?
- Building understanding of creating basic workflow in langgraph.