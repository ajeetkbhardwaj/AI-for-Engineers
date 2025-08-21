# LLM-powed applications used in production with focus on augmenting model and deploying them

# Table of Contents


# LlamaIndex -

What are the use of the LlamaIndex ?

1. Prompting
2. RAG Systems
3. AI-ChatBots
4. Structured Data Extraction from Unstructured Data
5. Fine-Tuning Models for Specific Tasks
6. Multi-Model Applications
7. AI-Agents

# AutoGen - AI Agentic Framework

> AutoGen provide abstraction to build llm application with conversable-multi-agents desiged to solve complex task through inter agentic conversations.
>
> 1. Conversable - Agents can do inter conversation among them via sending and recieving messages.
> 2. Customization - Agents can be customized to integrate llms, tools, human or a combination of them

What are the use of AutoGen ?

## Ch-1 : Running LLMs

How to run the LLMs ?

1. Locally
2. Cloud based API

What is an LLM API ?

A service that allows developers to integrate and interact with large language models like GPT, Claude, Llama, and others in their applications without having to manage the underlying model infrastructure themselves

The common llm api providers including OpenAI, Google(Gemini/VertexAI), HuggingFace, Anthropic(Cloude), Groq, AWS Bedrock many more...

These APIs power use cases like content generation, summarization, code completion, Q&A, translation, chatbots, and many more...

How the LLM API call works ?

1. **User through application sends a prompt** (text) to an API endpoint.
2. **API gateway** checks authentication/authorization via api key and forwards the prompt to the appropriate model service.
3. **Model generates a response** and processes the output.
4. **API formats and returns the result** to the application, often as JSON

What are the key-features needed to know before using llm api ?

1. Token based pricing - Most LLM APIs charge per number of tokens processed (input + output), where tokens are small chunks of text
2. Authentication - The use of llm api requires an API key generated through registration with the provider.
3. Rate limiting, monitoring, billing - API providers manage quotas, track use, and provide dashboards/analytics for usage.
4. Security and privacy - Sending data to remote models raises privacy concerns; API providers often outline compliance and data usage policies

What are the advantages of llm apis over local running model ?

1. Zero infrastructure - No need for local hardware, GPUs, or managing model parameters
2. Scalability - Handle many requests efficiently because managed by provider.
3. Speed to deployment - Quickly access the latest models and features.

## References

1. Running model locally and access them through local server : https://github.com/bentoml/OpenLLM
2. 

## Ch-2 : Building a Vector Storage

2.1 Ingesting documents
2.1 Splitting documents
2.3 Embedding models
2.4 Vector databases

## Ch-3 : Retrieval Augmented Generation

3.2 Orchestrators
3.2 Retrievers
3.3 Memory
3.4 Evaluation

## Ch-4 : Advanced RAG

4.1 Query Construction
4.2 Agents and Tools
4.3 Post processing
4.4 Program LLMs

## Ch-5  : Agents

5.1 Agents fundamentals
5.2 Agent frameworks
5.3 multi-agents

## Ch-6 : Deployment of LLMs

6.1 local
6.2 demo
6.3 server
6.4 edge

Inference Optimization
Security

https://github.com/mlabonne/llm-course

https://huggingface.co/learn/cookbook/en/enterprise_cookbook_gradio

https://huggingface.co/learn/cookbook/en/enterprise_cookbook_gradio
