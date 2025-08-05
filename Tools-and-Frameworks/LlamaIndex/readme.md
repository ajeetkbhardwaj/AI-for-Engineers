# LlamaIndex

What is LlamaIndex ?

LlamaIndex is  **a complete toolkit for creating LLM-powered agents over our data using indexes and workflows**

### 1. Introduction

How to build LLM powered agents using LlamaIndex toolkit ?

Building agents in LlamaIndex require following

1. Components - Basic building blocks in LlamaIndex like prompts, models, database and they helps to connect LlamaIndex with other tools and libraries.
2. Tools - Components with specific capabilities like searching, calculating or accessing external services. Thus, they are the building blocks that enables agents agents to perform tasks
3. Agents - Autonomous components that can use tools, make decisions, and coordinate tools usage to accomplish the task.
4. Workflows - agentic workflows are a way to structure agentic behaviour without the explicit use of agents.

What speciality LlamaIndex has ?

1. Clear Workflow System - Design workflow to help breakdown the tasks how agents should make decisions step by step using an event-driven and async-first syntax.
2. Advanced document persing with LlamaParse - Integration with external is seamless.
3. Many ready-to-use Components - It has many tested and reliable components like LLMs, Retrievers, Indexes and many more.
4. LlamaHub - Its a registry for many components, tools and agents.

What is LlamaHub ?

> LlamaHub is a registry of hundreds of integrations, agents and tools that you can use within LlamaIndex.

For building llm powered applications, we requires different components integration. So, How to find and install the dependencies for the component we need ?

```terminal
pip install llama-index-{component-type}-{framework-name}
```

Ex : Install the dependencies for an LLM and embedding component using [HuggingFace API  Interface Integration](https://llamahub.ai/l/llms/llama-index-llms-huggingface-api?from=llms)

```
pip install llama-index-llms-huggingface-api llama-index-embeddings-huggingface
```

how to find, install and use the integrations for the components
 how we can use them to build our own agents ?

### 2. Components

Problem-1 : LLMs are trained on general world knowledge but they are not trained on relevent and upto date data.

Solution : RAG solve the problem by finding and retrieving relevent information from our external relevent and upto date data source.

Note : Any agent needs a way to find and understand relevent data.
QueryEngine - a key component for building agentic RAG workflows in LlamaIndex, provides this capability.

How to combine the components to create RAG Pipeline ?
Creating a RAG Pipeline using components, requires 5 stages withing RAG

1. **Loading** : Getting our data from where it lives into our workflow. LlamaHub provides integration for our data.
2. **Indexing** : Creating data structure that allows for querying in data such as Vector Embedding of text data, to find the contextual relevent data based on the input prompt text properties.
3. **Storing** : After Embedding(indexing) of our data is created, we needed to store our index and metadata to abvoid having to re-index it.
4. **Querying** : Given indexing strategy, there are many ways we can utilize LLMs and LlamaIndex data structure to query, including sub-query, multi-step queries and hybrid strategies.
5. **Evaluation** : Evaluation provides objective measures of how accurate, faithful and fast your responses to queries are. How effective it is relative to the other strategies ?

##### Loading and Embedding Documents

There are 3 main ways to load data into LlamaIndex

1. SimpleDirectoryReader: A built-in loader for various file types from a local directory.
2. LlamaParse: LlamaParse, LlamaIndex’s official tool for PDF parsing, available as a managed API.
3. LlamaHub: A registry of hundreds of data-loading libraries to ingest data from any source.

**SimpleDirectoryReader** can load various file types from a folder and convert thm into Document object that LlamaIndex can work with.

How to breake the loaded docs into smaller chunks called Node Objects ?

**InstigationPipeline** can help to create nodes through 2 key transformations

1. SentenceSplitter : It breaks down documents into manageable chunks by splitting them at natural sentence boundaries.
2. HuggingFaceEmbedding : It converts each chunks into numberical embedding vector representation that captures the semantic meaning of text and llm can efficently process them.

How to organise our documents in a way that's more usefull for searching and analysis ?

How to index the node objects to make them searchable ?

Vector Embedding both query and nodes in same vector space, we can find the relevent matches

VectorStoreIndex : We needed to use the same embedding model here, as during ingestion to ensure consitency.

How to create the index from our vector store and embedding ?

How to query a VectorStoreIndex with prompts and LLMs ?

Converting our index into query interface for making query. There are many conversion options

as_retriever: For basic document retrieval, returning a list of NodeWithScore objects with similarity scores
as_query_engine: For single question-answer interactions, returning a written response
as_chat_engine: For conversational interactions that maintain memory across multiple messages, returning a written response using chat history and indexed context

Note : query_engine : It is commonly used for agent-like interactions, we pass in an LLM to query engine to use for the response.

Response Processing
Under the hood, the query engine doesn’t only use the LLM to answer the question but also uses a ResponseSynthesizer as a strategy to process the response. Once again, this is fully customisable but there are three main strategies that work well out of the box:

refine: create and refine an answer by sequentially going through each retrieved text chunk. This makes a separate LLM call per Node/retrieved chunk.
compact (default): similar to refining but concatenating the chunks beforehand, resulting in fewer LLM calls.
tree_summarize: create a detailed answer by going through each retrieved text chunk and creating a tree structure of the answer.
Take fine-grained control of your query workflows with the low-level composition API. This API lets you customize and fine-tune every step of the query process to match your exact needs, which also pairs great with Workflows
The language model won’t always perform in predictable ways, so we can’t be sure that the answer we get is always correct. We can deal with this by evaluating the quality of the answer.

Evaluation and observability
LlamaIndex provides built-in evaluation tools to assess response quality. These evaluators leverage LLMs to analyze responses across different dimensions. Let’s look at the three main evaluators available:

FaithfulnessEvaluator: Evaluates the faithfulness of the answer by checking if the answer is supported by the context.
AnswerRelevancyEvaluator: Evaluate the relevance of the answer by checking if the answer is relevant to the question.
CorrectnessEvaluator: Evaluate the correctness of the answer by checking if the answer is correct.


### 3. Using Tools in LlamaIndex

defining a clear set of tools is crucial to performance because a clear tool interface are easier for llms to use. There are 4 main types of tools in LlamaIndex

1. FunctionTool : It can convert any python function inot a tool that an agent can use
2. QueryEngineTool : A tool that lets agents use query engines
3. Toolspecs : set of tools created by community
4. Utility Tools : Special tools that help handle large amount of data from other tools.

##### FunctionTools : 

name and description of params of the python function

### References

1. https://docs.llamaindex.ai/en/stable/examples/llm/huggingface/
2. https://huggingface.co/learn/agents-course/unit2/llama-index/introduction
3. components guide : https://docs.llamaindex.ai/en/stable/module_guides/
4. RAG guide : https://docs.llamaindex.ai/en/stable/understanding/rag/
5. LlamaHub for more complex data sources. : https://docs.llamaindex.ai/en/stable/module_guides/loading/connector/
6. LlamaPerser for more complex data sources.: https://github.com/run-llama/llama_cloud_services/blob/main/parse.md
7. Overview of different Vector Stores supported by LlamaIndex : https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/
8.
