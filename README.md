# AI for Engineers

### Core Components of an LLM Application

1. Adding RAG (Retrieval-Augmented Generation): This is a key technique for connecting an LLM to your specific data. The process includes:
   - Loading & Ingestion: Using connectors from LlamaHub to import data from various sources (PDFs, APIs, databases).
   - Indexing & Embedding: Structuring the data for efficient and relevant retrieval.
   - Storing: Saving the indexed data, often in a specialized vector store.
   - Querying: Retrieving the most relevant information to answer questions accurately.
2. Building Agents: These are described as LLM-powered "knowledge workers" that can interact with the world using a set of tools. Key aspects include:
   - Creating Agents: Building single agents that can perform tasks.
   - Using Tools: Integrating pre-built tools from the LlamaHub registry.
   - Advanced Features: Maintaining state, streaming output for user feedback, and incorporating a "human in the loop" for verification.
3. Building Workflows: This is a lower-level, event-driven framework for creating advanced and complex agentic systems.
   - Multi-Agent Systems: Using AgentWorkflow to enable multiple agents to collaborate.
   - Control Flow: Implementing logic like looping, branching, and concurrent execution.
   - State Management: Creating stateful workflows that remember information across steps.

## Books 

1. https://github.com/PacktPublishing/Building-LLM-Powered-Applications
2.
