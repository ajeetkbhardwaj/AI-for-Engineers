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

## Resources

### 1.  Machine Learning Systems

[0]. https://github.com/Nyandwi/machine_learning_complete/blob/main/010_mlops/1_mlops_guide.md
[1]. https://dcai.csail.mit.edu/
[2]. DataCentricAI : https://youtube.com/playlist?list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5&si=OyEQkZEEO3AqKw0j
[3]. https://madewithml.com/
[4]. https://stanford-cs329s.github.io/
[5]. https://fullstackdeeplearning.com/