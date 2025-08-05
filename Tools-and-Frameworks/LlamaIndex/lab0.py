"""
For building llm powered applications, we requires different components integration. So, How to find and install the dependencies for the component we need ?

```terminal
pip install llama-index-{component-type}-{framework-name}
```

Ex : Install the dependencies for an LLM and embedding component using [HuggingFace API  Interface Integration](https://llamahub.ai/l/llms/llama-index-llms-huggingface-api?from=llms)

```
pip install llama-index-llms-huggingface-api llama-index-embeddings-huggingface
```

"""

#%%
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Retrieve HF_TOKEN from the environment variables
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen3-235B-A22B",
    token=hf_token,
    provider="together",  # this will use the best provider available
)

#%%
response = llm.complete("Who is the current prime minister of india?")
print(response)

#%% How to load the data from a folder using SimpleDirectoryReader
from llama_index.core import  SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir = "data")
documents = reader.load_data()
print(documents)
#%% How to breake the loaded docs into smaller chunks called Node Objects ?
from llama_index.core import Document 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

# create the pipeline transformation
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')
    ]
)

nodes = await pipeline.arun(documents=[Document.example()])
print(nodes)
#%% How to store our vector embedding of our documents ?
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

db = chromadb.PersistentClient(path='./llama_agent_db')
chroma_collection = db.get_or_create_collection('llama_agent')
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')
    ],
    vector_store=vector_store,
)

#%% How to index the node objects to make them searchable ?
# > Vector Embedding both query and nodes in same vector space, we can find the relevent matches
# > VectorStoreIndex : We needed to use the same embedding model here, as during ingestion to ensure consitency. 
# How to create the index from our vector store and embedding ?

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

print(index)
"""
All information automatically persisted within the ChromaVectorStore object and
passed directory path.

Now, we have loaded, chunks as node, create embedding of nodes and indexing and store them into chromadb database.
"""
#%% How to query a VectorStoreIndex with prompts and LLMs ?
# Converting our index into query interface for making query. There are many conversion options

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="",
)
query_engine.query("What is the bhagwat gita?")
# The meaning of life is 42
# %% How to evaluate the RAG response ?
from llama_index.core.evaluation import FaithfulnessEvaluator

query_engine = index.as_query_engine(
    llm = llm,
    response='compact'
    )# from the previous section
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct") # from the previous section

# query index
evaluator = FaithfulnessEvaluator(llm=llm)
response = query_engine.query(
    "What battles took place in New York City in the American Revolution?"
)
eval_result = evaluator.evaluate_response(response=response)
eval_result.passing
# %%
