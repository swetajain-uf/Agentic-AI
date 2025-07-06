# Implementation of simple program for RAG
# RAG is an approach to indexing a set of documents so that user queries in the form of prompts
# can be answered.  The answers are then embellished by the LLM
import os
import pickle
import requests
from crewai import Agent, Task, Crew, Process
from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, load_index_from_storage, StorageContext
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

dotenv_path = "/Users/swetajain/Desktop/Agentic AI/CrewAI/.env"
load_dotenv(dotenv_path)


os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_STORAGE_PATH = "/Users/swetajain/Desktop/Agentic AI/LlamaIndexTool/ISM6485/IndexStorage"

if os.path.exists(INDEX_STORAGE_PATH):
    print("📂 Loading existing index from storage...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_PATH)
    index = load_index_from_storage(storage_context)
else:
    print("📂 Creating a new index and saving it...")
   # documents = SimpleDirectoryReader("/Users/swetajain/Desktop/Agentic AI/LlamaIndexTool/ISM6485").load_data()
    documents = SimpleDirectoryReader("/Users/swetajain/Desktop/Agentic AI/LlamaIndexTool/ISM6485",recursive=True).load_data()
    print(f"📄 Loaded {len(documents)} documents:")
    for doc in documents:
        print("-", doc.metadata.get("file_path", "Unknown file"))
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=INDEX_STORAGE_PATH)  # Save index to disk
 
query_engine = index.as_query_engine()

#create a class that wraps contify API that can be used as a tool in CrewAI.

class LlamaIndexQueryTool(BaseTool):
    name: str = "LlamaIndexQuery"
    description: str = "Searches indexed documents and retrieves relevant information."
    
    def _run(self, query: str):
        """Perform a query on indexed documents."""
        return query_engine.query(query).response

research_agent = Agent(
    role="Research Assistant",
    goal="Answer knowledge-based questions using a document database.",
    backstory="An AI assistant specializing in retrieving and summarizing information from research papers.",
    verbose=True,
    tools=[LlamaIndexQueryTool()]
)

research_task = Task(
    description="Find information about company competitive advantage, technology, and innovation by reading all the documents in the folder: competitive advantage, JP Morgan, Mutual Benefit Life, On, P&G, PNB, Snapper, Walmart Blockchain.",
    expected_output="A summary of the competitive advantage, technology and innovation of companies. Create a paragraph of detailed content for each company. I want data on all these companies: JP Morgan, Mutual Benefit Life, On, P&G, PNB, Snapper, Walmart Blockchain",
    agent=research_agent
)
editor_agent = Agent(
    role = "Editor Agent",
    goal = "Creates a formatted mark down file",
    backstory = "An AI assistant specializes in taking content and presenting it in a well formatted output.",
    verbose=True
)
editor_task = Task(
    description="Takes findings and content from the research agent and creates a summarizing table.",
    expected_output="A table outline each companies competitive advantage, technology and innovation. Create a table where each row is about one company. First columns is company name, parameters related to competitive advantage, industry, and specialized technology and innovation. After creating the table, create a paragraph of detailed content for each company. I want data on all these companies: JP Morgan, Mutual Benefit Life, On, P&G, PNB, Snapper, Walmart Blockchain",
    agent=editor_agent
)

crew = Crew(
    agents=[research_agent, editor_agent],
    tasks=[research_task, editor_task],
    process=Process.sequential,
    manager_llm=ChatOpenAI(model="gpt-4o", 
                           temperature=0.7),
    verbose=True  # Runs tasks one after another
)

result = crew.kickoff()
# print("\n🔍 Research Output:", result)
from IPython.display import Markdown
Markdown(result.raw)
