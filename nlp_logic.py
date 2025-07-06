import os
from typing import List, Dict, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

load_dotenv()

if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    raise ValueError("Missing OpenAI and Pinecone API keys ")

PINECONE_INDEX_NAME = "nlp-pipeline"

llm = ChatOpenAI(
    model="usf1-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.us.inc/usf/v1/hiring", 
    default_headers={
        "x-api-key": os.environ.get("OPENAI_API_KEY")
    }
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})


class Classification(BaseModel):
    category: str = Field(description="The category of the text.", enum=["Sports", "Technology", "Business", "Entertainment", "Science"])

def classify_text(text: str) -> Dict:
    parser = JsonOutputParser(pydantic_object=Classification)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a text classification expert. Classify the user's text into one of the following categories: Sports, Technology, Business, Entertainment, Science. Respond with a JSON object containing the category."),
        ("human", "{text}")
    ])
    chain = prompt | llm | parser
    return chain.invoke({"text": text})

class Entity(BaseModel):
    name: str = Field(description="The name of the extracted entity.")
    type: str = Field(description="The type of the entity (e.g., Person, Organization, Location, Product, Currency).")

class Entities(BaseModel):
    entities: List[Entity]

def extract_entities(text: str) -> Dict:
    parser = JsonOutputParser(pydantic_object=Entities)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in named entity recognition. Extract all relevant entities (People, Organizations, Locations, Products, Currencies) from the user's text. Respond with a JSON object containing a list of entities."),
        ("human", "{text}")
    ])
    chain = prompt | llm | parser
    return chain.invoke({"text": text})

class Sentiment(BaseModel):
    sentiment: str = Field(description="The sentiment of the text.", enum=["Positive", "Negative", "Neutral"])

def get_sentiment(text: str) -> Dict:
    parser = JsonOutputParser(pydantic_object=Sentiment)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the sentiment of the user's text. Respond with a JSON object indicating if the sentiment is Positive, Negative, or Neutral."),
        ("human", "{text}")
    ])
    chain = prompt | llm | parser
    return chain.invoke({"text": text})
    
def summarize_text_simple(text: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a skilled text summarizer. Provide a concise summary of the following text."),
        ("human", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


class GraphState(TypedDict):
    text_to_summarize: str
    retrieved_examples: List[str]
    final_summary: str

def retrieve_node(state: GraphState):
    text = state["text_to_summarize"]
    retrieved_docs = retriever.invoke(text)
    examples = [doc.page_content for doc in retrieved_docs]
    print(f"FOUND {len(examples)} EXAMPLES")
    return {"retrieved_examples": examples}


def summarize_node(state: GraphState):
    text_to_summarize = state["text_to_summarize"]
    examples = state["retrieved_examples"]
    
    example_str = "\n\n".join([f"EXAMPLE {i+1}:\n{ex}" for i, ex in enumerate(examples)])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert text summarizer. You create high-quality, concise summaries. \n"
                f"Here are some examples of summaries in the style you should adopt:\n\n"
                f"{example_str}\n\n"
                f"Now, using a similar style, summarize the following new article. Do not mention the examples in your output. Just provide the summary."),
        ("human", "{text}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"text": text_to_summarize})
    return {"final_summary": summary}

def build_rag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("summarize", summarize_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "summarize")
    workflow.add_edge("summarize", END)

    return workflow.compile()

rag_summarizer_app = build_rag_graph()

def summarize_text_rag(text: str) -> str:
    inputs = {"text_to_summarize": text}
    result_state = rag_summarizer_app.invoke(inputs)
    final_summary = result_state['final_summary']

    vectorstore.add_texts(
        texts=[final_summary], 
        metadatas=[{"original_text": text}]
    )
    
    return final_summary