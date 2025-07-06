import uuid
import hashlib
from typing import List, Optional
import httpx
import asyncio
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, HttpUrl

from nlp_logic import (
    classify_text,
    extract_entities,
    get_sentiment,
    summarize_text_simple,
    summarize_text_rag,
)


app = FastAPI(
    title="NLP Pipeline API",
    description="A  NLP service with classification, entity extraction, sentiment analysis, and RAG-powered summarization.",
)

in_memory_cache = {}
tasks_db = {}

class TextListRequest(BaseModel):
    texts: List[str]

class AsyncSummarizeRequest(BaseModel):
    text: str
    webhook_url: Optional[HttpUrl] = None

class ClassificationResponse(BaseModel):
    category: str

class Entity(BaseModel):
    name: str
    type: str

class EntitiesResponse(BaseModel):
    entities: List[Entity]

class SentimentResponse(BaseModel):
    sentiment: str

class SummaryResponse(BaseModel):
    summary: str

class AsyncSubmitResponse(BaseModel):
    task_id: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict | str] = None


def get_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

async def run_summary_task(task_id: str, text: str, webhook_url: Optional[str]):
    try:
        summary = await asyncio.to_thread(summarize_text_rag, text)
        tasks_db[task_id]['status'] = 'completed'
        tasks_db[task_id]['result'] = {"summary": summary}

        if webhook_url:
            print(f"Sending webhook for task_id: {task_id} to {webhook_url}")
            async with httpx.AsyncClient() as client:
                try:
                    await client.post(str(webhook_url), json=tasks_db[task_id], timeout=10)
                    print(f"Webhook sent successfully for task_id: {task_id}")
                except httpx.RequestError as e:
                    print(f"Error sending webhook for task_id {task_id}: {e}")

    except Exception as e:
        print(f"Error during async summary for task_id {task_id}: {e}")
        tasks_db[task_id]['status'] = 'failed'
        tasks_db[task_id]['result'] = {"error": str(e)}


@app.get("/", summary="Root", include_in_schema=False)
def root():
    return {"message": "NLP Pipeline API is running"}

@app.post("/classify", response_model=List[ClassificationResponse], summary="Classify a list of texts")
def api_classify(request: TextListRequest):
    results = []
    for text in request.texts:
        results.append(classify_text(text))
    return results

@app.post("/extract-entities", response_model=List[EntitiesResponse], summary="Extract entities from a list of texts")
def api_extract_entities(request: TextListRequest):
    results = []
    for text in request.texts:
        results.append(extract_entities(text))
    return results

@app.post("/sentiment", response_model=List[SentimentResponse], summary="Analyze sentiment of a list of texts")
def api_sentiment(request: TextListRequest):
    results = []
    for text in request.texts:
        results.append(get_sentiment(text))
    return results

@app.post("/summarize", response_model=List[SummaryResponse], summary="Summarize a list of texts (synchronous)")
def api_summarize(request: TextListRequest):
    results = []
    for text in request.texts:
        cache_key = get_cache_key(text)
        if cache_key in in_memory_cache:
            print(f"Cache hit. Returning summary for key: {cache_key[:10]}")
            results.append({"summary": in_memory_cache[cache_key]})
            continue
        
        summary = summarize_text_simple(text)
        
        in_memory_cache[cache_key] = summary

        results.append({"summary": summary})
    return results


@app.post("/summarize_async", response_model=AsyncSubmitResponse, status_code=202, summary="Submit text for asynchronous summarization")
def api_summarize_async(request: AsyncSummarizeRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {'status': 'processing', 'result': None}
    
    background_tasks.add_task(
        run_summary_task, 
        task_id, 
        request.text, 
        str(request.webhook_url) if request.webhook_url else None
    )   
    return {"task_id": task_id}

@app.get("/tasks/{task_id}/status", response_model=TaskStatusResponse, summary="Check the status of an asynchronous task")
def get_task_status(task_id: str):
    task = tasks_db.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatusResponse(task_id=task_id, status=task['status'], result=task.get('result'))
