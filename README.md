﻿NLP Pipeline API
A FastAPI-based NLP service providing text classification, entity extraction, sentiment analysis, and advanced RAG-powered summarization.

Features
    Text Classification: Categorize text into predefined classes (e.g., Sports, Tech).
    Named Entity Recognition: Extract entities like People, Organizations, and Locations.
    Sentiment Analysis: Determine if text is Positive, Negative, or Neutral.
    Simple Summarization: A quick, synchronous summarization endpoint with caching.
    Advanced RAG Summarization: A powerful, asynchronous summarization task that uses a vector database (Pinecone) to find similar examples, improving summary quality. The results are fed back into the database to continuously improve.
    Asynchronous Task Handling: Submit long-running summarization jobs and check their status later, with optional webhook notifications.


1. Prerequisites
Before you begin, ensure you have the following:
Python 3.8+
A Pinecone Account:
You will need a free Pinecone account to store and retrieve vectors for the RAG summarizer.
Get your API Key from the Pinecone dashboard.
A Custom "OpenAI-compatible" API Key:
The project is configured to use a specific API endpoint: https://api.us.inc/usf/v1/hiring.
You will need an API Key that is valid for this service. A standard key from openai.com will not work.


2. Setup and Installation
Follow these steps to get your environment set up.
Step 1: Get the Code
Clone the repository or download the main.py and nlp_logic.py files into a new project directory.
git clone <your-repo-url>
cd <your-repo-name>

Step 2: Create a Python Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.
Generated bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate


Step 3: Install Dependencies
Create a file named requirements.txt in your project directory and add the following lines:
requirements.txt
fastapi
uvicorn[standard]
pydantic
httpx
python-dotenv
langchain
langchain-openai
langchain-pinecone
pinecone-client
langgraph

Now, install all the required packages with a single command:
Generated bash
pip install -r requirements.txt



Step 4: Set Up Environment Variables
Create a file named .env in the root of your project directory. This file will securely store your API keys.
.env
# Get this key from the custom service provider (not openai.com)
OPENAI_API_KEY="your_custom_api_key_for_api.us.inc"

# Get this key from your Pinecone dashboard
PINECONE_API_KEY="your_pinecone_api_key"


Step 5: Configure Pinecone Index
The RAG summarizer requires a specific index in your Pinecone account.
Log in to your Pinecone account.
Go to the Indexes page and click "Create Index".
Fill in the details exactly as follows:
Index Name: nlp-pipeline
Dimensions: 1536 (This is required for the text-embedding-3-small model used in the code).
Metric: cosine
Click "Create Index". It may take a minute or two for the index to be ready.
3. Running the Application
Once you have completed the setup, you can run the API server using uvicorn.

uvicorn main:app --reload


main: The file main.py.
app: The FastAPI object created in main.py.
--reload: This flag makes the server restart automatically after you make changes to the code.
The server will start, and you will see output similar to this:
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx]
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
Use code with caution.
You can now access the interactive API documentation (powered by Swagger UI) in your browser at:
http://127.0.0.1:8000/docs

Key Enhancements for Production Readiness
These are critical improvements to move the service from a prototype to a scalable, robust application.
    1. Externalize State with Redis
    Current Problem: The cache (in_memory_cache) and task database (tasks_db) are Python dictionaries. This means they are lost if the server restarts and are not shared if you run multiple copies of the application for scaling.
    The Enhancement: Replace both in_memory_cache and tasks_db with a Redis database.
    For Caching: Use Redis's key-value store with a TTL (Time-To-Live). This allows the cache to be shared across all application instances and for stale entries to expire automatically.
    For Task Tracking: Store task status and results in Redis using the task_id as the key. This provides a persistent and shared "database" for all asynchronous jobs.
    Benefit: This change makes the application truly stateless, enabling horizontal scaling (running multiple servers behind a load balancer) and making it resilient to restarts.
    2. Implement a Robust Task Queue with Celery
    Current Problem: FastAPI's BackgroundTasks is simple but not durable. If the application crashes while a task is running, the task is lost forever.
    The Enhancement: Replace BackgroundTasks with a dedicated task queue system like Celery, using Redis or RabbitMQ as the message broker.
    Benefit:
    Durability: Tasks are stored in the broker. If a worker process fails, the task can be automatically re-assigned to another worker.
    Independent Scaling: You can scale your API servers and your processing "workers" independently, which is much more efficient.
    Monitoring: Celery provides tools (like Flower) to monitor task progress, see failures, and manage the queue.

