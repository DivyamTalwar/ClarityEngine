import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

print("Attempting to connect to OpenAI...")

# Check if the key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\nERROR: OpenAI API key not found in .env file!")
else:
    print("API Key found. Initializing ChatOpenAI...")
    try:
        # Initialize the model
        llm = ChatOpenAI(
            model="usf1-mini",
            temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.us.inc/usf/v1/hiring", 
            default_headers={
                "x-api-key": os.environ.get("OPENAI_API_KEY") # This ensures the key is sent in the correct header
            }
)
        
        # Send a simple request
        print("Sending a test message to OpenAI...")
        response = llm.invoke("Hello, world!")
        
        print("\n✅ SUCCESS! Connection to OpenAI is working.")
        print("Response:", response.content)
        
    except Exception as e:
        print("\n❌ FAILED: An error occurred while contacting OpenAI.")
        print("----------------------------------------------------")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("----------------------------------------------------")
        print("\nTROUBLESHOOTING:")
        print("1. Double-check your API key in the .env file for typos.")
        print("2. Make sure your OpenAI account has credits or a valid payment method at platform.openai.com.")