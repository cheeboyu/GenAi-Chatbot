# ----------------------------------------
# Import necessary libraries
# ----------------------------------------
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from markdown import markdown
from flask import Flask, request, render_template, jsonify, Response, stream_with_context
import time
import logging

# ----------------------------------------
# Set up logging
# ----------------------------------------
logging.basicConfig(level=logging.ERROR)

# ----------------------------------------
# Load environment variables from .env
# ----------------------------------------
load_dotenv()

# api_key = os.getenv("LANGCHAIN_API_KEY")
# if not api_key:
#     raise ValueError("The LANGCHAIN_API_KEY environment variable is not set correctly.")
# os.environ["LANGCHAIN_API_KEY"] = api_key
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ----------------------------------------
# Initialize Flask application
# ----------------------------------------
app = Flask(__name__)
# Disable caching of static files (useful during development)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# ----------------------------------------
# Set up rate limiter
# ----------------------------------------
# Controls how many requests the server allows per second to prevent abuse
rate_limiter = InMemoryRateLimiter(requests_per_second=1, max_bucket_size=5)

# ----------------------------------------
# Function to initialize and build the chatbot
# ----------------------------------------


def initialize_chatbot():
    """
    Initializes and returns the LangChain chatbot components:
    - Creates a structured prompt
    - Sets up the language model
    - Defines how to parse output
    - Chains them together
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Provide accurate and concise responses to user queries."),
        ("user", "Question: {question}")
    ])

    # Initialize the language model (OllamaLLM)
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        streaming=True  # ✅ Enable streaming
    )

    # Parse the model's raw output into clean plain text
    output_parser = StrOutputParser()

    # Chain the prompt, model, and output parser together
    chain = prompt | llm | output_parser

    return chain


# ----------------------------------------
# Initialize chatbot once to reuse across all requests
# ----------------------------------------
chain = initialize_chatbot()

# ----------------------------------------
# Route for Home page ('/')
# Supports both GET (initial load) and POST (user submits question)
# ----------------------------------------

# Home route - serves HTML page
# ----------------------------------------


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# ----------------------------------------
# Streaming response route (SSE)
# ----------------------------------------


@app.route('/stream', methods=['GET'])
def stream():
    question = request.args.get('input_text', '').strip()

    def generate():
        if question:
            try:
                rate_limiter.acquire()
                full_response = ""  # To accumulate the full response text

                # Streaming model response
                for chunk in chain.stream({'question': question}):
                    print(chunk)  # Log to track the chunk
                    full_response += chunk  # Keep appending the chunks

                    # Yield chunks to the client as they are received
                    yield f"data: {chunk}\n\n"
                    time.sleep(0.1)

                print(full_response)  # Log the complete response

                # Wrap the complete response in markdown for better rendering on the front-end
                yield f"data: <markdown>{full_response}</markdown>\n\n"
                yield "event: done\ndata: end\n\n"  # End the streaming
                return  # Explicitly exit after sending the full response
            except Exception as e:
                logging.error(f"Streaming error for question '{question}': {e}")
                yield f"data: Error: An unexpected error occurred.\n\n"
                return

    return Response(stream_with_context(generate()), content_type='text/event-stream')


# ----------------------------------------
# Run Flask app
# ----------------------------------------
if __name__ == '__main__':
    app.run(debug=True, threaded=True)