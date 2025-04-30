# ----------------------------------------
# Import necessary libraries
# ----------------------------------------
import os
import time
import logging
from dotenv import load_dotenv
from flask import Flask, request, render_template, Response, stream_with_context
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter

# ----------------------------------------
# Load environment variables from .env
# ----------------------------------------
load_dotenv()

# Set LangChain tracing (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ----------------------------------------
# Set up logging
# ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ----------------------------------------
# Initialize Flask application
# ----------------------------------------
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching of static files

# ----------------------------------------
# Set up rate limiter
# ----------------------------------------
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
        streaming=True  # Enable streaming
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
                    logging.info(f"Chunk received: {chunk}")  # Log each chunk
                    full_response += chunk  # Append the chunk to the full response

                    # Yield chunks to the client as they are received
                    yield f"data: {chunk}\n\n"
                    time.sleep(0.1)  # Slight delay to simulate streaming

                # Log the complete response
                logging.info(f"Full response: {full_response}")

                # Wrap the complete response in markdown tags for frontend rendering
                yield f"data: <markdown>{full_response}</markdown>\n\n"
                yield "event: done\ndata: end\n\n"  # Indicate the end of the stream
                return  # Exit after sending the full response
            except Exception as e:
                logging.error(
                    f"Streaming error for question '{question}': {e}")
                yield f"data: Error: An unexpected error occurred.\n\n"
                return

    return Response(stream_with_context(generate()), content_type='text/event-stream')


# ----------------------------------------
# Run Flask app
# ----------------------------------------
if __name__ == '__main__':
    app.run(debug=True, threaded=True)