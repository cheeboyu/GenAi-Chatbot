# ----------------------------------------
# Import necessary libraries
# ----------------------------------------
import os
from dotenv import load_dotenv  # Load environment variables from a .env file

# LangChain libraries for LLM integration and chaining
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter

# Converts a Markdown-formatted string into HTML
from markdown import markdown

# Flask libraries for web server
from flask import Flask, request, render_template, jsonify

# ----------------------------------------
# Load environment variables from .env
# ----------------------------------------
load_dotenv()

# Set LangChain-related environment variables (for API authentication and tracing)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# Enable LangChain tracing (optional, useful for dev)
os.environ["LANGCHAIN_TRACING_V2"] = "true"

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
        max_tokens=500  # Limit response size
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


@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Handles GET and POST requests:
    - GET: Show the empty form
    - POST: Process user input, run chatbot, and show the response
    """
    input_text = None
    output = None

    if request.method == 'POST':
        input_text = request.form.get('input_text', '').strip()

        if input_text:
            try:
                # Apply rate limiting to prevent abuse
                rate_limiter.acquire()

                # Pass user input to the chain and receive generated output
                output = chain.invoke({'question': input_text})

                # Convert markdown bold syntax (**bold**) into HTML <strong> tags
                output = markdown(output)

            except Exception as e:
                # Log the error
                app.logger.error(f"Error occurred: {e}")

                # Show user-friendly error message
                output = "Oops! Something went wrong. Please try again later."

    # Render the HTML page and pass input/output to it
    return render_template('index.html', input_text=input_text, output=output)


# ----------------------------------------
# Start the Flask development server
# ----------------------------------------
if __name__ == '__main__':
    app.run(debug=True)