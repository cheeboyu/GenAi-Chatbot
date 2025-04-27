
# Chatbot with LLama3

This is a simple chatbot application built using the LLama3 model from Meta. The chatbot is deployed using Flask and can be accessed via a web interface.

## Features

- Uses the LLama3 model from Langchain for natural language processing.
- Utilizes dotenv for managing environment variables.
- Implements a ChatPromptTemplate for defining user and system messages.
- Supports querying the chatbot with user input.
- Web-based interface for easy interaction.
- Uses Bootstrap for styling.

## Prerequisite

- You have to install [Ollama](https://ollama.com/download) in your system.
- After installing the Ollama you have to install llama3 by using this command

## Getting Started

## Installation

1. Run this command in your terminal:

   ```bash
   olllma pull llama3
   ```


2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. In `.env` file paste your Langchain API key.

4. Run this command:

   ```bash
   flask --app app.py run
   ```

5. Open your browser and go to `http://localhost:5000` to access the chatbot.

## Usage

- Enter your query in the input field and click "Submit."
- The chatbot will process your query and respond.