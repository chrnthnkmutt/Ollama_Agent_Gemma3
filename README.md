# Ollama Agent with Gemma3 and Pydantic

## Project Description

Ollama Agent with Gemma3 and Pydantic is a project designed to process images using Ollama's API. The main functionality of the project is to analyze and describe the content of images by sending them to the Ollama API and receiving a response. This project leverages the capabilities of the Ollama model to provide detailed descriptions of images.

## Purpose

The purpose of this project is to demonstrate the use of Ollama's API for image processing. It showcases how to integrate the API with a Python application to send images and receive descriptive responses.

## Dependencies

The project has the following dependencies, which are listed in the `requirements.txt` file:

- `pydantic>=2.0.0`
- `aiohttp>=3.8.0`

## Setup Instructions

To set up the project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

To run the project, follow these steps:

1. Ensure that Ollama is running by executing the following command:
   ```bash
   ollama serve
   ```
2. Make sure you have the required model by executing:
   ```bash
   ollama pull gemma3:4b
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

## Main Functionality

The main functionality of the project is to process images using Ollama's API. The `main.py` script initializes the configuration, prepares the image, creates a structured multimodal message, and processes the message to get a response from Ollama. The response contains a detailed description of the image content.
