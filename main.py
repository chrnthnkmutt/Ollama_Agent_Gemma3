import os
import base64
import asyncio
import aiohttp
from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field


# Define Pydantic models for data validation
class ModelInfo(BaseModel):
    """Information about LLM capabilities"""
    vision: bool = True
    json_output: bool = False
    function_calling: bool = False
    family: str = "unknown"
    structured_output: bool = False


class OllamaConfig(BaseModel):
    """Configuration for Ollama API"""
    model: str
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"  # Not used by Ollama but included for compatibility
    model_info: ModelInfo


class TextContent(BaseModel):
    """Text content for messages"""
    type: Literal["text"] = "text"
    data: str


class ImageContent(BaseModel):
    """Image content for messages"""
    type: Literal["image"] = "image"
    data: bytes
    format: str = "jpeg"


class MultiModalMessage(BaseModel):
    """Message containing text and/or images"""
    content: List[Union[TextContent, ImageContent]]
    source: str


class AssistantResponse(BaseModel):
    """Response from the assistant"""
    role: str = "assistant"
    content: str


class VisionAgent:
    """Agent that processes images using Ollama's API"""
    
    def __init__(self, config: OllamaConfig):
        """Initialize the agent with configuration"""
        self.config = config
        self.name = "vision_agent"
    
    async def process_message(self, message: MultiModalMessage) -> AssistantResponse:
        """Process a multimodal message and return a response"""
        # Extract text and images from the message
        prompt = " ".join([item.data for item in message.content if item.type == "text"])
        image_items = [item for item in message.content if item.type == "image"]
        
        print(f"Processing request with {len(image_items)} image(s)")
        print(f"Prompt: {prompt}")
        
        try:
            # Prepare for API call
            print("\nSending request to Ollama...")
            headers = {"Content-Type": "application/json"}
            
            # Prepare base64 encoded images
            image_contents = []
            for item in image_items:
                encoded = base64.b64encode(item.data).decode('utf-8')
                image_contents.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
                })
            
            # Combine text and images into API payload
            payload = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "user", 
                        "content": [{"type": "text", "text": prompt}] + image_contents
                    }
                ],
            }
            
            # Make the API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.base_url}/chat/completions", 
                    headers=headers, 
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error (status {response.status}): {error_text}")
                    
                    result = await response.json()
                    return AssistantResponse(content=result["choices"][0]["message"]["content"])
                    
        except Exception as e:
            error_message = f"Error processing image with Ollama: {str(e)}"
            print(error_message)
            return AssistantResponse(content=f"Error: {str(e)}")


async def main():
    """Main entry point for the application"""
    print("=== Pydantic Vision Agent for Ollama ===")
    print("Using structured data validation with Pydantic")
    
    # Check Ollama requirements
    print("Prerequisites:")
    print("1. Make sure Ollama is running: ollama serve")
    print("2. Make sure you have the model: ollama pull gemma3:4b")
    
    # Initialize configuration
    config = OllamaConfig(
        model="gemma3:4b",
        model_info=ModelInfo(vision=True)
    )
    
    # Initialize the agent
    agent = VisionAgent(config)
    
    # Prepare the image
    image_path = "GettyImages.jpg"
    
    # Verify the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        print(f"Current working directory: {os.getcwd()}")
        print("Available files:")
        print(os.listdir("."))
        return
    
    # Load the image data
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Create a structured multimodal message
    message = MultiModalMessage(
        content=[
            TextContent(data="Can you describe the content of this image?"),
            ImageContent(data=image_data)
        ],
        source="user"
    )
    
    # Process the message and get response
    response = await agent.process_message(message)
    print("\nResponse from Ollama:")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())