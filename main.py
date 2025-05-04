import os
import base64
import requests
from openai import OpenAI

def main():
    print("Starting vision test with Gemma3:4b via Ollama...")
    
    # Initialize OpenAI client for Ollama
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # Ollama doesn't need a real API key but requires one
    )
    
    try:
        # Load and prepare the image
        image_path = "GettyImages.jpg"
        print(f"Loading image from: {image_path}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check if Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                raise ConnectionError("Ollama server is not responding")
            
            models = response.json().get("models", [])
            gemma_models = [m for m in models if "gemma" in m["name"].lower()]
            
            if not gemma_models:
                print("Warning: No Gemma models found. Make sure to pull with: ollama pull gemma3:4b")
            else:
                print(f"Found Gemma model: {gemma_models[0]['name']}")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Ollama server is not running. Start it with: ollama serve")
            
        # Encode image to base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        print("Sending message to the Gemma model...")
        
        # Create the vision message
        response = client.chat.completions.create(
            model="gemma3:4b",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful vision assistant that can describe images in detail."
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Can you describe the content of this image in detail?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_string}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        # Print the response
        print("\nGemma's response:")
        print(response.choices[0].message.content)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the image file exists in the correct location.")
    except ConnectionError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error details: {type(e).__name__} - {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure Ollama is running with: ollama serve")
        print("2. Confirm you have pulled the Gemma model: ollama pull gemma3:4b")
        print("3. Verify the model supports vision capabilities")
        print("4. Check if the image is valid and can be properly encoded")

if __name__ == "__main__":
    main()
