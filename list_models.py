"""List available Gemini models."""
import httpx
import asyncio
import os
import json
from dotenv import load_dotenv

load_dotenv()

async def list_models():
    api_key = os.getenv("GEMINI_API_KEY")
    
    # List models endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    print(f"Fetching models from: {url[:80]}...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                print(f"\nFound {len(models)} models:\n")
                for model in models:
                    name = model.get("name", "")
                    supported_methods = model.get("supportedGenerationMethods", [])
                    if "generateContent" in supported_methods:
                        print(f"✓ {name.replace('models/', '')}")
                        print(f"  Methods: {', '.join(supported_methods)}")
            else:
                print(f"Error Response: {response.text}")
                
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(list_models())
