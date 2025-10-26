"""Direct test of Gemini API to check correct endpoint format."""
import httpx
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    
    models_to_test = [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest"
    ]
    
    for model in models_to_test:
        print(f"\nTesting model: {model}")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": "Say hello"
                }]
            }]
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload)
                print(f"  Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"  SUCCESS! Reply: {data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')[:100]}")
                    return model
                else:
                    print(f"  Error: {response.text[:150]}")
        except Exception as e:
            print(f"  Exception: {e}")
    
    return None

if __name__ == "__main__":
    asyncio.run(test_gemini())
