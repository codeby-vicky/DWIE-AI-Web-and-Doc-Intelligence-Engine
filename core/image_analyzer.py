import base64
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "llava:7b"  # Make sure: ollama pull llava


def analyze_image_with_ollama(image_bytes):
    """
    Sends extracted image to Ollama vision model
    Returns textual description
    """

    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": VISION_MODEL,
                "prompt": "Describe this image, graph, or diagram in detail. Extract key insights.",
                "images": [base64_image],
                "stream": False,
            },
            timeout=120,
        )

        if response.status_code != 200:
            return "Image analysis failed."

        result = response.json()

        return result.get("response", "Image analysis failed.")

    except Exception as e:
        return f"Image analysis error: {str(e)}"
