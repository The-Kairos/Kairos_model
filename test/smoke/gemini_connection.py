# Run: python test/smoke/gemini_connection.py

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(".env")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables.")

with open(r"output\frames\scene_006\frame_01.jpg", "rb") as file_handle:
    image_bytes = file_handle.read()

client = genai.Client(vertexai=True, api_key=api_key)
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        ),
        """
        Before this scene: a video frame of a cartoon character sitting at a table
        Now: a video frame of a sponge sponge with a piece of paper
        After this scene: a video frame of a sponge sponge with a piece of paper

        Based on the image and this context, concisely describe what is happening in this frame, focusing on new details or clarifications
        """,
    ],
)

print(response.text)
