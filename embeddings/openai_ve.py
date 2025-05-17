import os

import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key)

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="What is the capital of Nigeria?",
)

print(response.data[0].embedding)
