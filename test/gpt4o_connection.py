import os
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

endpoint = "https://60099-m1xc2jq0-australiaeast.openai.azure.com/"
model_name = "gpt-4o"
deployment = os.getenv("GPT_DEPLOYMENT")

subscription_key = os.getenv("SUPSCRIPTION_KEY")
api_version = os.getenv("API_VERSION")

from openai import AzureOpenAI
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are an expert Filipino Chef.",
        },
        {
            "role": "user",
            "content": "How to make Ube bread?",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment
)

print(response.choices[0].message.content)


