"""
clean_agent.py — A simple Anthropic-powered agent for EvalForge testing.

Run with:  uvicorn clean_agent:app --port 8000
"""

from dotenv import load_dotenv
load_dotenv()

import os

import anthropic
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Test Agent (Anthropic)")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class AgentRequest(BaseModel):
    input: str


class AgentResponse(BaseModel):
    output: str


@app.post("/run", response_model=AgentResponse)
async def run_agent(req: AgentRequest):
    """Process a prompt through Claude and return the response."""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            system="You are a helpful, concise AI assistant.",
            messages=[{"role": "user", "content": req.input}],
            temperature=0.4,
            max_tokens=512,
        )
        output = response.content[0].text
    except Exception as e:
        output = f"Error: {e}"

    return AgentResponse(output=output)
