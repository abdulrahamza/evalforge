"""
agent_runner.py — Lightweight client for calling external AI agent APIs.
"""

import requests


def call_agent(
    endpoint: str,
    prompt: str,
    input_field: str = "input",
    timeout: int = 15,
) -> dict:
    """
    Send a prompt to an external AI agent API via POST.

    Args:
        endpoint:    Full URL of the agent API.
        prompt:      The text prompt to send.
        input_field: JSON key used for the prompt payload (default "input").
        timeout:     Request timeout in seconds (default 15).

    Returns:
        dict with keys "output", "status", and "error".
    """
    result = {"output": None, "status": "failed", "error": None}

    try:
        response = requests.post(
            endpoint,
            json={input_field: prompt},
            timeout=timeout,
        )
        result["status"] = response.status_code

        if not response.content:
            result["error"] = "Empty response body"
            return result

        try:
            result["output"] = response.json()
        except requests.exceptions.JSONDecodeError:
            result["output"] = response.text

    except requests.exceptions.ConnectionError:
        result["error"] = "Could not reach the API endpoint"
    except requests.exceptions.Timeout:
        result["error"] = "Request timed out"
    except requests.exceptions.RequestException as exc:
        result["error"] = str(exc)

    return result
