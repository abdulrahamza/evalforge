"""
analyzer.py — LLM-powered failure analysis for evaluation runs.
"""

import json
import os

import anthropic


_FALLBACK = {
    "summary": "Analysis unavailable",
    "key_failures": [],
    "recommendations": [],
}


def format_results_for_prompt(results: list[dict]) -> list[dict]:
    """Extract failed scenarios for the analysis prompt (max 5)."""
    failed = []
    for r in results:
        eval_data = r.get("evaluation", {})
        if eval_data.get("success"):
            continue
        failed.append({
            "prompt": r.get("scenario", {}).get("prompt", "")[:300],
            "output": r.get("output", "")[:300],
            "failure_type": eval_data.get("failure_type", "unknown"),
        })
        if len(failed) >= 5:
            break
    return failed


def analyze_results(
    failure_stats: dict,
    results: list,
    success_rate: float,
) -> dict:
    """
    Call Anthropic Claude to produce a structured failure analysis.

    Returns JSON with keys: summary, key_failures, recommendations.
    Falls back to a safe default on any error.
    """
    # Serialize results for the prompt
    if results and hasattr(results[0], "model_dump"):
        serialized = [r.model_dump() for r in results]
    elif results and hasattr(results[0], "dict"):
        serialized = [r.dict() for r in results]
    else:
        serialized = results

    failed_samples = format_results_for_prompt(serialized)

    system_prompt = "You are an expert AI evaluation analyst."

    user_prompt = (
        "Analyze these agent test results.\n\n"
        f"Success rate: {success_rate}%\n\n"
        f"Failure stats: {json.dumps(failure_stats)}\n\n"
        f"Sample failed scenarios ({len(failed_samples)} of total failures):\n"
        f"{json.dumps(failed_samples, indent=2)}\n\n"
        "Respond with ONLY valid JSON (no markdown, no explanation) in this exact format:\n"
        "{\n"
        '  "summary": "short overall assessment of agent reliability",\n'
        '  "key_failures": ["pattern 1", "pattern 2"],\n'
        '  "recommendations": ["actionable fix 1", "actionable fix 2"]\n'
        "}\n\n"
        "Rules:\n"
        "- Be concise and critical\n"
        "- Focus on patterns, not individual cases\n"
        "- Avoid generic advice like 'add more tests'\n"
        "- Limit to 3-5 items per list"
    )

    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if the model wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()

        analysis = json.loads(raw)

        # Validate expected keys
        if not all(k in analysis for k in ("summary", "key_failures", "recommendations")):
            return _FALLBACK

        return analysis

    except Exception:
        return _FALLBACK
