"""
utils.py — Response normalization utilities.
"""

# Keys to probe, in priority order.
_KNOWN_KEYS = ("output", "response", "result", "message")


def extract_text_output(data) -> str:
    """
    Normalize an API response of any shape into a plain string.

    Checks common top-level keys ("output", "response", "result", "message")
    and returns the first match.  Falls back to ``str(data)`` so the function
    never raises and always returns a string.

    Args:
        data: Parsed JSON (dict/list), a string, or any other value.

    Returns:
        A non-empty string representation of the response.
    """
    try:
        if data is None:
            return ""

        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            for key in _KNOWN_KEYS:
                if key in data and data[key] is not None:
                    value = data[key]
                    return value if isinstance(value, str) else str(value)

        return str(data)

    except Exception:
        return str(data) if data is not None else ""
