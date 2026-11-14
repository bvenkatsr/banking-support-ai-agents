import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"


def _read_secret(name):
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, "")


def get_api_key():
    return _read_secret("OPENAI_API_KEY").strip()


def get_model():
    return _read_secret("OPENAI_MODEL").strip() or DEFAULT_MODEL


def llm_available():
    return bool(get_api_key())


_client = None
_client_key = None


def _get_client():
    global _client, _client_key
    key = get_api_key()
    if not key:
        return None
    if _client is None or _client_key != key:
        from openai import OpenAI

        _client = OpenAI(api_key=key)
        _client_key = key
    return _client


def chat(system_prompt, user_prompt, json_mode=False, temperature=0.2):
    client = _get_client()
    trace = {
        "model": get_model(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "json_mode": json_mode,
    }
    if client is None:
        trace["error"] = "No OpenAI API key configured"
        return None, trace

    kwargs = {
        "model": get_model(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    started = time.time()
    try:
        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        trace["raw_output"] = text
        trace["latency_ms"] = int((time.time() - started) * 1000)
        if response.usage:
            trace["tokens"] = response.usage.total_tokens
        if json_mode:
            return json.loads(text), trace
        return text.strip(), trace
    except Exception as exc:
        trace["error"] = f"{type(exc).__name__}: {exc}"
        trace["latency_ms"] = int((time.time() - started) * 1000)
        return None, trace
