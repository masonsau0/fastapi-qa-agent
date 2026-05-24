"""Streamlit UI.

Talks to the FastAPI server. Shows the question, the answer, AND the tool
trace — because the trace is half the demo. Recruiters watching the demo
should see *that the agent called tools*, not just see an answer.

Run with:
    streamlit run ui/app.py

Assumes the server is running on http://localhost:8000.
"""

from __future__ import annotations

import os
import time

import httpx
import streamlit as st

API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8000")


st.set_page_config(page_title="FastAPI Codebase QA Agent", page_icon="🔎", layout="wide")
st.title("FastAPI Codebase QA Agent")
st.caption(
    "Ask a question about the FastAPI codebase. The agent will search the code, the docs, and git history to answer."
)


# Sidebar with a few example questions to make demos easier.
with st.sidebar:
    st.header("Examples")
    examples = [
        "How does FastAPI handle dependency injection?",
        "What changed in routing.py recently and why?",
        "How do I add a custom exception handler?",
        "Where is the OpenAPI schema generated?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["prefill"] = ex
    st.divider()
    st.caption("API: " + API_URL)


# Health check up top.
try:
    health = httpx.get(f"{API_URL}/health", timeout=2.0)
    if health.status_code == 200:
        st.success("✓ Server reachable")
    else:
        st.error(f"Server returned {health.status_code}")
except httpx.HTTPError:
    st.error(f"Can't reach server at {API_URL}. Is uvicorn running?")
    st.stop()


# Question input.
default_q = st.session_state.pop("prefill", "")
question = st.text_area("Question", value=default_q, height=80, placeholder="How does FastAPI...")

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Ask", type="primary", use_container_width=True)


if submit and question.strip():
    with st.spinner("Agent is working..."):
        t0 = time.time()
        try:
            resp = httpx.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=120.0,
            )
            elapsed = time.time() - t0
        except httpx.HTTPError as e:
            st.error(f"Request failed: {e}")
            st.stop()

    if resp.status_code != 200:
        st.error(f"Server returned {resp.status_code}: {resp.text}")
        st.stop()

    data = resp.json()

    # Render answer.
    st.subheader("Answer")
    st.markdown(data["answer"])

    # Metadata row.
    a, b, c = st.columns(3)
    a.metric("Iterations", data["iterations"])
    b.metric("Tool calls", len(data["tool_trace"]))
    c.metric("Elapsed", f"{elapsed:.1f}s")

    # Tool trace.
    if data["tool_trace"]:
        with st.expander("Tool trace", expanded=True):
            for i, name in enumerate(data["tool_trace"], start=1):
                st.write(f"{i}. `{name}`")
