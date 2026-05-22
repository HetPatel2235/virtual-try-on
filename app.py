"""Backward-compatible entrypoint for Streamlit frontend."""

import runpy
import streamlit as st

# -----------------------------
# Run Frontend
# -----------------------------

runpy.run_module("frontend.app", run_name="__main__")