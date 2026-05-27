"""Browser-side auth persistence (sessionStorage survives page refresh but dies on tab close)."""

from __future__ import annotations

import json
from typing import Any, MutableMapping

import streamlit as st
import streamlit.components.v1 as components
from streamlit_javascript import st_javascript

from frontend.auth import (
    AUTH_COOKIE_NAME,
    is_session_valid,
    login_session,
    logout_session,
    validate_persistent_token,
)


def get_cookie_manager() -> Any:
    """Legacy interface support. We no longer use CookieManager."""
    return None


def set_auth_cookie(token: str) -> None:
    """SessionStorage: cleared when the tab is closed, survives refresh."""
    token_js = json.dumps(token)
    st_javascript(f"sessionStorage.setItem('{AUTH_COOKIE_NAME}', {token_js});")


def clear_auth_cookie(cookies: Any = None) -> None:
    st_javascript(f"sessionStorage.removeItem('{AUTH_COOKIE_NAME}');")


def bootstrap_auth_from_browser(
    session_state: MutableMapping[str, Any],
    cookies: Any = None,
) -> None:
    """
    Restore login after refresh from sessionStorage.
    """
    if is_session_valid(session_state):
        return

    # 1. First execution returns 0 while JS evaluates
    token = st_javascript(f"sessionStorage.getItem('{AUTH_COOKIE_NAME}')")
    
    # 2. If it's exactly 0, we must stop and wait for the rerun
    if token == 0:
        st.stop()

    # 3. Once evaluated, if token exists, log the user in
    if isinstance(token, str) and token and token != "null":
        user = validate_persistent_token(token)
        if user:
            login_session(session_state, user, existing_token=token)
        else:
            clear_auth_cookie()


def persist_login(session_state: MutableMapping[str, Any], user: dict[str, Any]) -> None:
    """Save login to Streamlit state and browser sessionStorage."""
    token = login_session(session_state, user)
    set_auth_cookie(token)


def persist_logout(session_state: MutableMapping[str, Any], cookies: Any = None) -> None:
    """Clear login everywhere."""
    logout_session(session_state, revoke_token=True)
    clear_auth_cookie()
