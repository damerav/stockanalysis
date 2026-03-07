"""Reusable collapsible chatbot widget for every dashboard page.

Usage (add to the top of any page function):
    from src.dashboard.chatbot_widget import render_chatbot_widget
    render_chatbot_widget(page_key="spy", page_title="SPY Predictor")
"""
import logging
import streamlit as st

logger = logging.getLogger(__name__)

_DEEP_DIVE_LABEL = "Deep Dive (70b)"
_FAST_LABEL = "Fast (14b)"


def _get_bot():
    """Return the shared HelpChatbot instance, creating it if necessary."""
    if "help_chatbot" not in st.session_state:
        try:
            from src.llm.help_chatbot import HelpChatbot
            st.session_state.help_chatbot = HelpChatbot()
        except Exception as e:
            logger.error("Failed to initialise HelpChatbot: %s", e)
            st.session_state.help_chatbot = None
    return st.session_state.help_chatbot


def render_chatbot_widget(page_key: str, page_title: str):
    """Render a collapsible help chatbot expander for the given page.

    Args:
        page_key:   A short, unique identifier for the page (e.g. 'spy').
        page_title: The human-readable page name shown in the expander header
                    and passed to the chatbot as page context.
    """
    msg_key = f"chatbot_msgs_{page_key}"
    mode_key = f"chatbot_mode_{page_key}"

    if msg_key not in st.session_state:
        st.session_state[msg_key] = []
    if mode_key not in st.session_state:
        st.session_state[mode_key] = _FAST_LABEL

    with st.expander(f"Help  -  {page_title}", expanded=False):
        col_mode, col_clear = st.columns([3, 1])
        with col_mode:
            st.session_state[mode_key] = st.radio(
                "Response mode",
                [_FAST_LABEL, _DEEP_DIVE_LABEL],
                horizontal=True,
                key=f"mode_radio_{page_key}",
                label_visibility="collapsed",
            )
        with col_clear:
            if st.button("Clear", key=f"clear_{page_key}", use_container_width=True):
                st.session_state[msg_key] = []
                st.rerun()

        for msg in st.session_state[msg_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources", expanded=False):
                        for src in msg["sources"]:
                            st.caption(f"`{src}`")

        if prompt := st.chat_input(
            f"Ask about {page_title}...", key=f"chat_input_{page_key}"
        ):
            st.session_state[msg_key].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            bot = _get_bot()
            if bot is None:
                st.error("Help chatbot is unavailable. Check that Ollama is running.")
            else:
                use_deep = st.session_state[mode_key] == _DEEP_DIVE_LABEL
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        result = bot.ask(
                            prompt,
                            page_context=page_title,
                            use_deep_model=use_deep,
                        )
                    st.markdown(result["answer"])
                    if result.get("sources"):
                        with st.expander("Sources", expanded=False):
                            for src in result["sources"]:
                                st.caption(f"`{src}`")

                st.session_state[msg_key].append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                })
