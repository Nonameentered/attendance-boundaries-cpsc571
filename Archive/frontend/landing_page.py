"""
Landing page for streamlit app; includes access code prompt/box if necessary
"""

import streamlit as st
import streamlit_modal as modal
import streamlit.components.v1 as components
from streamlit2 import include_markdown

import referral_codes


def referral_code_entered():
    """Checks whether a referral code entered by the user is correct."""

    # TODO: update this to actually check against possible referral codes
    True


def check_consent():
    """Returns `True` if the user provides consent and also"""

    consent_text_placeholder = st.empty()
    access_code_entry_placeholder = st.empty()
    continue_button_placeholder = st.empty()
    read_more_button_placeholder = st.empty()

    # with consent_text_placeholder:
    #     include_markdown("couhes_message")

    st.session_state["continue_pressed"] = True

    with read_more_button_placeholder.container():
        with st.expander("Learn more"):
            st.write("\n\n\n")
            include_markdown("more_about_project")

    return [
        consent_text_placeholder,
        access_code_entry_placeholder,
        continue_button_placeholder,
        read_more_button_placeholder,
    ]
