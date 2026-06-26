import sys
from datetime import datetime
from pathlib import Path

# Add project root (parent of src/) to path so "from src...." works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st


st.set_page_config(
    page_title="Alzheimer's Support Chatbot",
    page_icon="🧠",
    layout="centered",
)

# Render the header IMMEDIATELY so the page is never fully blank,
# even while the model is still loading.
st.title("🧠 Alzheimer's Support Chatbot")
st.caption(
    "A supportive chatbot for general Alzheimer's information and caregiver guidance. "
    "This app does not diagnose disease or replace professional medical care."
)

# Prominent disclaimer banner, always visible at the top.
st.warning(
    "⚠️ **This is not medical advice.** This chatbot gives general information only. "
    "It cannot diagnose, prescribe, or replace a qualified healthcare professional. "
    "For emergencies, severe symptoms, or medication changes, contact a doctor or emergency services."
)


# Starter questions shown as clickable buttons for new users.
EXAMPLE_QUESTIONS = [
    "What are the early signs of Alzheimer's?",
    "How can I support a loved one with memory loss?",
    "What helps with daily routines for a person with dementia?",
    "How do I handle confusion or agitation calmly?",
]


@st.cache_resource
def load_chatbot():
    # Import here so any import error surfaces inside the try/except below.
    from src.inference import AlzheimerChatbot
    return AlzheimerChatbot()


def build_transcript() -> str:
    """Turns the current chat into plain text for download."""
    lines = ["Alzheimer's Support Chatbot - Conversation"]
    lines.append("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("(General information only - not medical advice.)")
    lines.append("=" * 50)
    lines.append("")

    for message in st.session_state.get("messages", []):
        speaker = "You" if message["role"] == "user" else "Chatbot"
        lines.append(f"{speaker}: {message['content']}")
        lines.append("")

    return "\n".join(lines)


def reset_messages():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello. I can help with general Alzheimer's information, memory-care questions, "
                "and caregiver support. How can I help you today?"
            ),
        }
    ]


def main():
    with st.sidebar:
        st.header("About")
        st.write(
            "This chatbot is trained on Alzheimer's-related conversation data. "
            "It is intended for general education and caregiver support."
        )

        st.warning(
            "For emergencies, severe symptoms, medication changes, or diagnosis, "
            "please contact a qualified healthcare professional."
        )

        st.divider()

        # Download the conversation as a text file.
        st.download_button(
            label="⬇️ Download conversation",
            data=build_transcript(),
            file_name="alzheimer_chat.txt",
            mime="text/plain",
            use_container_width=True,
        )

        if st.button("🗑️ Clear chat", use_container_width=True):
            reset_messages()
            st.session_state.pop("pending_question", None)
            st.rerun()

    # Load the model with visible feedback and error reporting.
    try:
        with st.spinner("Loading the chatbot model (first time can take a while)..."):
            chatbot = load_chatbot()
    except Exception as error:
        st.error("The chatbot model failed to load. Details below:")
        st.exception(error)
        st.stop()

    if "messages" not in st.session_state:
        reset_messages()

    # Show example-question buttons only at the very start of a conversation
    # (i.e. before the user has asked anything).
    user_has_asked = any(m["role"] == "user" for m in st.session_state.messages)
    if not user_has_asked:
        st.markdown("**Try one of these to get started:**")
        cols = st.columns(2)
        for index, question in enumerate(EXAMPLE_QUESTIONS):
            column = cols[index % 2]
            if column.button(question, key=f"example_{index}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

    # Render the existing conversation.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # A question can come either from the chat box or from an example button.
    typed_input = st.chat_input("Ask a question about Alzheimer's or caregiving...")
    user_input = typed_input or st.session_state.pop("pending_question", None)

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chatbot.chat(user_input)
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # If the question came from an example button, rerun so the buttons
        # disappear and the layout updates cleanly.
        if not typed_input:
            st.rerun()


if __name__ == "__main__":
    main()
