import streamlit as st
import requests


st.set_page_config(page_title="📄 RAG Q&A Chatbot", layout="wide")
st.title("📄 Retrieval-Augmented Q&A with Backend PDFs + Chat History")


##Backend API URL
BACKEND_URL = "http://127.0.0.1:10000/ask"

st.sidebar.header("⚙️ Configuration")
st.sidebar.write("- Backend PDFs auto-load from FastAPI\n- Ask questions freely!")

session_id = st.text_input("🆔 Enter Session ID", value="default_session")
user_question = st.chat_input("Ask your question here...")


##store history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if user_question:
    with st.spinner("🤔 Thinking..."):
        response = requests.post(
            BACKEND_URL,
            json={"session_id": session_id, "question": user_question}
        )

    data = response.json()
    answer = data.get("answer", data.get("error", "No response."))


    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("assistant", answer))

# Display messages
for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)


# View history
if st.session_state.chat_history:
    with st.expander("📖 View Full Chat History"):
        for role, msg in st.session_state.chat_history:
            st.markdown(f"**{role.title()}:** {msg}")
