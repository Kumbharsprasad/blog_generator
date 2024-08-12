import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
import os

load_dotenv()
st.subheader("Blog Generator")  

api_key = os.getenv('FIREWORKS_API_KEY')

with st.sidebar:
    st.title("Blog Generator")
    st.subheader("This app generates blog posts based on your topic [👉]")

# Initialize session state variables
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant"}
    ]

if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Only initialize ChatOpenAI and ConversationChain if API key is provided
if api_key:
    if st.session_state.conversation is None:
        llm = ChatOpenAI(
            model="accounts/fireworks/models/llama-v3p1-405b-instruct",
            openai_api_key=api_key,
            openai_api_base="https://api.fireworks.ai/inference/v1"
        )
        st.session_state.conversation = ConversationChain(
            memory=st.session_state.buffer_memory, 
            llm=llm
        )

    # Chat input for blog topic
    blog_topic = st.text_input("Enter the topic for your blog post:")

    if st.button("Generate Blog Post"):
        if blog_topic:
            prompt = blog_topic
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate and display blog post
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Generating blog post..."):
                        response = st.session_state.conversation.predict(input=prompt)
                        st.write(response)
                        message = {"role": "assistant", "content": response}
                        st.session_state.messages.append(message)
        else:
            st.warning("Please enter a topic for the blog post.")
else:
    st.warning("Please enter your Fireworks API Key in the sidebar to start generating blog posts.")
