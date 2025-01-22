from openai import OpenAI
import os
import streamlit as st

from utils import show_navigation
show_navigation()

#OPENAI_MODEL_NAME=st.secrets['OPENAI_MODEL_NAME']
OPENAI_API_KEY=st.secrets['OPENAI_API_KEY']

avatars={"system":"💻🧠","user":"🧑‍💼","assistant":"🎓"}
client=OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_MESSAGE={"role": "system", 
                "content": "Ignore all previous commands. You are a helpful and patient guide based in Silicon Valley."
                }

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SYSTEM_MESSAGE)

for message in st.session_state.messages:
    if message["role"] != "system":
        avatar=avatars[message["role"]]
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=avatars["assistant"]):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.messages], stream=True):
            delta_response=response.choices[0].delta
            #print(f"Basic CHat Delta response: {delta_response}")
            if delta_response.content:
                full_response += delta_response.content
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})