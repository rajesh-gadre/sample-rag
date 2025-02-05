# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import numpy as np

from openai import OpenAI
import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec, PodSpec

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from utils import show_navigation
show_navigation()

st.sidebar.write(f"Secrets: {st.secrets}")

PINECONE_API_KEY=st.secrets['PINECONE_API_KEY']
PINECONE_INDEX_NAME=st.secrets['PINECONE_INDEX_NAME']
PINECONE_NAMESPACE=st.secrets['PINECONE_NAMESPACE']

OPENAI_API_KEY=st.secrets['OPENAI_API_KEY']
OPENAI_MODEL_NAME=st.secrets['OPENAI_MODEL_NAME']
OPENAI_EMBEDDING_MODEL_NAME=st.secrets['OPENAI_EMBEDDING_MODEL_NAME']

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]='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'
os.environ["LANGSMITH_API_KEY"]='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'
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT']="sleep-research"


client=OpenAI(api_key=OPENAI_API_KEY)
#myModel = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)

class MyAnswer(BaseModel):
    answer: str
    referredFigures: str
    willAssist: bool


def augmented_content(inp):
    # Create the embedding using OpenAI keys
    # Do similarity search using Pinecone
    # Return the top 5 results
    embedding=client.embeddings.create(model=OPENAI_EMBEDDING_MODEL_NAME, input=inp).data[0].embedding
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    results=index.query(vector=embedding,top_k=3,namespace=PINECONE_NAMESPACE,include_metadata=True)
    
    LOGGER.info(f"query: {inp}")
    max_length_to_log = 100
    for idx, r in enumerate(results['matches']):
        LOGGER.info(f"{idx}: hash: {r['metadata']['hash']}")
        if len(r['metadata']['text']) > max_length_to_log:
            LOGGER.info(f"{idx}: text: {r['metadata']['text'][:max_length_to_log]}" + "...")    
        else:
            LOGGER.info(f"{idx}: hash: {r['metadata']['text']}")
        LOGGER.info(f"{idx}: score: {r['score']}")

    with st.sidebar.expander("Retrieved records"):
        st.write(f"Results: {results}")
    rr=[ r['metadata']['text'] for r in results['matches']]
    #print(f"RR: {rr}")
    #st.write(f"RR: {rr}")
    return rr


# SYSTEM_MESSAGE={"role": "system", 
#                 "content": f"""Ignore all previous commands. 
#                 You are a helpful and patient guide based in Silicon Valley. 
#                 Please answer the questions based only on the information provided.
#                 Else only say that you do not know. Do not try to answer the question based on your training data.
#                 """
#                 }

# SYSTEM_MESSAGE={"role": "system", 
#                 "content": f"""Ignore all previous commands. 
#                 Please answer the questions based only on the information provided.
#                 Else only say that you do not know. Do not try to answer the question based on your training data.
#                 In addition to answering the question, return only the list of referred figures and only true or false if the figures will assist your response.
#                 """
#                 }

# SYSTEM_MESSAGE={"role": "system", 
#                 "content": f"""Ignore all previous commands. 
#                 Please answer the questions based only on the information provided.
#                 Else only say that you do not know. Do not try to answer the question based on your training data.
#                 In addition to answering the question, return only the list of referred figures and only true or false if the figures will assist your response. Note that figure references start with word 'Figure'.
#                 """
#                 }


# system_message_content="""Ignore all previous commands.
#                 Please answer the questions based only on the information provided.
#                 Else only say that you do not know. Do not try to answer the question based on your training data.
#                 In addition to answering the question,
#                 Set referredFigures to the list of figures referred in the provided information only if they will assist understanding your answer, else set it to null. Note that figure references start with word 'Figure'.
#                 Set willAssist to true only if the figures will assist understanding your answer, else set it to false.
#                 """

# system_message_content="""
#         Ignore all previous commands.
#         Please answer the questions based only on the information provided.
#         Else only say that you do not know. Do not try to answer the question based on your training data.
#         In addition to answering the question,
#         Set willAssist to true only if any of the figures referred to in the provided information will help further understand your answer, else set it to false.
#         Set referredFigures to the list of only those figures that will help further understand your answer, else set it to null. Note that figure references start with word 'Figure'.
#         """

system_message_content="""
        Ignore all previous commands.
        Please answer the questions based only on the information provided.
        Else only say that you do not know. Do not try to answer the question based on your training data.
        In addition to answering the question,
        Set willAssist to true only if any of the figures referred to in the provided information will help further understand your answer, else set it to false.
        Set referredFigures to the list of only those figures that will help further understand your answer, else set it to null. Note that figure references start with word 'Figure'.
        """


SYSTEM_MESSAGE={"role": "system", 
                "content": f"""{system_message_content}
                """
                }

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SYSTEM_MESSAGE)

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    retreived_content = augmented_content(prompt)
    prompt_guidance=f"""
Please guide the user with the following information:
{retreived_content}
The user's question was: {prompt}
    """
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # messageList=[{"role": m["role"], "content": m["content"]}
        #               for m in st.session_state.messages]
        # messageList.append({"role": "user", "content": prompt_guidance})
        
        # sm=SystemMessage(content="""
        # Ignore all previous commands.
        # Please answer the questions based only on the information provided.
        # Else only say that you do not know. Do not try to answer the question based on your training data.
        # In addition to answering the question,
        # Set referredFigures to the list of figures referred in the provided information only if they will help user understand your answer, else set it to null. Note that figure references start with word 'Figure'.
        # Set willAssist to true only if the figures will help user understand your answer, else set it to false.
        # """)
        
        sm=SystemMessage(content=system_message_content)

        hm=HumanMessage(content=prompt_guidance)
        messages=[sm,hm]
        myModel = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
        llm_response = myModel.with_structured_output(MyAnswer).invoke(messages)
        full_response = llm_response.answer
        if llm_response.willAssist:
            full_response += f"""\n\nHere are the figures to further illustrate the answer: {llm_response.referredFigures}"""
        message_placeholder.markdown(full_response)

        
        # for response in client.chat.completions.create(
        #     model=OPENAI_MODEL_NAME,
        #     messages=messageList, stream=True):
        #     delta_response=response.choices[0].delta
        #     #print(f"RAG Delta response: {delta_response}")
        #     if delta_response.content:
        #         full_response += delta_response.content
        #     message_placeholder.markdown(full_response + "▌")
        # message_placeholder.markdown(full_response)

    with st.sidebar.expander("Retreival context provided to LLM"):
        st.write(f"{retreived_content}")
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.sidebar.write(f"messages: {st.session_state.messages}")
