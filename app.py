import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

## arxiv and wikipedia tools
api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search=DuckDuckGoSearchRun(name="search")

st.title("langchain-chat with search")
"""
in this example, we are using th streamlitcallbackhandle to display the thought and action of an agent in an interative streamlit app.
try more langchain streamlit agent example at [github.com/langchain-ai/streamlit-agent].

"""

## sidebar for settings
st.sidebar.title("settings")
api_key = st.sidebar.text_input("enter your groq API key:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant", "content":"hi, i am a chatbot who can search the web. how can i help you"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="what is the machine learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wiki]
   
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsig_errors=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)