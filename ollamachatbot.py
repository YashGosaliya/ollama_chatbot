import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

load_dotenv()

##Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot with Ollama"

##Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant. PLease respond to the question asked"),
        ("user","Question:{question}")
    ]
)

##streamlit framework
st.title("Langchain Demo with LLAMA3.2")
##input_text=st.text_input("What question you have in mind?")

##Ollama Llama2 model
def generate_response(question,engine,temperature,max_tokens):
    llm = Ollama(model="llama3.2")
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer


##Select the model
engine=st.sidebar.selectbox("Select AI Model", ["llama3.2"])

##Adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

##Main Interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, engine="llama3.2", temperature=0.5, max_tokens=100)
    st.write(response)
else:
    st.write("Please provide the user input")