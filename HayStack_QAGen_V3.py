import streamlit as st
import os
import docx
import haystack
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever,TfidfRetriever,PromptNode, PromptTemplate
from haystack.nodes import FARMReader,TransformersReader
from haystack.pipelines import ExtractiveQAPipeline,DocumentSearchPipeline,GenerativeQAPipeline,Pipeline
from haystack.utils import print_answers
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack import Document
from haystack.utils import print_answers
import pdfplumber
from haystack.nodes import OpenAIAnswerGenerator
from haystack.nodes.prompt import PromptTemplate
import re

document_store = InMemoryDocumentStore()
documents=[]
def add_document(document_store, file):
    if file.type == 'text/plain':
        text = file.getvalue().decode("utf-8")
        # document_store.write_documents(dicts)
        # st.write(file.name)
        # st.write(text)
    elif file.type == 'application/pdf':
        with pdfplumber.open(file) as pdf:
            text = "\n\n".join([page.extract_text() for page in pdf.pages])
            # document_store.write_documents([{"text": text, "meta": {"name": file.name}}])
            # st.write(text)
    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc = docx.Document(file)
        text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
        # document_store.write_documents([{"text": text, "meta": {"name": file.name}}])
        # st.write(text)
    else:
        st.warning(f"{file.name} is not a supported file format")
    dicts = {
            'content': text,
            'meta': {'name': file.name}
            }  
    documents.append(dicts) 

    
# create Streamlit app
st.set_page_config(page_title='Contextualized Search for Document Archive',layout="wide")#Update V2
st.write("""
    <style>
        footer {visibility: hidden;}
        body {
            font-family: Arial, sans-serif;
        }
    </style>
""", unsafe_allow_html=True)#Update V2
# st.title("Contextualized Search for Document Archive")#Update V2
st.write(f"<h1 style='font-size: 36px; color: #00555e; font-family: Arial;text-align: center;'>Contextualized Search for Document Archive using OpenAI</h1>", unsafe_allow_html=True)
API_KEY = st.secrets['OPENAI_API_KEY']
# create file uploader
uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

# loop through uploaded files and add them to document store
if uploaded_files:
    for file in uploaded_files:
        add_document(document_store, file)
    document_store.write_documents(documents)
# display number of documents in document store
    st.write(f"<p style='font-size: 16px; color: #00555e;font-family: Arial;'>Number of documents uploaded to document store: {document_store.get_document_count()}</p>", unsafe_allow_html=True)

if (document_store.get_document_count()!=0):
    # question = st.text_input('Ask a question') #Update V2
    st.write(f"<p style='font-size: 16px; color: red;font-family: Arial;'>Ask a question:</p>",unsafe_allow_html=True)
    question = st.text_input(label='Ask a question:',label_visibility="collapsed")
    retriever = TfidfRetriever(document_store=document_store)
     # QA pipeline using prompt node 
    if question != '':  
        my_template =  PromptTemplate(
                             prompt="""Given the context please answer the question.
                             If the question cannot be answered from the context, reply with 'No relevant information present in attached documents'.
                             Context: {join(documents)}; 
                             Question: {query}; 
                             Answer:""")

        my_template1=PromptTemplate(prompt="Create a concise and informative answer for a given question,based from context on the given documents"
            "Provide a clear and detailed response from the relevant information presented in the context. "
            "If the question cannot be answered from the context, reply with 'No relevant information present in attached documents\n"
            "Context: {join(documents, delimiter=new_line, pattern=new_line+'Document[$idx]: $content', str_replace={new_line: ' ', '[': '(', ']': ')'})} \n Question: {query}; Answer: ")  
        
        candidate_documents = retriever.retrieve(query=question,top_k=2)

        prompt_node = PromptNode("gpt-3.5-turbo-16k", api_key=API_KEY)
        output=prompt_node.prompt(prompt_template=my_template, query=question, documents=candidate_documents)
        output1=prompt_node.prompt(prompt_template=my_template, query=question, documents=candidate_documents)

        if output:
            answer = output[0]
            if(answer=='No relevant information present in attached documents.'):
               st.write(f"<p style='font-size: 16px; color: #00008B;font-family: Arial;'>Result generated using Default prompt:</p>",unsafe_allow_html=True)
               st.write(f"<p style='font-size: 16px; color: #00555e;font-family: Arial;'>No document found with relevant context.</p>",unsafe_allow_html=True)
            else:
                st.write(f"<p style='font-size: 16px; color: #00008B;font-family: Arial;'>Result generated using Default prompt:</p>",unsafe_allow_html=True)
                st.write(f"<p style='font-size: 16px; color: #00555e;font-family: Arial;'>Source document: {candidate_documents[0].meta['name']}</p>",unsafe_allow_html=True)
                st.write(f"<p style='font-size: 16px; color: #00555e;font-family: Arial;'>Answer: {answer}</p>",unsafe_allow_html=True)
        else:
            st.write('Please try with another question')

        if output1:
            answer = output1[0]
            if(answer=='No relevant information present in attached documents.'):
               st.write(f"<p style='font-size: 16px; color: #00008B;font-family: Arial;'>Result generated using Customised prompt:</p>",unsafe_allow_html=True)
               st.write(f"<p style='font-size: 16px; color: #00555e;font-family: Arial;'>No document found with relevant context.</p>",unsafe_allow_html=True)
            else:
                st.write(f"<p style='font-size: 16px; color: #00008B;font-family: Arial;'>Result generated using Customised prompt:</p>",unsafe_allow_html=True)
                st.write(f"<p style='font-size: 16px; color: #00555e;font-family: Arial;'>Source document: {candidate_documents[0].meta['name']}</p>",unsafe_allow_html=True)
                st.write(f"<p style='font-size: 16px; color: #00555e;font-family: Arial;'>Answer: {answer}</p>",unsafe_allow_html=True)
        else:
            st.write('Please try with another question') 
    






        