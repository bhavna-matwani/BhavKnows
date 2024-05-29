import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.prompts import load_prompt
from streamlit import session_state as ss
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import uuid
import json
import datetime
import streamlit.components.v1 as components

footer_html = """
<div style="text-align: left; color: white;"><span>Connect: </span><a href="https://www.linkedin.com/in/bhavna-matwani/" target="_blank">LinkedIn</a><span style="padding: 0 10px;">|</span><a href="https://github.com/bhavna-matwani" target="_blank">GitHub</a></div>
"""

# Function to check if a string is a valid JSON
def is_valid_json(data):
    try:
        json.loads(data)
        return True
    except json.JSONDecodeError:
        return False

# Function to initialize connection to Firebase Firestore
@st.cache_resource
def init_connection():
    key_dict = json.loads(st.secrets["textkey"])
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred)
    return firestore.client()

# Attempt to connect to Firebase Firestore
try:
    db = init_connection()
except Exception as e:
    st.write("Failed to connect to Firebase:", e)

# Access Firebase Firestore collection
if 'db' in locals():
    conversations_collection = db.collection('conversations')
else:
    st.write("Unable to access conversations collection. Firebase connection not established.")

# Retrieve OpenAI API key
if "OPENAI_API_KEY" in os.environ:
    openai_api_key = os.getenv("OPENAI_API_KEY")
else:
    openai_api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit app title and disclaimer
st.title("BhavKnow - Resume Bot for Bhavna")
st.html(footer_html)
with st.expander("⚠️Disclaimer"):
    st.write("""This bot is a LLM trained on GPT-3.5-turbo model to answer questions about Bhavna's professional background and qualifications. Your responses are recorded in a database for quality assurance and improvement purposes. Please be respectful and avoid asking personal or inappropriate questions.""")

# Define file paths and load initial settings
path = os.path.dirname(__file__)
prompt_template = path+"/templates/template.json"
prompt = load_prompt(prompt_template)
faiss_index = path+"/faiss_index"
data_source = path+"/data/about_bhavna.csv"
pdf_source = path+"/data/Bhavna_Matwani_Resume.pdf"

# Function to store conversation in Firebase
def store_conversation(conversation_id, user_message, bot_message, answered):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "user_message": user_message,
        "bot_message": bot_message,
        "answered": answered
    }
    conversations_collection.add(data)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Load FAISS index or create a new one if it doesn't exist
if os.path.exists(faiss_index):
    vectors = FAISS.load_local(faiss_index, embeddings, allow_dangerous_deserialization=True)
else:
    # Load data from PDF and CSV sources
    pdf_loader = PyPDFLoader(pdf_source)
    pdf_data = pdf_loader.load_and_split()
    csv_loader = CSVLoader(file_path=data_source, encoding="utf-8")
    csv_data = csv_loader.load()
    data = pdf_data + csv_data

    # Create embeddings for the documents and save the index
    vectors = FAISS.from_documents(data, embeddings)
    vectors.save_local("faiss_index")

# Initialize conversational retrieval chain
retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 6, "include_metadata": True, "score_threshold": 0.6})
chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.5, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key), 
                                              retriever=retriever, return_source_documents=True, verbose=True, chain_type="stuff",
                                              max_tokens_limit=4097, combine_docs_chain_kwargs={"prompt": prompt})

# Function to handle conversational chat
def conversational_chat(query):
    with st.spinner("Thinking..."):
        result = chain({"system": 
        "You are a Resume Bot, a comprehensive, interactive resource for exploring Bhavns's background, skills, and expertise. Be polite and provide answers based on the provided context only. You can make up projects with the skills and projects I have if the question requests a skill set related to Machine Learning, Database management or Computer sciences.", 
                        "question": query, 
                        "chat_history": st.session_state['history']})
    
    # Check if the result is a valid JSON
    if is_valid_json(result["answer"]):              
        data = json.loads(result["answer"])
    else:
        data = json.loads('{"answered":"false", "response":"Hmm... Something is not right. I\'m experiencing technical difficulties. Try asking your question again or ask another question about  Bhavna\'s professional background and qualifications. Thank you for your understanding.", "questions":["What is  Bhavna\'s professional experience?","What projects has  Bhavna worked on?","What are  Bhavna\'s career goals?"]}')
    
    answered = data.get("answered")
    response = data.get("response")
    questions = data.get("questions")

    full_response="--"

    # Append user query and bot response to chat history
    st.session_state['history'].append((query, response))
    
    # Process the response based on the answer status
    if ('I am tuned to only answer questions' in response) or (response == ""):
        full_response = """Unfortunately, I can't answer this question. My capabilities are limited to providing information about  Bhavna's professional background and qualifications. If you have other inquiries, I recommend reaching out to  Bhavna on [LinkedIn](https://www.linkedin.com/in/bhavna-matwani/). I can answer questions like: \n - What is  Bhavna's educational background? \n - Can you list  Bhavna's professional experience? \n - What skills does  Bhavna possess? \n"""
        store_conversation(st.session_state["uuid"], query, full_response, answered)
    else: 
        markdown_list = ""
        for item in questions:
            markdown_list += f"- {item}\n"
        full_response = response + "\n\n What else would you like to know about  Bhavna? You can ask me: \n" + markdown_list
        store_conversation(st.session_state["uuid"], query, full_response, answered)
    return(full_response)

# Initialize session variables if not already present
if "uuid" not in st.session_state:
    st.session_state["uuid"] = str(uuid.uuid4())

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        welcome_message = """
            Welcome! I'm Resume Bot, a virtual assistant dedicated to showcasing Bhavna Matwani's remarkable qualifications and career achievements. Here to provide an in-depth view of her skills, experiences, and academic background, I can offer insights into various facets of her professional journey.

            - Her Master's in Computer Science from NYU
            - Her hands-on experience developing AI solutions like SmartSMS generator, Enhancing LLM responses using RAG and FrameForesight
            - Her proven track record in roles at Vimbly Group, Mastercard and IISc Bangalore
            - Her proficiency in programming languages, software development, ML frameworks, and cloud platforms
            - Her passion for leveraging transformative technologies for positive societal impact

            Feel free to ask about any details of her education, work experience, technical skills, or her contributions to various projects and roles. What would you like to know more about?
            """
        message_placeholder.markdown(welcome_message)

if 'history' not in st.session_state:
    st.session_state['history'] = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process user input and display bot response
if prompt := st.chat_input("Ask me about Bhavna"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        user_input=prompt
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        full_response = conversational_chat(user_input)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
