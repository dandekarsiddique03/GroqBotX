import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
import chardet
from docx import Document
from reportlab.pdfgen import canvas

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

st.title('GroqBot X – "Chat with Your Documents, Smarter & Faster!"')

# Storage directory
DOC_DIR = "us_census"
if not os.path.exists(DOC_DIR):
    os.makedirs(DOC_DIR)

# File uploader (accepts PDF, DOCX, TXT)
uploaded_files = st.file_uploader(
    "Upload your documents", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

# Function to detect text file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']

# Function to convert TXT to PDF
def convert_txt_to_pdf(txt_path, pdf_path):
    encoding = detect_encoding(txt_path)
    with open(txt_path, "r", encoding=encoding) as f:
        text = f.read()
    
    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica", 12)
    
    y_position = 800
    for line in text.split("\n"):
        if y_position < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = 800
        c.drawString(50, y_position, line)
        y_position -= 20
    c.save()

# Function to convert DOCX to PDF
def convert_docx_to_pdf(docx_path, pdf_path):
    doc = Document(docx_path)
    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica", 12)
    
    y_position = 800
    for para in doc.paragraphs:
        if y_position < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = 800
        c.drawString(50, y_position, para.text)
        y_position -= 20
    c.save()

# Save uploaded files and convert TXT/DOCX to PDF
def process_files(uploaded_files):
    converted_files = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DOC_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            
            if uploaded_file.name.endswith(".txt"):
                pdf_path = file_path.replace(".txt", ".pdf")
                convert_txt_to_pdf(file_path, pdf_path)
                converted_files.append(pdf_path)

            elif uploaded_file.name.endswith(".docx"):
                pdf_path = file_path.replace(".docx", ".pdf")
                convert_docx_to_pdf(file_path, pdf_path)
                converted_files.append(pdf_path)
    
    # Start document processing after conversion
    vector_embedding()
    st.success(f"✅ {len(uploaded_files)} files processed successfully!")

# Process all PDFs
def vector_embedding():
    all_documents = []
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    pdf_files = [os.path.join(DOC_DIR, file) for file in os.listdir(DOC_DIR) if file.endswith(".pdf")]
    if not pdf_files:
        st.warning("⚠️ No PDFs found in the 'us_census' folder. Please upload files first.")
        return

    for file_path in pdf_files:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        all_documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(all_documents)
    
    st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)
    st.success("✅ Vector Store DB is Ready!")

# Single button for both conversion and processing
if st.button("Process Documents"):
    if uploaded_files:
        process_files(uploaded_files)
    else:
        st.warning("⚠️ Please upload at least one document.")

# Query input field
prompt1 = st.text_input("Ask a question from the documents")

if prompt1:
    if 'vectors' in st.session_state:
        document_chain = create_stuff_documents_chain(ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192"), ChatPromptTemplate.from_template("""
            Answer the questions based on the provided context only.
            <context>
            {context}
            </context>
            Question: {input}
        """))
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': prompt1})

        st.write("📌 **Response:**", response['answer'])
        
        with st.expander("Document Similarity"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("-----------------------------")
    else:
        st.warning("⚠️ Please process the documents first!")

# Button to clear all files
if st.sidebar.button("Clear All Files"):
    for file in os.listdir(DOC_DIR):
        os.remove(os.path.join(DOC_DIR, file))
    if "faiss_index" in os.listdir():
        os.remove("faiss_index")
    st.success("✅ All uploaded files and stored data have been cleared!")
