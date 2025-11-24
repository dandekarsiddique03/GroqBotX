import streamlit as st
import os
import time
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

# ------------------------------------------------------
#  STREAMLIT CLOUD SECRET LOADING (IMPORTANT FIX)
# ------------------------------------------------------
groq_api_key = st.secrets["GROQ_API_KEY"]

st.title('GroqBot X – "Chat with Your Documents, Smarter & Faster!"')

# Storage directory
DOC_DIR = "us_census"
if not os.path.exists(DOC_DIR):
    os.makedirs(DOC_DIR)

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# ---------- ENCODING ----------
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw = f.read()
    return chardet.detect(raw)['encoding']

# ---------- TXT → PDF ----------
def convert_txt_to_pdf(txt_path, pdf_path):
    enc = detect_encoding(txt_path)
    with open(txt_path, "r", encoding=enc) as f:
        text = f.read()

    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica", 12)
    y = 800

    for line in text.split("\n"):
        if y < 50:
            c.showPage(); c.setFont("Helvetica", 12); y = 800
        c.drawString(50, y, line)
        y -= 20
    c.save()

# ---------- DOCX → PDF ----------
def convert_docx_to_pdf(docx_path, pdf_path):
    doc = Document(docx_path)
    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica", 12)
    y = 800

    for para in doc.paragraphs:
        if y < 50:
            c.showPage(); c.setFont("Helvetica", 12); y = 800
        c.drawString(50, y, para.text)
        y -= 20
    c.save()

# ---------- PROCESS FILES ----------
def process_files(uploaded_files):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DOC_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            if file_path.endswith(".txt"):
                convert_txt_to_pdf(file_path, file_path.replace(".txt", ".pdf"))

            elif file_path.endswith(".docx"):
                convert_docx_to_pdf(file_path, file_path.replace(".docx", ".pdf"))

    vector_embedding()
    st.success(f"✅ {len(uploaded_files)} files processed successfully!")

# ---------- VECTOR EMBEDDING ----------
def vector_embedding():
    pdf_files = [
        os.path.join(DOC_DIR, f)
        for f in os.listdir(DOC_DIR)
        if f.endswith(".pdf")
    ]

    if not pdf_files:
        st.warning("⚠️ No PDF files found. Upload a file first.")
        return

    all_docs = []
    for pdf in pdf_files:
        loader = PyMuPDFLoader(pdf)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)
    st.success("✅ Vector Store DB is Ready!")

# ---------- PROCESS BUTTON ----------
if st.button("Process Documents"):
    if uploaded_files:
        process_files(uploaded_files)
    else:
        st.warning("⚠️ Upload at least one document.")

# ---------- USER QUERY ----------
prompt1 = st.text_input("Ask a question from the documents")

if prompt1:
    if "vectors" in st.session_state:

        llm = ChatGroq(
            api_key=groq_api_key,
            model_name="Llama3-8b-8192"
        )

        prompt = ChatPromptTemplate.from_template("""
        Answer based ONLY on the provided context.
        If the answer is not found, say:
        "I cannot find this in the documents."

        <context>
        {context}
        </context>

        Question: {input}
        """)

        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({"input": prompt1})

        st.write("### 📌 Response")
        st.write(response["answer"])

        with st.expander("Document Context Used"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("---")

    else:
        st.warning("⚠️ Process the documents first!")

# ---------- CLEAR ALL ----------
if st.sidebar.button("Clear All Files"):
    for f in os.listdir(DOC_DIR):
        os.remove(os.path.join(DOC_DIR, f))
    st.success("✅ All files cleared successfully!")
