from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings
from langchain_core.tools import tool

@tool
def retrieve_context(query: str):
    loader = ArxivLoader(query=query, load_max_docs=3)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=doc_splits, collection_name='energetic_docs', embedding=NVIDIAEmbeddings())
    retriever = vectorstore.as_retriever()
    results = retriever.invoke(query)
    return '\n'.join([doc.page_content for doc in results])
