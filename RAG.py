from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.tools import tool


@tool
def retrieve_context(query: str) -> dict:
    """
    Retrieve matching documents from arXiv academic repository
    :param query: user input for retrieving information
    :return: dict that contains source content and its metadata
    """
    loader = ArxivLoader(query=query, load_max_docs=3)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(docs)
    # vectorstore = Chroma.from_documents(documents=doc_splits, collection_name='energetic_docs',
    #                                     embedding=NVIDIAEmbeddings())
    vectorstore = Chroma.from_documents(documents=doc_splits, collection_name='energetic_docs',
                                        embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    retrieved_chunks = retriever.invoke(query)
    results = {}
    for i, chunk in enumerate(retrieved_chunks):
        results[i] = {'content': chunk.page_content, 'metadata': chunk.metadata}
    return results



