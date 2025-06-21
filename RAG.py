from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from openai import embeddings
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool
from auxiliary import ChemBERT_ChEMBL_pretrained_embeddings


@tool
def retrieve_context(query: str) -> list:
    """
    Retrieve matching documents from arXiv academic repository
    :param query: user input for retrieving information
    :return: a list that contains dictionaries, each with a relevant content chunk (key 'Content'),
     title of the paper (key 'Title') and authors of the paper (key 'Authors').
    """
    loader = ArxivLoader(query=query, load_max_docs=10)
    # text_splitter = TokenTextSplitter(chunk_size=800, chunk_overlap=80, encoding_name="cl100k_base")
    text_splitter = SemanticChunker(embeddings=ChemBERT_ChEMBL_pretrained_embeddings())
    doc_splits = loader.load_and_split(text_splitter)
    # vectorstore = Chroma.from_documents(documents=doc_splits, collection_name='energetic_docs',
    #                                     embedding=NVIDIAEmbeddings())
    # vectorstore = Chroma.from_documents(documents=doc_splits, collection_name='energetic_docs',
    #                                     embedding=OpenAIEmbeddings(model='text-embedding-3-large'))
    # vectorstore = Chroma.from_documents(documents=doc_splits, collection_name='energetic_docs',
    #                                     embedding=HuggingFaceEmbeddings(
    #                                         model_name="sentence-transformers/allenai-specter"))
    # vectorstore = Chroma.from_documents(documents=doc_splits, collection_name='energetic_docs',
    #                                     embedding=HuggingFaceEmbeddings(
    #                                         model_name="sentence-transformers/all-MiniLM-L6-v2"))
    vectorstore = Chroma.from_documents(documents=doc_splits, collection_name='energetic_docs',
                                        embedding=ChemBERT_ChEMBL_pretrained_embeddings())

    retriever = vectorstore.as_retriever(search_type='similarity_score_threshold', search_kwargs={'k': 10, 'score_threshold': 0.9})
    retrieved_chunks = retriever.invoke(query)
    results = []
    for i, chunk in enumerate(retrieved_chunks):
        results.append({'Content': chunk.page_content, 'Title': chunk.metadata['Title'],
                        'Authors': chunk.metadata['Authors']})
        # results.append(chunk.page_content)
    return results
