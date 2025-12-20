"""
RAG-based modification strategy using Arxiv search and LangChain.
"""

import os
import logging
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .modification_tools import apply_all_modifications

logger = logging.getLogger(__name__)


class RAGModificationStrategy:
    """RAG-based molecular modification using Arxiv literature."""
    
    def __init__(self, config):
        """
        Initialize RAG strategy.
        
        Args:
            config: RAGConfig object
        """
        self.config = config
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        
        if config.enable_rag:
            self._initialize_llm()
            self._initialize_embeddings()
    
    def _initialize_llm(self):
        """Initialize ChatOpenAI."""
        try:
            self.llm = ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                api_key=self.config.openai_api_key
            )
            logger.info(f"Initialized ChatOpenAI with model {self.config.llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize OpenAI embeddings."""
        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=self.config.openai_api_key
            )
            logger.info(f"Initialized embeddings with model {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def generate_rag_query(self, parent_smiles: str, property_gap: Dict[str, float]) -> str:
        """
        Generate an Arxiv search query based on property gaps.
        
        Args:
            parent_smiles: Parent molecule SMILES
            property_gap: Dictionary of property gaps (target - current)
            
        Returns:
            Search query string
        """
        # Analyze which properties need improvement
        needs_increase = []
        needs_decrease = []
        
        for prop, gap in property_gap.items():
            if gap > 0:
                needs_increase.append(prop)
            elif gap < 0:
                needs_decrease.append(prop)
        
        # Build query based on properties
        query_parts = ["energetic materials", "high energy density"]
        
        if 'Density' in needs_increase:
            query_parts.append("increase density")
        if 'Det Velocity' in needs_increase or 'Det Pressure' in needs_increase:
            query_parts.append("detonation performance")
            query_parts.append("nitro compounds")
        if 'Hf solid' in needs_increase:
            query_parts.append("formation enthalpy")
        
        # Add structural keywords
        query_parts.extend(["molecular design", "functional groups"])
        
        query = " ".join(query_parts)
        logger.info(f"Generated RAG query: {query}")
        return query
    
    def search_arxiv_papers(self, query: str) -> List[Document]:
        """
        Search Arxiv for relevant papers.
        
        Args:
            query: Search query
            
        Returns:
            List of Document objects
        """
        try:
            loader = ArxivLoader(
                query=query,
                load_max_docs=self.config.arxiv_max_results
            )
            docs = loader.load()
            logger.info(f"Retrieved {len(docs)} papers from Arxiv")
            return docs
        except Exception as e:
            logger.error(f"Error searching Arxiv: {e}")
            return []
    
    def chunk_and_embed_papers(self, papers: List[Document]) -> Optional[FAISS]:
        """
        Chunk papers and create vector store using FAISS.
        
        Args:
            papers: List of paper documents
            
        Returns:
            FAISS vectorstore
        """
        if not papers:
            return None
        
        try:
            # Use RecursiveCharacterTextSplitter for consistent chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            # Split documents
            chunks = text_splitter.split_documents(papers)
            logger.info(f"Split papers into {len(chunks)} chunks")
            
            # Create FAISS vector store (much more stable than ChromaDB)
            try:
                logger.info(f"Creating FAISS vectorstore with {len(chunks)} chunks...")
                vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
                logger.info(f"Successfully created FAISS vectorstore")
                return vectorstore
            except Exception as ve:
                logger.error(f"FAISS creation failed: {ve}")
                import traceback
                logger.error(traceback.format_exc())
                return None
        
        except Exception as e:
            logger.error(f"Error chunking and embedding papers: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def query_rag_system(self, question: str, vectorstore: FAISS, k: int = 5) -> str:
        """
        Query the RAG system for relevant context.
        
        Args:
            question: Question to ask
            vectorstore: Vector store to query
            k: Number of documents to retrieve
            
        Returns:
            Retrieved context as string
        """
        try:
            docs = vectorstore.similarity_search(question, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            logger.error(f"Error querying vectorstore: {e}")
            return ""
    
    def extract_modification_strategies(self, parent_smiles: str, 
                                       property_gap: Dict[str, float],
                                       context: str) -> List[str]:
        """
        Use LLM to extract modification strategies from context.
        
        Args:
            parent_smiles: Parent molecule SMILES
            property_gap: Property gaps
            context: Retrieved context from papers
            
        Returns:
            List of suggested SMILES modifications
        """
        # Build prompt
        gap_str = ", ".join([f"{k}: {v:+.2f}" for k, v in property_gap.items() if abs(v) > 0.01])
        
        prompt = f"""You are an expert in energetic materials chemistry. Given a parent molecule and desired property changes, suggest specific molecular modifications.

Parent Molecule (SMILES): {parent_smiles}

Property Gaps (target - current):
{gap_str}

Relevant Literature Context:
{context[:2000]}  

Based on the literature and chemical principles, suggest {self.config.max_modifications_per_call} specific modifications to the parent molecule. 
Focus on functional group additions, substitutions, or structural changes that would improve the target properties.

Provide your answer as a numbered list of modification strategies in plain text (e.g., "1. Add nitro group to aromatic ring", "2. Replace hydrogen with azido group").
Be specific and chemistry-focused, but concise."""

        try:
            response = self.llm.invoke(prompt)
            strategies_text = response.content
            logger.info(f"LLM suggested strategies:\n{strategies_text}")
            
            # Parse strategies (simple numbered list extraction)
            strategies = []
            for line in strategies_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering
                    strategy = line.lstrip('0123456789.-•) ').strip()
                    if strategy:
                        strategies.append(strategy)
            
            return strategies
        
        except Exception as e:
            logger.error(f"Error extracting strategies from LLM: {e}")
            return []
    
    def apply_rag_modifications(self, smiles: str, strategies: List[str], target_count: int = 10) -> List[str]:
        """
        Apply RAG-suggested modifications to molecule.
        
        For now, this is a simplified implementation that uses the default
        modification tools. In a full implementation, you would parse the
        LLM strategies and apply them using SMARTS reactions.
        
        Args:
            smiles: Parent SMILES
            strategies: List of modification strategies
            target_count: Target number of modifications to return
            
        Returns:
            List of modified SMILES (up to target_count)
        """
        # Simplified: use default modifications
        # In a full implementation, parse strategies and apply specific transformations
        logger.info(f"Applying RAG strategies to {smiles} (target: {target_count} modifications)")
        modifications = apply_all_modifications(smiles)
        
        # Return up to target_count unique modifications
        return modifications[:target_count]
    
    def rag_modification_strategy(self, smiles: str, property_gap: Dict[str, float], target_count: int = 10) -> List[str]:
        """
        Main RAG-based modification strategy.
        
        Args:
            smiles: Parent molecule SMILES
            property_gap: Property gaps (target - current)
            target_count: Target number of modifications to generate
            
        Returns:
            List of modified SMILES (attempts to generate target_count modifications)
        """
        if not self.config.enable_rag:
            logger.info("RAG disabled, returning empty list")
            return []
        
        try:
            # 1. Generate query
            query = self.generate_rag_query(smiles, property_gap)
            
            # 2. Search Arxiv
            papers = self.search_arxiv_papers(query)
            if not papers:
                logger.warning("No papers found, falling back")
                return []
            
            # 3. Chunk and embed
            vectorstore = self.chunk_and_embed_papers(papers)
            if vectorstore is None:
                return []
            
            # 4. Query for relevant info
            try:
                question = f"How to modify molecules to improve {', '.join(property_gap.keys())}?"
                logger.info(f"Querying vectorstore with: {question}")
                context = self.query_rag_system(question, vectorstore)
                
                if not context:
                    logger.warning("No context retrieved from vectorstore")
                    return []
                
                logger.info(f"Retrieved context ({len(context)} chars)")
            except Exception as qe:
                logger.error(f"Vectorstore query failed: {qe}")
                import traceback
                logger.error(traceback.format_exc())
                return []
            
            # 5. Extract strategies
            try:
                strategies = self.extract_modification_strategies(smiles, property_gap, context)
                if not strategies:
                    logger.warning("No strategies extracted from LLM")
                    return []
            except Exception as se:
                logger.error(f"Strategy extraction failed: {se}")
                return []
            
            # 6. Apply modifications to generate target_count
            modified_smiles = self.apply_rag_modifications(smiles, strategies, target_count=target_count)
            
            logger.info(f"RAG strategy generated {len(modified_smiles)} modifications (target: {target_count})")
            return modified_smiles
        
        except Exception as e:
            logger.error(f"RAG strategy failed: {e}")
            return []


def default_modification_strategy(smiles: str, property_gap: Dict[str, float]) -> List[str]:
    """
    Default heuristic-based modification strategy (fallback).
    
    Args:
        smiles: Parent molecule SMILES
        property_gap: Property gaps (target - current)
        
    Returns:
        List of modified SMILES
    """
    logger.info(f"Using default modification strategy for {smiles}")
    
    # Use standard modification tools
    from .modification_tools import (
        addition_modification,
        substitution_modification,
        ring_modification
    )
    
    modifications = []
    
    # Strategy based on property gaps
    if property_gap.get('Density', 0) > 0:
        # Increase density: add heavy atoms, fuse rings
        modifications.extend(ring_modification(smiles))
        modifications.extend(addition_modification(smiles, ['[N+](=O)[O-]']))
    
    if property_gap.get('Det Velocity', 0) > 0 or property_gap.get('Det Pressure', 0) > 0:
        # Increase detonation properties: add energetic groups
        modifications.extend(addition_modification(smiles, ['[N+](=O)[O-]', 'N=[N+]=[N-]']))
        modifications.extend(substitution_modification(smiles))
    
    if not modifications:
        # If no specific strategy applies, use all modifications
        modifications = apply_all_modifications(smiles)
    
    # Remove duplicates
    modifications = list(set(modifications))
    
    return modifications[:5]  # Return max 5
