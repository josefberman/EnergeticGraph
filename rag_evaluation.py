import simplejson as json
import time
from typing import List, Dict, Tuple
from RAG import retrieve_context
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from auxiliary import ChemBERT_ChEMBL_pretrained_embeddings
import warnings
warnings.filterwarnings("ignore")
import numpy as np

class RAGEvaluator:
    def __init__(self):
        self.queries = [
            {
                "query": "Heat of formation and enthalpy of explosion of nitramine‐based energetics",
                "keywords": ["heat of formation", "enthalpy of explosion", "nitramine", "thermochemistry", "calorimetry", "DFT calculations"]
            },
            {
                "query": "Crystal density and packing efficiency effects on detonation velocity",
                "keywords": ["crystal density", "packing efficiency", "detonation velocity", "X-ray diffraction", "unit-cell parameters", "pycnometry"]
            },
            {
                "query": "Impact and friction sensitivity measurements of high-nitrogen compounds",
                "keywords": ["impact sensitivity", "friction sensitivity", "BAM drop-weight", "Peters friction tester", "high-nitrogen", "safety parameters"]
            },
            {
                "query": "Thermal decomposition kinetics and activation energy of nitrate-ester propellants",
                "keywords": ["thermal decomposition", "kinetics", "activation energy", "DSC/TGA", "Arrhenius plot", "decomposition intermediates"]
            },
            {
                "query": "Gas‐volume generation and combustion products in composite propellants",
                "keywords": ["gas generation", "combustion products", "composite propellant", "burning rate", "gas chromatography", "CO", "CO2", "N2 yields"]
            },
            {
                "query": "Viscosity and rheological behavior of energetic plasticizer formulations",
                "keywords": ["viscosity", "rheology", "plasticizer", "shear rate", "viscoelastic modulus", "temperature dependence"]
            },
            {
                "query": "Crystal morphology influence on mechanical strength of pressed explosive pellets",
                "keywords": ["crystal morphology", "pellet compression", "mechanical strength", "SEM imaging", "granulometry", "hardness test"]
            },
            {
                "query": "Oxygen balance and theoretical detonation pressure of fuel-rich mixtures",
                "keywords": ["oxygen balance", "detonation pressure", "thermochemical code", "Cheetah", "fuel-rich", "equilibrium composition", "Chapman–Jouguet point"]
            },
            {
                "query": "Heat capacity and thermal conductivity of solid energetics at cryogenic temperatures",
                "keywords": ["heat capacity", "thermal conductivity", "cryogenic", "laser-flash analysis", "calorimeter", "thermal diffusivity"]
            },
            {
                "query": "Sensitivity and performance trade-offs in metalized energetic formulations (Al, Mg additives)",
                "keywords": ["metalized formulation", "aluminum particles", "magnesium additives", "sensitivity", "specific impulse", "burn-rate enhancement"]
            }
        ]
    
    def evaluate_relevance(self, content: str, keywords: List[str]) -> bool:
        """
        Evaluate if a document content is relevant based on keyword presence.
        Returns True if at least 2 keywords are found in the content.
        """
        content_lower = content.lower()
        keyword_count = 0
        
        for keyword in keywords:
            # Handle special characters in keywords
            # For regular keywords, use word boundaries
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, content_lower):
                keyword_count += 1
        
        # Consider relevant if at least 2 keywords are found
        return keyword_count >= 2
    
    def calculate_semantic_similarity(self, query, content):
        model = ChemBERT_ChEMBL_pretrained_embeddings()
        query_embedding = model.embed_documents([query])  # Wrap in list to get 2D array
        content_embedding = model.embed_documents([content])  # Wrap in list to get 2D array
        similarity = cosine_similarity(query_embedding, content_embedding)[0][0]
        return similarity >= 0.9
    
    def calculate_precision_at_k(self, relevant_docs: List[bool], k: int) -> float:
        """
        Calculate precision@k for a given list of relevance scores.
        """
        if k == 0:
            return 0.0
        
        k = min(k, len(relevant_docs))
        relevant_count = sum(relevant_docs[:k])
        return relevant_count / k
    
    def evaluate_query(self, query_data: Dict, max_k: int = 10) -> Dict:
        """
        Evaluate a single query and return precision@k metrics.
        """
        query = query_data["query"]
        keywords = query_data["keywords"]
        
        print(f"\nEvaluating query: {query}")
        print(f"Keywords: {', '.join(keywords)}")
        
        try:
            # Retrieve documents using the RAG module
            retrieved_docs = retrieve_context(query)
            
            # Evaluate relevance for each retrieved document
            relevance_scores = []
            relevant_docs_details = []
            
            for i, doc in enumerate(retrieved_docs):
                content = doc.get('Content', '')
                title = doc.get('Title', 'Unknown Title')
                authors = doc.get('Authors', 'Unknown Authors')
                
                # is_relevant = self.evaluate_relevance(content, keywords)
                is_relevant = self.calculate_semantic_similarity(query, content)
                relevance_scores.append(is_relevant)
                
                if is_relevant:
                    relevant_docs_details.append({
                        'rank': i + 1,
                        'title': title,
                        'authors': authors,
                        'content_preview': content[:200] + "..." if len(content) > 200 else content
                    })
                
                print(f"  Doc {i+1}: {'✓' if is_relevant else '✗'} - {title}")
            
            # Calculate precision@k for k = 1 to max_k
            precision_at_k = {}
            for k in range(1, max_k + 1):
                precision_at_k[f"precision@{k}"] = self.calculate_precision_at_k(relevance_scores, k)
            
            # Calculate additional metrics
            total_retrieved = len(retrieved_docs)
            total_relevant = sum(relevance_scores)
            
            results = {
                'query': query,
                'keywords': keywords,
                'total_retrieved': total_retrieved,
                'total_relevant': total_relevant,
                'precision_at_k': precision_at_k,
                # 'relevant_docs': relevant_docs_details,
                # 'all_relevance_scores': relevance_scores
            }
            
            print(f"  Total retrieved: {total_retrieved}")
            print(f"  Total relevant: {total_relevant}")
            print(f"  Precision@5: {precision_at_k['precision@5']:.3f}")
            print(f"  Precision@10: {precision_at_k['precision@10']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"  Error evaluating query: {str(e)}")
            return {
                'query': query,
                'keywords': keywords,
                'error': str(e),
                'total_retrieved': 0,
                'total_relevant': 0,
                'precision_at_k': {f"precision@{k}": 0.0 for k in range(1, max_k + 1)},
                # 'relevant_docs': [],
                # 'all_relevance_scores': []
            }
    
    def run_evaluation(self, max_k: int = 10) -> Dict:
        """
        Run evaluation on all queries and return comprehensive results.
        """
        print("Starting RAG Evaluation")
        print("=" * 50)
        
        all_results = []
        overall_stats = defaultdict(list)
        
        for i, query_data in enumerate(self.queries):
            print(f"\nQuery {i+1}/10:")
            result = self.evaluate_query(query_data, max_k)
            all_results.append(result)
            
            # Collect statistics for overall analysis
            if 'error' not in result:
                for k in range(1, max_k + 1):
                    overall_stats[f"precision@{k}"].append(result['precision_at_k'][f"precision@{k}"])
                overall_stats['total_relevant'].append(result['total_relevant'])
                overall_stats['total_retrieved'].append(result['total_retrieved'])
            
            # Add delay to avoid overwhelming the arXiv API
            if i < len(self.queries) - 1:
                print("Waiting 2 seconds before next query...")
                time.sleep(2)
        
        # Calculate overall statistics
        overall_summary = {}
        for metric, values in overall_stats.items():
            if values:
                overall_summary[f"avg_{metric}"] = sum(values) / len(values)
                overall_summary[f"std_{metric}"] = np.std(values)
                overall_summary[f"min_{metric}"] = min(values)
                overall_summary[f"max_{metric}"] = max(values)
        
        # Calculate overall precision@k averages
        overall_precision_at_k = {}
        for k in range(1, max_k + 1):
            precision_values = overall_stats[f"precision@{k}"]
            if precision_values:
                overall_precision_at_k[f"precision@{k}"] = sum(precision_values) / len(precision_values)
        
        final_results = {
            'individual_results': all_results,
            'overall_summary': overall_summary,
            'overall_precision_at_k': overall_precision_at_k,
            'total_queries': len(self.queries),
            'successful_queries': len([r for r in all_results if 'error' not in r])
        }
        
        return final_results
    
    def print_summary(self, results: Dict):
        """
        Print a comprehensive summary of the evaluation results.
        """
        print("\n" + "=" * 60)
        print("RAG EVALUATION SUMMARY")
        print("=" * 60)
        
        summary = results['overall_summary']
        precision_at_k = results['overall_precision_at_k']
        
        print(f"Total queries evaluated: {results['total_queries']}")
        print(f"Successful queries: {results['successful_queries']}")
        print(f"Failed queries: {results['total_queries'] - results['successful_queries']}")
        
        if 'avg_total_retrieved' in summary:
            print(f"\nAverage documents retrieved per query: {summary['avg_total_retrieved']:.2f}")
            print(f"Average relevant documents per query: {summary['avg_total_relevant']:.2f}")
        
        print("\nPrecision@k Results:")
        print("-" * 30)
        for k in range(1, 11):
            if f"precision@{k}" in precision_at_k:
                print(f"Precision@{k:2d}: {precision_at_k[f'precision@{k}']:.3f}")
        
        print("\nDetailed Statistics:")
        print("-" * 30)
        for k in range(1, 11):
            if f"avg_precision@{k}" in summary:
                print(f"Precision@{k:2d} - Avg: {summary[f'avg_precision@{k}']:.3f}, "
                      f"Std: {summary[f'std_precision@{k}']:.3f}, "
                      f"Min: {summary[f'min_precision@{k}']:.3f}, "
                      f"Max: {summary[f'max_precision@{k}']:.3f}")
        
        # Show best and worst performing queries
        successful_results = [r for r in results['individual_results'] if 'error' not in r]
        if successful_results:
            best_query = max(successful_results, key=lambda x: x['precision_at_k']['precision@10'])
            worst_query = min(successful_results, key=lambda x: x['precision_at_k']['precision@10'])
            
            print(f"\nBest performing query (Precision@10: {best_query['precision_at_k']['precision@10']:.3f}):")
            print(f"  {best_query['query']}")
            
            print(f"\nWorst performing query (Precision@10: {worst_query['precision_at_k']['precision@10']:.3f}):")
            print(f"  {worst_query['query']}")
    
    def save_results(self, results: Dict, filename: str = "rag_evaluation_results.json"):
        """
        Save evaluation results to a JSON file.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {filename}")


def main():
    """
    Main function to run the RAG evaluation.
    """
    evaluator = RAGEvaluator()
    
    print("RAG Module Evaluation Tool")
    print("This tool will evaluate the RAG module using 10 predefined queries")
    print("and calculate precision@k metrics up to k=10.")
    print("\nNote: This evaluation may take several minutes due to API rate limiting.")
    
    # Run the evaluation
    results = evaluator.run_evaluation(max_k=10)
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    evaluator.save_results(results)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main() 