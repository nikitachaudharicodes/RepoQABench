import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm

class JsonCodeRetrieval:
    """
    A system for retrieving relevant code snippets based on GitHub issue text.
    Works directly with JSON-based embeddings without requiring FAISS.
    """
    
    def __init__(
        self,
        code_embeddings_dir: str,
        issues_file: str,
        model_name: str = "microsoft/codebert-base",
        use_gpu: bool = torch.cuda.is_available()
    ):
        """
        Initialize the retrieval system.
        
        Args:
            code_embeddings_dir: Directory containing code embeddings JSON files
            issues_file: Path to the JSON file containing issue data
            model_name: Name of the CodeBERT model to use
            use_gpu: Whether to use GPU for inference
        """
        self.code_embeddings_dir = code_embeddings_dir
        self.issues_file = issues_file
        self.model_name = model_name
        
        # Set up device
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CodeBERT model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load issues
        self.issues = self._load_issues(issues_file)
        print(f"Loaded {len(self.issues['qa_pairs'])} issues")
        
        # Load code embeddings
        self.file_embeddings = []
        self.function_embeddings = []
        self.class_embeddings = []
        self._load_code_embeddings(code_embeddings_dir)
    
    def _load_issues(self, issues_file: str) -> Dict[str, Any]:
        """Load issues from a JSON file."""
        with open(issues_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_code_embeddings(self, embeddings_dir: str) -> None:
        """Load code embeddings from JSON files in the specified directory."""
        # Load file embeddings
        file_path = os.path.join(embeddings_dir, "file_embeddings.json")
        if os.path.exists(file_path):
            print(f"Loading file embeddings from {file_path}...")
            with open(file_path, 'r') as f:
                self.file_embeddings = json.load(f)
            print(f"Loaded {len(self.file_embeddings)} file embeddings")
        
        # Load function embeddings
        function_path = os.path.join(embeddings_dir, "function_embeddings.json")
        if os.path.exists(function_path):
            print(f"Loading function embeddings from {function_path}...")
            with open(function_path, 'r') as f:
                self.function_embeddings = json.load(f)
            print(f"Loaded {len(self.function_embeddings)} function embeddings")
        
        # Load class embeddings
        class_path = os.path.join(embeddings_dir, "class_embeddings.json")
        if os.path.exists(class_path):
            print(f"Loading class embeddings from {class_path}...")
            with open(class_path, 'r') as f:
                self.class_embeddings = json.load(f)
            print(f"Loaded {len(self.class_embeddings)} class embeddings")
        
        # Calculate total embeddings
        total = len(self.file_embeddings) + len(self.function_embeddings) + len(self.class_embeddings)
        print(f"Total embeddings loaded: {total}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for a piece of text using CodeBERT."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype('float32')
        
        return embedding[0]
    
    def extract_code_context(self, issue_text: str) -> Dict[str, Any]:
        """Extract code-related context from issue text."""
        # Extract code blocks
        code_block_pattern = r'```(?:\w+)?\s*([\s\S]*?)```'
        code_blocks = re.findall(code_block_pattern, issue_text)
        
        # Extract inline code
        inline_code_pattern = r'`([^`]+)`'
        inline_codes = re.findall(inline_code_pattern, issue_text)
        
        # Extract function and class references
        function_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        functions = list(set(re.findall(function_pattern, issue_text)))
        
        class_pattern = r'\b([A-Z][a-zA-Z0-9_]*)\b'
        classes = list(set(re.findall(class_pattern, issue_text)))
        
        # Extract file references
        file_pattern = r'\b[\w\-\.\/]+\.(js|jsx|ts|tsx|py|java|html|css|json|md|xml)\b'
        files = list(set(re.findall(file_pattern, issue_text)))
        
        return {
            "code_blocks": code_blocks,
            "inline_code": inline_codes,
            "functions": functions,
            "classes": classes,
            "files": files
        }
    
    def process_issue(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single issue to extract key information."""
        issue_title = issue.get("issue_title", "")
        issue_body = issue.get("issue_body", "")
        
        # Combine comments
        comments_text = " ".join([comment.get("comment", "") for comment in issue.get("comments", [])])
        
        # Combined text for analysis
        full_text = f"{issue_title} {issue_body} {comments_text}"
        
        # Extract code context
        code_context = self.extract_code_context(full_text)
        
        # Determine issue type from labels
        labels = issue.get("labels", [])
        issue_type = "enhancement" if "enhancement" in labels else "bug" if "bug" in labels else "other"
        
        return {
            "issue_number": issue.get("issue_number"),
            "issue_title": issue_title,
            "issue_type": issue_type,
            "code_context": code_context,
            "full_text": full_text
        }
    
    def formulate_search_query(self, processed_issue: Dict[str, Any]) -> str:
        """Formulate a search query from a processed issue."""
        query_parts = [processed_issue["issue_title"]]
        
        # Add code references as explicit parts of the query
        code_context = processed_issue["code_context"]
        
        if code_context["files"]:
            query_parts.append("Files: " + ", ".join(code_context["files"]))
        
        if code_context["functions"]:
            query_parts.append("Functions: " + ", ".join(code_context["functions"]))
        
        if code_context["classes"]:
            query_parts.append("Classes: " + ", ".join(code_context["classes"]))
        
        # Add code blocks if they're not too long
        for code in code_context["code_blocks"]:
            if len(code) < 500:  # Limit code block size
                query_parts.append(f"Code: {code}")
        
        # Include a few inline code references
        if code_context["inline_code"]:
            inline_samples = code_context["inline_code"][:5]  # Limit to 5 inline references
            query_parts.append("References: " + ", ".join(inline_samples))
        
        return " ".join(query_parts)
    
    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        # Convert to numpy arrays if they aren't already
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
            
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        # Handle zero division
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def search_code(
        self, 
        query_embedding: np.ndarray, 
        entity_type: Optional[str] = None, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for code snippets based on query embedding.
        
        Args:
            query_embedding: The embedding vector for the query
            entity_type: Optional filter for entity type (file, function, class)
            k: Number of results to return
        
        Returns:
            List of dictionaries containing search results
        """
        results = []
        
        # Search file embeddings
        if entity_type is None or entity_type == "file":
            for item in self.file_embeddings:
                similarity = self.calculate_cosine_similarity(query_embedding, item["embedding"])
                result = {k: v for k, v in item.items() if k != "embedding"}
                result["entity_type"] = "file"
                result["similarity"] = similarity
                results.append(result)
        
        # Search function embeddings
        if entity_type is None or entity_type == "function":
            for item in self.function_embeddings:
                similarity = self.calculate_cosine_similarity(query_embedding, item["embedding"])
                result = {k: v for k, v in item.items() if k != "embedding"}
                result["entity_type"] = "function"
                result["similarity"] = similarity
                results.append(result)
        
        # Search class embeddings
        if entity_type is None or entity_type == "class":
            for item in self.class_embeddings:
                similarity = self.calculate_cosine_similarity(query_embedding, item["embedding"])
                result = {k: v for k, v in item.items() if k != "embedding"}
                result["entity_type"] = "class"
                result["similarity"] = similarity
                results.append(result)
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top k results
        return results[:k]
    
    def find_relevant_code_for_issue(
        self, 
        issue_number: int,
        k: int = 5, 
        entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find relevant code snippets for a specific issue."""
        # Find the issue
        issue = next((i for i in self.issues["qa_pairs"] if i.get("issue_number") == issue_number), None)
        if not issue:
            print(f"Issue #{issue_number} not found")
            return []
        
        # Process the issue
        processed_issue = self.process_issue(issue)
        
        # Formulate a search query
        query = self.formulate_search_query(processed_issue)
        
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)
        
        # Search for relevant code
        results = self.search_code(query_embedding, entity_type, k)
        
        return results
    
    def batch_process_issues(
        self, 
        issue_numbers: Optional[List[int]] = None,
        k: int = 5
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Process a batch of issues and find relevant code for each."""
        if not issue_numbers:
            # Process all issues if none specified
            issue_numbers = [issue.get("issue_number") for issue in self.issues["qa_pairs"]]
        
        results = {}
        for issue_number in tqdm(issue_numbers, desc="Processing issues"):
            issue_results = self.find_relevant_code_for_issue(issue_number, k=k)
            results[issue_number] = issue_results
        
        return results
    
    def evaluate_solution_fit(self, issue: Dict[str, Any], code_snippet: Dict[str, Any]) -> float:
        """
        Evaluate how well a code snippet fits as a solution for an issue.
        Returns a score between 0 and 1.
        """
        # This is a simple heuristic that could be improved with ML
        issue_text = issue.get("full_text", "")
        
        # Check if the file path appears in the issue
        file_match = 0.0
        if "file_path" in code_snippet and code_snippet["file_path"] in issue_text:
            file_match = 0.3
        
        # Check if the function/class name appears in the issue
        name_match = 0.0
        if "name" in code_snippet and code_snippet["name"] in issue_text:
            name_match = 0.3
        
        # Use the similarity score from the search
        similarity = code_snippet.get("similarity", 0.0)
        
        # Combine scores (with similarity having the highest weight)
        combined_score = 0.4 * similarity + 0.3 * file_match + 0.3 * name_match
        
        return min(combined_score, 1.0)  # Cap at 1.0

    def suggest_code_changes(
        self,
        issue_number: int,
        max_suggestions: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Suggest specific code changes that could address an issue.
        Returns a list of suggestions with the code context and potential changes.
        """
        # Find the issue
        issue = next((i for i in self.issues["qa_pairs"] if i.get("issue_number") == issue_number), None)
        if not issue:
            print(f"Issue #{issue_number} not found")
            return []
        
        # Process the issue
        processed_issue = self.process_issue(issue)
        
        # Find relevant code
        relevant_code = self.find_relevant_code_for_issue(issue_number, k=max_suggestions*2)
        
        # Evaluate each code snippet as a potential solution
        suggestions = []
        for code in relevant_code:
            fit_score = self.evaluate_solution_fit(processed_issue, code)
            
            if fit_score > 0.5:  # Only suggest if it's a reasonable fit
                suggestion = {
                    "code_metadata": code,
                    "fit_score": fit_score,
                    "suggestion_type": "refactor" if processed_issue["issue_type"] == "enhancement" else "fix",
                    "confidence": "high" if fit_score > 0.8 else "medium" if fit_score > 0.65 else "low"
                }
                suggestions.append(suggestion)
        
        # Sort by fit score and limit to max_suggestions
        suggestions.sort(key=lambda x: x["fit_score"], reverse=True)
        return suggestions[:max_suggestions]


def main():
    """Example usage of the JsonCodeRetrieval system."""
    # Initialize the system
    retrieval = JsonCodeRetrieval(
        code_embeddings_dir="code_embeddings_fossology_FOSSologyUI",
        issues_file="fossologyUI_repo.json"
    )
    
    # Example 1: Find relevant code for a specific issue
    issue_number = 253  # Dark Mode Switch Delay issue
    print(f"\nFinding relevant code for issue #{issue_number}:")
    results = retrieval.find_relevant_code_for_issue(issue_number, k=3)
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result.get('entity_type', '').capitalize()}: {result.get('name', 'N/A')}")
        print(f"   File: {result.get('file_path', 'N/A')}")
        print(f"   Similarity: {result.get('similarity', 0):.4f}")
        
        if "docstring" in result and result["docstring"]:
            print(f"   Description: {result['docstring']}")
    
    # Example 2: Suggest code changes for an issue
    print(f"\nSuggesting code changes for issue #{issue_number}:")
    suggestions = retrieval.suggest_code_changes(issue_number)
    
    for i, suggestion in enumerate(suggestions):
        code = suggestion["code_metadata"]
        print(f"\n{i+1}. {suggestion['suggestion_type'].upper()} with {suggestion['confidence'].upper()} confidence:")
        print(f"   {code.get('entity_type', '').capitalize()}: {code.get('name', 'N/A')}")
        print(f"   File: {code.get('file_path', 'N/A')}")
        print(f"   Fit score: {suggestion['fit_score']:.4f}")
    
    # Example 3: Batch process multiple issues
    batch_issues = [253, 173, 225]  # A mix of different issue types
    print(f"\nBatch processing issues: {batch_issues}")
    batch_results = retrieval.batch_process_issues(batch_issues, k=2)
    
    for issue_id, results in batch_results.items():
        issue = next((i for i in retrieval.issues["qa_pairs"] if i.get("issue_number") == issue_id), None)
        if issue:
            print(f"\nIssue #{issue_id}: {issue.get('issue_title')}")
            print(f"Found {len(results)} relevant code snippets")
            
            if results:
                top_result = results[0]
                print(f"Top match: {top_result.get('entity_type')} '{top_result.get('name', 'N/A')}' " +
                      f"in {top_result.get('file_path', 'N/A')}")

if __name__ == "__main__":
    main()