import os
import json
import requests
import ast
import torch
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from tqdm import tqdm
import re

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}", 
    "Accept": "application/vnd.github.v3+json"
}

# Repository information
OWNER = "pandas-dev"  # Change this to the owner of the repository you want to analyze
REPO = "pandas"  # Change this to the repository name you want to analyze
OUTPUT_DIR = f"code_embeddings_{OWNER}_{REPO}"

# Initialize the embedding model (using CodeBERT as an example)
MODEL_NAME = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def fetch_repo_contents(owner, repo, path="", recursive=True):
    """
    Fetch repository contents recursively.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"Error fetching contents: {response.status_code}")
        print(f"Response: {response.text}")
        return []
    
    contents = response.json()
    items = []
    
    # If response is a list, it's a directory
    if isinstance(contents, list):
        for content in contents:
            if content["type"] == "file":
                file_content = fetch_file_content(content["download_url"])
                items.append({
                    "path": content["path"],
                    "type": "file",
                    "content": file_content
                })
            elif content["type"] == "dir" and recursive:
                # Recursively fetch contents of subdirectories
                items.extend(fetch_repo_contents(owner, repo, content["path"], recursive))
    # If response is a dictionary, it's a file
    else:
        file_content = fetch_file_content(contents["download_url"])
        items.append({
            "path": contents["path"],
            "type": "file",
            "content": file_content
        })
    
    return items

def fetch_file_content(download_url):
    """
    Fetch the content of a file.
    """
    try:
        response = requests.get(download_url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error fetching file content: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None

def generate_embedding(code_text):
    """
    Generate embeddings for a piece of code.
    """
    inputs = tokenizer(code_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the [CLS] token embedding as the code embedding
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]
    return embedding

def extract_python_components(file_path, file_content):
    """
    Extract functions and classes from Python file.
    """
    if not file_path.endswith('.py') or file_content is None:
        return None, None
    
    functions = []
    classes = []
    
    try:
        tree = ast.parse(file_content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function
                function_code = file_content[node.lineno-1:node.end_lineno]
                docstring = ast.get_docstring(node)
                
                functions.append({
                    "name": node.name,
                    "code": function_code,
                    "docstring": docstring,
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno
                })
                
            elif isinstance(node, ast.ClassDef):
                # Extract class
                class_code = file_content[node.lineno-1:node.end_lineno]
                docstring = ast.get_docstring(node)
                
                # Get class methods
                methods = []
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        methods.append(child.name)
                
                classes.append({
                    "name": node.name,
                    "code": class_code,
                    "docstring": docstring,
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno,
                    "methods": methods
                })
                
        return functions, classes
        
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return [], []

def extract_javascript_components(file_path, file_content):
    """
    Extract functions and classes from JavaScript/TypeScript file.
    """
    if not any(file_path.endswith(ext) for ext in ['.js', '.jsx', '.ts', '.tsx']) or file_content is None:
        return None, None
        
    functions = []
    classes = []
    
    # Simple regex for function extraction
    # This is a simplified approach - a proper parser would be better
    function_pattern = r'(function\s+(\w+)\s*\([^)]*\)\s*{)|((const|let|var)\s+(\w+)\s*=\s*function\s*\([^)]*\)\s*{)|((const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>)'
    class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{'
    
    # Find functions
    for match in re.finditer(function_pattern, file_content):
        start_pos = match.start()
        # This is a simplified way to find the end of the function
        # A proper parser would handle nested braces correctly
        name = match.group(2) or match.group(5) or match.group(8)
        if name:
            functions.append({
                "name": name,
                "code": match.group(0),  # This only gets the function signature, not the full body
                "lineno": file_content[:start_pos].count('\n') + 1
            })
    
    # Find classes
    for match in re.finditer(class_pattern, file_content):
        start_pos = match.start()
        name = match.group(1)
        extends = match.group(2)
        
        classes.append({
            "name": name,
            "extends": extends,
            "code": match.group(0),  # This only gets the class signature, not the full body
            "lineno": file_content[:start_pos].count('\n') + 1
        })
    
    return functions, classes

def process_repository(owner, repo):
    """
    Process a repository to generate code embeddings.
    """
    print(f"Downloading repository {owner}/{repo}...")
    contents = fetch_repo_contents(owner, repo)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(OUTPUT_DIR)
    
    file_embeddings = []
    function_embeddings = []
    class_embeddings = []
    
    print("Generating embeddings...")
    for item in tqdm(contents):
        if item["type"] == "file" and item["content"] is not None:
            file_path = item["path"]
            file_content = item["content"]
            
            # Skip non-code files or files that are too large
            if (file_path.endswith(('.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h')) 
                and len(file_content) < 100000):
                
                # Generate file-level embedding
                file_embedding = generate_embedding(file_content)
                file_embeddings.append({
                    "path": file_path,
                    "embedding": file_embedding
                })
                
                # Extract and generate function/class level embeddings
                if file_path.endswith('.py'):
                    functions, classes = extract_python_components(file_path, file_content)
                elif any(file_path.endswith(ext) for ext in ['.js', '.jsx', '.ts', '.tsx']):
                    functions, classes = extract_javascript_components(file_path, file_content)
                else:
                    functions, classes = None, None
                
                if functions:
                    for func in functions:
                        function_code = func.get("code", "")
                        if function_code:
                            func_embedding = generate_embedding(function_code)
                            function_embeddings.append({
                                "file_path": file_path,
                                "name": func.get("name", "unknown"),
                                "embedding": func_embedding,
                                "docstring": func.get("docstring", ""),
                                "lineno": func.get("lineno", 0)
                            })
                
                if classes:
                    for cls in classes:
                        class_code = cls.get("code", "")
                        if class_code:
                            class_embedding = generate_embedding(class_code)
                            class_embeddings.append({
                                "file_path": file_path,
                                "name": cls.get("name", "unknown"),
                                "embedding": class_embedding,
                                "docstring": cls.get("docstring", ""),
                                "methods": cls.get("methods", []),
                                "lineno": cls.get("lineno", 0)
                            })
    
    # Save embeddings to files
    with open(os.path.join(OUTPUT_DIR, "file_embeddings.json"), "w") as f:
        json.dump(file_embeddings, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "function_embeddings.json"), "w") as f:
        json.dump(function_embeddings, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "class_embeddings.json"), "w") as f:
        json.dump(class_embeddings, f, indent=2)
    
    print(f"Embeddings saved to {OUTPUT_DIR}")
    print(f"Generated {len(file_embeddings)} file embeddings, {len(function_embeddings)} function embeddings, and {len(class_embeddings)} class embeddings")

def main():
    # Process the specified repository
    process_repository(OWNER, REPO)
    
    # Example of how to use the embeddings for similarity search
    print("\nExample: Finding similar functions...")
    with open(os.path.join(OUTPUT_DIR, "function_embeddings.json"), "r") as f:
        function_embeddings = json.load(f)
    
    if len(function_embeddings) > 1:
        # Convert the first function's embedding to a tensor
        query_embedding = torch.tensor(function_embeddings[0]["embedding"])
        
        # Calculate cosine similarity with all other functions
        similarities = []
        for i, func in enumerate(function_embeddings[1:], 1):
            target_embedding = torch.tensor(func["embedding"])
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0), 
                target_embedding.unsqueeze(0)
            ).item()
            similarities.append((i, similarity))
        
        # Print the most similar functions
        print(f"Functions most similar to {function_embeddings[0]['name']}:")
        for idx, sim in sorted(similarities, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {function_embeddings[idx]['name']} (similarity: {sim:.4f})")

if __name__ == "__main__":
    main()