import sys
import os
import json
from json_code_retrieval import JsonCodeRetrieval

def example_dark_mode_issue():
    """Example for retrieving code related to dark mode theme switching issue."""
    retrieval = JsonCodeRetrieval(
        code_embeddings_dir="code_embeddings_fossology_FOSSologyUI",
        issues_file="fossologyUI_repo.json"
    )
    
    # Find the Dark Mode Theme Switch Delay issue (issue #253)
    issue_number = 253
    issue = next((i for i in retrieval.issues["qa_pairs"] if i.get("issue_number") == issue_number), None)
    
    if not issue:
        print(f"Issue #{issue_number} not found")
        return
    
    print("=" * 80)
    print(f"ISSUE #{issue_number}: {issue['issue_title']}")
    print("=" * 80)
    print(f"Description: {issue['issue_body'][:300]}...\n")
    
    # Process the issue to extract code context
    processed_issue = retrieval.process_issue(issue)
    
    print("EXTRACTED CODE CONTEXT:")
    print(f"- Functions mentioned: {processed_issue['code_context']['functions']}")
    print(f"- Classes mentioned: {processed_issue['code_context']['classes']}")
    print(f"- Files mentioned: {processed_issue['code_context']['files']}")
    print(f"- Inline code refs: {processed_issue['code_context']['inline_code'][:3]}")
    print(f"- Code blocks: {len(processed_issue['code_context']['code_blocks'])}")
    
    # Formulate a search query
    search_query = retrieval.formulate_search_query(processed_issue)
    print(f"\nSEARCH QUERY: {search_query[:150]}...\n")
    
    # Generate query embedding
    query_embedding = retrieval.generate_embedding(search_query)
    
    # Find relevant code
    print("SEARCHING FOR RELEVANT CODE...")
    results = retrieval.search_code(query_embedding, k=5)
    
    print("\nTOP 5 RELEVANT CODE SNIPPETS:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result.get('entity_type', '').upper()}: {result.get('name', 'Unknown')}")
        print(f"   File: {result.get('file_path', 'Unknown')}")
        print(f"   Similarity: {result.get('similarity', 0):.4f}")
    
    # Find the most relevant file and function
    if results:
        top_file = next((r for r in results if r.get('entity_type') == 'file'), None)
        top_function = next((r for r in results if r.get('entity_type') == 'function'), None)
        
        if top_file:
            print(f"\nMOST RELEVANT FILE: {top_file.get('path', 'Unknown')}")
            print(f"Similarity: {top_file.get('similarity', 0):.4f}")
        
        if top_function:
            print(f"\nMOST RELEVANT FUNCTION: {top_function.get('name', 'Unknown')}")
            print(f"In file: {top_function.get('file_path', 'Unknown')}")
            print(f"Similarity: {top_function.get('similarity', 0):.4f}")
    
    # Suggest code changes
    print("\nSUGGESTING CODE CHANGES...")
    suggestions = retrieval.suggest_code_changes(issue_number)
    
    print("\nSUGGESTIONS:")
    for i, suggestion in enumerate(suggestions):
        code = suggestion["code_metadata"]
        print(f"\n{i+1}. {suggestion['suggestion_type'].upper()} with {suggestion['confidence'].upper()} confidence:")
        print(f"   {code.get('entity_type', '').capitalize()}: {code.get('name', 'Unknown')}")
        print(f"   File: {code.get('file_path', 'Unknown')}")
        print(f"   Fit score: {suggestion['fit_score']:.4f}")

def example_search_bar_issue():
    """Example for retrieving code related to search bar issue."""
    retrieval = JsonCodeRetrieval(
        code_embeddings_dir="code_embeddings_fossology_FOSSologyUI",
        issues_file="fossologyUI_repo.json"
    )
    
    # Find the Search Bar issue (issue #225)
    issue_number = 225
    issue = next((i for i in retrieval.issues["qa_pairs"] if i.get("issue_number") == issue_number), None)
    
    if not issue:
        print(f"Issue #{issue_number} not found")
        return
    
    print("=" * 80)
    print(f"ISSUE #{issue_number}: {issue['issue_title']}")
    print("=" * 80)
    print(f"Description: {issue['issue_body'][:300]}...\n")
    
    # Find relevant code focusing specifically on functions
    print("SEARCHING FOR RELEVANT FUNCTIONS...")
    results = retrieval.find_relevant_code_for_issue(issue_number, k=3, entity_type="function")
    
    print("\nMOST RELEVANT FUNCTIONS:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result.get('name', 'Unknown')}")
        print(f"   File: {result.get('file_path', 'Unknown')}")
        print(f"   Similarity: {result.get('similarity', 0):.4f}")
    
    # Find relevant files
    print("\nSEARCHING FOR RELEVANT FILES...")
    processed_issue = retrieval.process_issue(issue)
    query = retrieval.formulate_search_query(processed_issue)
    query_embedding = retrieval.generate_embedding(query)
    file_results = retrieval.search_code(query_embedding, entity_type="file", k=2)
    
    print("\nMOST RELEVANT FILES:")
    for i, result in enumerate(file_results):
        print(f"\n{i+1}. {result.get('path', 'Unknown')}")
        print(f"   Similarity: {result.get('similarity', 0):.4f}")

def compare_multiple_issues():
    """Compare code retrieval results for multiple issues."""
    retrieval = JsonCodeRetrieval(
        code_embeddings_dir="code_embeddings_fossology_FOSSologyUI",
        issues_file="fossologyUI_repo.json"
    )
    
    # Select a few different issue types
    issue_numbers = [
        253,  # Dark Mode Switch Delay (bug)
        173,  # Dark theme toggle button (enhancement)
        225   # Search bar in browse page (bug)
    ]
    
    print("=" * 80)
    print("COMPARING CODE RETRIEVAL ACROSS MULTIPLE ISSUES")
    print("=" * 80)
    
    issue_summaries = []
    for issue_number in issue_numbers:
        issue = next((i for i in retrieval.issues["qa_pairs"] if i.get("issue_number") == issue_number), None)
        if not issue:
            continue
        
        # Process issue
        processed = retrieval.process_issue(issue)
        
        # Get top results
        results = retrieval.find_relevant_code_for_issue(issue_number, k=3)
        
        # Create summary
        summary = {
            "issue_number": issue_number,
            "title": issue["issue_title"],
            "type": processed["issue_type"],
            "entities_mentioned": {
                "functions": len(processed["code_context"]["functions"]),
                "classes": len(processed["code_context"]["classes"]),
                "files": len(processed["code_context"]["files"]),
                "code_blocks": len(processed["code_context"]["code_blocks"])
            },
            "top_results": [
                {
                    "name": r.get("name", r.get("path", "Unknown")),
                    "type": r.get("entity_type"),
                    "similarity": r.get("similarity", 0)
                }
                for r in results
            ]
        }
        
        issue_summaries.append(summary)
    
    # Print comparison
    for summary in issue_summaries:
        print(f"\nISSUE #{summary['issue_number']}: {summary['title']}")
        print(f"Type: {summary['type']}")
        
        # Print entity mentions
        print("Entities mentioned:")
        for entity_type, count in summary["entities_mentioned"].items():
            print(f"  - {entity_type}: {count}")
        
        # Print top results
        print("Top matching code elements:")
        for i, result in enumerate(summary["top_results"]):
            print(f"  {i+1}. {result['type']}: {result['name']} (similarity: {result['similarity']:.4f})")
        
        print("-" * 60)

def main():
    """Run all examples."""
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "dark_mode":
            example_dark_mode_issue()
        elif example == "search_bar":
            example_search_bar_issue()
        elif example == "compare":
            compare_multiple_issues()
        else:
            print(f"Unknown example: {example}")
            print("Available examples: dark_mode, search_bar, compare")
    else:
        # Run specific example
        example_dark_mode_issue()

if __name__ == "__main__":
    main()