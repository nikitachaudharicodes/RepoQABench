import json
import os
import re
from openai import OpenAI
from collections import defaultdict

class RepoQAExtractor:
    """
    Extract and generate question-answer pairs from GitHub issue comments for RepoQABench.
    """
    
    def __init__(self, input_file_path, openai_api_key=None):
        """
        Initialize with the path to the scraped GitHub issue JSON file.
        
        Args:
            input_file_path (str): Path to the JSON file containing issue data
            openai_api_key (str, optional): OpenAI API key for generating QA pairs
        """
        self.input_file_path = input_file_path
        self.data = None
        self.questions = []
        self.golden_answers = []
        self.questions_generated = []
        self.golden_answers_generated = []
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
    def load_data(self):
        """Load the JSON data from the input file."""
        with open(self.input_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def save_data(self, output_file_path=None):
        """
        Save the extracted and generated Q&A pairs back to the JSON file.
        
        Args:
            output_file_path (str, optional): Path to save the output JSON file. 
                                             If None, updates the input file.
        """
        if output_file_path is None:
            output_file_path = self.input_file_path
            
        # Add Q&A pairs to the data
        self.data['questions'] = self.questions
        self.data['golden_answers'] = self.golden_answers
        self.data['questions_generated'] = self.questions_generated
        self.data['golden_answers_generated'] = self.golden_answers_generated
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
            
        print(f"Saved {len(self.questions)} extracted Q&A pairs and {len(self.questions_generated)} generated Q&A pairs to {output_file_path}")
    
    def is_question(self, text):
        """
        Determine if a comment contains a genuine technical question.
        
        Args:
            text (str): The comment text to analyze
            
        Returns:
            bool: True if it contains a relevant question, False otherwise
        """
        # Ignore common non-technical questions
        if re.search(r'fancy making a pull request|want(ed)? to contribute|can I (pick|take) (up|it|this)|still working on (it|this)|could I help', text, re.IGNORECASE):
            return False
            
        # Check if text contains a question mark
        has_question_mark = '?' in text
        
        # Look for technical question indicators
        technical_indicators = [
            r'how (can|do|would|should|could) i',
            r'where (can|do|would|should|could) i',
            r'what (is|are|should|would|does|do)',
            r'why (is|are|does|do|would|should)',
            r'which (is|are|would|should)',
            r'can (you|i|we|someone)',
            r'could (you|i|we|someone)',
            r'should (i|we|you)',
            r'have you',
            r'(where|how) to',
        ]
        
        # Check for code-related content
        has_code = bool(re.search(r'`[^`]+`|\[here\]\(|\.py|pandas\.|\)\.|_[a-z_]+', text))
        
        # Check for technical indicators
        is_technical = any(re.search(pattern, text.lower()) for pattern in technical_indicators)
        
        # Must have a question mark and be technical in nature
        return has_question_mark and (is_technical or has_code)
    
    def is_good_answer(self, text):
        """
        Determine if a comment contains a good technical answer.
        
        Args:
            text (str): The comment text to analyze
            
        Returns:
            bool: True if it contains a useful answer, False otherwise
        """
        # Ignore very short responses
        if len(text.split()) < 10:
            return False
            
        # Look for technical answer indicators
        technical_indicators = [
            r'`[^`]+`',  # Code snippets
            r'\[.*\]\(.*\)',  # Links, especially to code
            r'\.py',  # Python file references
            r'pandas\.',  # Package references
            r'import ',  # Import statements
            r'function',  # Code discussions
            r'module',  # Module references
            r'according to (the )?docs',  # Documentation references
            r'implementat(ed|ion)',  # Implementation details
            r'from [a-z_.]+ import',  # Import statements
        ]
        
        # Check for at least one technical indicator
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in technical_indicators)
    
    def extract_qa_pairs(self):
        """
        Extract question-answer pairs from the issue comments.
        """
        if not self.data:
            self.load_data()
            
        comments = self.data['issue_comments']
        user_map = defaultdict(list)
        
        # Map comments by user for easier analysis
        for i, comment in enumerate(comments):
            user_map[comment['user']].append((i, comment))
        
        # Look for question-answer patterns in comments
        for i, comment in enumerate(comments):
            if self.is_question(comment['body']):
                question = comment['body']
                answer = None
                
                # Look ahead for answers (prioritize immediate next responses)
                for j in range(i+1, min(i+3, len(comments))):
                    if j < len(comments) and self.is_good_answer(comments[j]['body']):
                        answer = comments[j]['body']
                        break
                
                # If we found a valid Q&A pair, add it
                if answer:
                    self.questions.append(question.strip())
                    self.golden_answers.append(answer.strip())
                    
        # Check for issue description to comment response patterns
        description = self.data['issue_description']
        if '?' in description:
            # Extract paragraphs containing questions
            paragraphs = description.split('\n\n')
            for para in paragraphs:
                if '?' in para and self.is_question(para):
                    # Look for answers in the first few comments
                    for i in range(min(3, len(comments))):
                        if self.is_good_answer(comments[i]['body']):
                            self.questions.append(para.strip())
                            self.golden_answers.append(comments[i]['body'].strip())
                            break
        
        print(f"Extracted {len(self.questions)} question-answer pairs")
    
    def generate_qa_pairs_with_gpt4o(self, num_pairs=5):
        """
        Generate artificial QA pairs using GPT-4o based on issue content.
        
        Args:
            num_pairs (int): Number of pairs to generate
        """
        if not self.data:
            self.load_data()
            
        if not self.openai_api_key:
            print("Warning: No OpenAI API key provided. Skipping QA pair generation.")
            return
            
        # Prepare context for GPT-4o
        repo_name = self.data['repo_name']
        issue_description = self.data['issue_description']
        text_context = self.data['text_context']
        code_context = ""
        
        # Format code context for better clarity
        if 'code_context' in self.data and self.data['code_context']:
            for file in self.data['code_context']:
                code_context += f"--- File: {file['filename']} ---\n"
                code_context += file['content']
                code_context += "\n\n"
        
        # Prepare the prompt
        prompt = f"""You are tasked with generating question-answer pairs for a repository-level QA benchmark dataset.

Repository: {repo_name}
Issue Description: 
{issue_description}

Discussion Context:
{text_context[:3000]}  # Truncate to avoid token limits

Code Context:
{code_context[:2000]}  # Truncate to avoid token limits

Based on the above information, generate {num_pairs} high-quality technical question-answer pairs that someone might ask about this repository issue. Focus on questions that require understanding of:
1. The repository's code structure
2. The specific functionality being discussed
3. Implementation details or technical decisions

The questions should be challenging but answerable based on the given context. 

Format your output as a JSON array of objects with 'question' and 'answer' fields:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
"""

        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=self.openai_api_key)
            
            # Call the OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extract content from response
            content = response.choices[0].message.content
            
            # Extract JSON array from response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
            if json_match:
                try:
                    qa_pairs = json.loads(json_match.group(0))
                    for pair in qa_pairs:
                        if 'question' in pair and 'answer' in pair:
                            self.questions_generated.append(pair['question'])
                            self.golden_answers_generated.append(pair['answer'])
                    print(f"Generated {len(qa_pairs)} QA pairs with GPT-4o")
                except json.JSONDecodeError:
                    print("Error: Could not parse JSON from GPT-4o response")
            else:
                print("Error: Could not find JSON array in GPT-4o response")
                
        except Exception as e:
            print(f"Error generating QA pairs with GPT-4o: {e}")

def process_file(file_path, openai_api_key=None, num_pairs=5):
    """
    Process a single JSON file to extract and generate Q&A pairs.
    
    Args:
        file_path (str): Path to the JSON file
        openai_api_key (str, optional): OpenAI API key
        num_pairs (int): Number of pairs to generate
    """
    extractor = RepoQAExtractor(file_path, openai_api_key)
    extractor.load_data()
    extractor.extract_qa_pairs()
    extractor.generate_qa_pairs_with_gpt4o(num_pairs)
    extractor.save_data()

def process_directory(directory_path, openai_api_key=None, num_pairs=5):
    """
    Process all JSON files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing JSON files
        openai_api_key (str, optional): OpenAI API key
        num_pairs (int): Number of pairs to generate per file
    """
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {file_path}...")
            process_file(file_path, openai_api_key, num_pairs)

def main():
    """Main function to run the extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and generate Q&A pairs from GitHub issue data for RepoQABench')
    parser.add_argument('input', help='JSON file or directory to process')
    parser.add_argument('--api-key', help='OpenAI API key for generating QA pairs')
    parser.add_argument('--num-pairs', type=int, default=5, help='Number of QA pairs to generate per file')
    
    args = parser.parse_args()
    
    # Get API key from args or environment variable
    openai_api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("Warning: No OpenAI API key provided. Set OPENAI_API_KEY environment variable or use --api-key.")
    
    if os.path.isdir(args.input):
        process_directory(args.input, openai_api_key, args.num_pairs)
    else:
        process_file(args.input, openai_api_key, args.num_pairs)

if __name__ == "__main__":
    main()