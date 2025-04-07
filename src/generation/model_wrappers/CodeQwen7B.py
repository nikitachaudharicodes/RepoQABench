import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义模板
TEMPLATE = """[Instruction]
This is an open-source repository issue understanding task.

The repository name is {repo_name}.
Here is the related context:
Text Context is the information in the issues that is related to the question.
[Text Context]
{text_context}

Code Context is the information in the repo's code part that is related to the question.
[Code Context]
{code_context}

Answer the following question based on the given information.
[Question]
{question}
[Answer]
"""

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/CodeQwen1.5-7B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")

def extract_info_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    repo_name = data.get("repo_name", "Unknown")
    text_context = data.get("text_context", "No text context available")
    code_context_list = data.get("code_context", [])
    code_context = "\n".join(item.get("content", "") for item in code_context_list)
    questions_generated = data.get("questions_generated", ["No question available"])
    question = questions_generated[0]
    golden_answers_generated = data.get("golden_answers_generated", ["No golden answer available"])
    golden_answer = golden_answers_generated[0]

    prompt = TEMPLATE.format(
        repo_name=repo_name,
        text_context=text_context,
        code_context=code_context,
        question=question
    )
    return repo_name, question, golden_answer, prompt

def generate_response(prompt):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=256,  # 减少生成长度
            num_beams=3,  # 增加生成质量
            do_sample=True,  # 启用采样
            temperature=0.7  # 减少随机性
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Clearing cache and retrying...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return "Error: CUDA out of memory."

def process_json_files(input_dir, output_file):
    results = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            repo_name, question, golden_answer, prompt = extract_info_from_json(file_path)

            # 生成预测
            prediction = generate_response(prompt)

            # 整合信息到item中
            item = {
                "repo_name": repo_name,
                "question": question,
                "prediction": prediction,
                "golden_answer": golden_answer
            }
            results.append(item)
            print(f"Processed {filename}")

            # 每次生成后清理显存
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # 保存所有结果到一个JSON文件
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(results, output, indent=4, ensure_ascii=False)
    print(f"All results saved to {output_file}")

# 指定输入和输出路径
input_directory = "/home/vincentz/directed_study/QA/RepoQABench/repoqabench"
output_filename = "/home/vincentz/directed_study/QA/output/CQ7B_output.json"

process_json_files(input_directory, output_filename)

