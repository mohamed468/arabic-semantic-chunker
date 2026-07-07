import random
import json
from locust import HttpUser, task, between
from transformers import AutoTokenizer
from faker import Faker

fake = Faker('ar')
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct")

system_message = "You are a professional multilingual NLP data parser.\nSplit the provided text into meaningful, self-contained semantic chunks.\nFollow the Task and Output Schema to generate the Output JSON.\nDo not add any introduction or conclusion.\nPreserve the original text exactly and respect its language — do not summarize, translate, or paraphrase."
schema_str = '{"properties": {"original_text_length": {"description": "Total number of characters in the original text.", "title": "Original Text Length", "type": "integer"}, "semantic_chunks": {"description": "List of sequential chunks from the original text. Each chunk is self-contained and preserves the original language(s).", "items": {"type": "string"}, "minItems": 2, "title": "Semantic Chunks", "type": "array"}}, "required": ["original_text_length", "semantic_chunks"], "title": "SemanticChunking", "type": "object"}'

class ChunkingLoadTest(HttpUser):
    wait_time = between(1, 3)

    @task
    def post_completion(self):
        model_id = "arabic-chunker"
        raw_text = fake.text(max_nb_chars=random.randint(400, 600))

        chunking_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"# Input Text:\n{raw_text}\n\n# Task:\nSplit the provided text into meaningful, self-contained semantic chunks. Do NOT summarize or leave out words.\n\n# Output Schema:\n{schema_str}\n\n# Output JSON:\n```json\n"}
        ]

        prompt = tokenizer.apply_chat_template(
            chunking_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        message = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.1
        }

        llm_response = self.client.post("/v1/completions", json=message)

        if llm_response.status_code == 200:
            with open("./vllm_tokens.txt", "a", encoding="utf-8") as dest:
                dest.write(json.dumps({
                    "prompt": prompt,
                    "response": llm_response.json()["choices"][0]["text"],
                }, ensure_ascii=False) + "\n")
