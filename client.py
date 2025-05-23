import os
import time
import json
from typing import List, Dict, Generator, Tuple
import requests
import argparse

class LLMClient:
    """Simple client for interacting with LocalAI's API with streaming support"""

    def __init__(self, api_base: str = "http://localhost:8080/v1"):
        self.api_base = api_base
        self.headers = {"Content-Type": "application/json"}

    def list_models(self) -> List[Dict]:
        """List available models"""
        response = requests.get(f"{self.api_base}/models")
        return response.json()

    def _format_phi_messages(self, messages: List[Dict]) -> str:
        """Format messages for Phi model's expected input format"""
        formatted = ""

        # Add system message if present
        system_msg = next((msg for msg in messages if msg["role"] == "system"), None)
        if system_msg:
            formatted += f"<|system|>{system_msg['content']}<|end|>"

        # Add user messages
        user_msgs = [msg for msg in messages if msg["role"] == "user"]
        if user_msgs:
            formatted += f"<|user|>{user_msgs[-1]['content']}<|end|><|assistant|>"

        return formatted

    def chat_stream(
        self,
        messages: List[Dict],
        model: str = "phi-3.5-mini-instruct",
    ) -> Generator[Tuple[str, float], None, None]:
        """Send a streaming chat completion request"""

        formatted_prompt = self._format_phi_messages(messages)

        # Update the request data to use the formatted prompt
        data = {
            "prompt": formatted_prompt,  # Use formatted prompt instead of messages
            "model": model,
            "stream": True,
            "top_p": 0.1,
            "temperature": 0.3,
            "stop": ["<|endoftext|>", "<|end|>"],  # Add <|end|> to stop tokens
            "max_tokens": 1024,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }

        start_time = time.time()

        with requests.post(
            f"{self.api_base}/completions",  # Change to completions endpoint
            headers=self.headers,
            json=data,
            stream=True,
        ) as response:
            if response.status_code != 200:
                raise Exception(f"Error: {response.text}")

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]

                    if line == "[DONE]":
                        continue

                    try:
                        chunk = json.loads(line)
                        if chunk.get("choices") and chunk["choices"][0].get(
                            "text"
                        ):  # Changed from delta to text
                            content = chunk["choices"][0]["text"]
                            current_time = time.time() - start_time
                            yield content, current_time
                    except json.JSONDecodeError:
                        continue



def demonstrate_capabilities():
    """Show basic capabilities of the LLM with streaming responses
    
    Example usage:

    ```
    python client.py -t "My Example" -c "This is an example of a data mart in SQL." "It has two tables: fact and dimension."
    ```

    This command will:
    1. Set the title of the example to "My Example"
    2. Create a user message with the content:
    "This is an example of a data mart in SQL. It has two tables: fact and dimension."
    3. Demonstrate streaming the LLM response for this prompt
    """

    llm = LLMClient(os.getenv("LLM_API_BASE", "http://localhost:8081/v1"))

    print("\nAvailable Models:")
    try:
        models = llm.list_models()
        print(json.dumps(models, indent=2))
    except Exception as e:
        print(f"Error: {e}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Add a new example')
    parser.add_argument('-t', '--title', type=str, help='Title of the example')
    parser.add_argument('-c', '--content', type=str, nargs='+', help='Content of the example')
    args = parser.parse_args()

    # Create examples list from the command line when running the script
    examples = []
    if args.title and args.content:
        examples.append({
            "title": args.title,
            "messages": [
                {
                    "role": "user",
                    "content": " ".join(args.content),
                },
            ],
        })
    else:
        # This should never be reached now since we set defaults above
        examples = [
            {
                "title": "AI Explanation",
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain what AI is in two sentences.",
                    },
                ],
            },
        ]

    for example in examples:
        print(f"\nExample: {example['title']}")
        print("Response:")

        try:
            total_time = 0
            total_response = ""
            for token, response_time in llm.chat_stream(example["messages"]):
                end_char = "\n" if token == " " else ""
                print(token, end=end_char, flush=True)
                total_time = response_time
                total_response += token
            print(f"\nTotal time: {total_time:.2f}s")
            print(f"\nTotal response: {total_response}")

        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    print("🤖 LocalAI Streaming Client Demo")
    print("Testing connection to LocalAI and demonstrating basic capabilities...")
    demonstrate_capabilities()
