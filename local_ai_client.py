import json
import typing
import requests
from graphiti_core.prompts import Message
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig


def extract_json_from_string(input_string: str) -> str:
    start_index = input_string.find("{")
    end_index = input_string.rfind("}")
    if start_index == -1 or end_index == -1 or start_index > end_index:
        return "{}"
    return input_string[start_index : end_index + 1]


class LocalAiClient(LLMClient):
    def __init__(
        self, config: LLMConfig | None = None, grammar_file: str | None = None
    ):
        if config is None:
            config = LLMConfig()

        super().__init__(config)
        self.base_url = config.base_url
        self.grammar_content = ""

        if grammar_file:
            with open(grammar_file, "r") as f:
                self.grammar_content = f.read()

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: typing.Optional[typing.Type[typing.Any]] = None,
        max_tokens: typing.Optional[int] = None,
        model_size: typing.Optional[int] = None,
    ) -> typing.Any:
        """Implement the abstract method from LLMClient"""
        response_json_str = self.execute_llm_query(messages)
        print(f"Raw response JSON string:\n{response_json_str}")

        try:
            llm_response_json = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            raise e

        print(f"Parsed LLM response JSON:\n{json.dumps(llm_response_json, indent=2)}")
        return llm_response_json

    def execute_llm_query(self, messages: list[Message]) -> str:
        # Convert Message objects to dicts Ollama expects
        messages_payload = [{"role": m.role, "content": m.content} for m in messages]
        print(f"Sending messages payload:\n{messages_payload}")

        request_body = {
            "model": "mistral",
            "messages": messages_payload,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens or 2000,
        }

        if self.grammar_content:
            request_body["grammar"] = self.grammar_content

        response = requests.post(self.base_url, json=request_body)
        response.raise_for_status()

        print("Raw response text from Ollama Mistral API:")
        print(response.text)

        resp_json = response.json()
        # Extract the content from the first choice
        response_message = (
            resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        print(f"Response message content:\n{response_message}")

        # Extract JSON substring from the response text
        extracted_json_str = extract_json_from_string(response_message)
        return extracted_json_str
