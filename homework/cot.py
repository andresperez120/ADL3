from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
            {"role": "system", "content": "You are a helpful assistant that performs unit conversions. Be concise. Write out the steps clearly in the form of <answer>NUMBER</answer>."},
            {"role": "user", "content": "How many centimeters are in 5 meters?"},
            {"role": "assistant", "content": "One meter is equal to 100 centimeters, so 5 times 100 is equal to <answer>500</answer>"},
            {"role": "user", "content": "How many grams are in 3 kilograms?"},
            {"role": "assistant", "content": "One kilogram is equal to 1000 grams, so 3 times 1000 is equal to <answer>3000</answer>"},
            {"role": "user", "content": "How many milliliters are in 2 liters?"},
            {"role": "assistant", "content": "One liter is equal to 1000 milliliters, so 2 times 1000 is equal to <answer>2000</answer>"},
            {"role": "user", "content": question}
        ]
        
        #make template since the model doesn't have a chat template
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"Instructions: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"Question: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Answer: {msg['content']}\n\n"
        
        return prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
