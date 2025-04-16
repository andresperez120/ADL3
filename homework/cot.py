from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
            {"role": "system", "content": "You are a helpful assistant that performs unit conversions with high precision. Always follow these rules:\n1. Show your work step by step\n2. Use the exact format: 'calculation = <answer>NUMBER</answer>'\n3. Be precise with calculations\n4. Include conversion factors\n5. Always output the final answer in the <answer>NUMBER</answer> format"},
            {"role": "user", "content": "How many centimeters are in 5 meters?"},
            {"role": "assistant", "content": "1 meter = 100 centimeters\n5 meters × 100 = <answer>500</answer>"},
            {"role": "user", "content": "Express 2 centuries as weeks."},
            {"role": "assistant", "content": "1 century = 100 years\n1 year = 52.17745696875 weeks (accounting for leap years)\n2 centuries × 100 years/century × 52.17745696875 weeks/year = <answer>10435.491393749998</answer>"},
            {"role": "user", "content": "Convert 3 kilometers per hour to meters per second."},
            {"role": "assistant", "content": "3 km/h × (1000 m/1 km) × (1 h/3600 s) = <answer>0.8333333333333334</answer>"},
            {"role": "user", "content": question}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )


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
