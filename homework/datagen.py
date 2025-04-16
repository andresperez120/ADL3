from .cot import CoTModel
from .data import Dataset
import json
from pathlib import Path

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #initialize model and dataset
    model = CoTModel()
    trainset = Dataset("train")
    dataset = []

    # Process each question in the training set
    for question, true_answer in trainset:
        # Generate multiple completions with temperature
        completions = model.batched_generate(
            [question],  # Needs to be a list for batched_generate
            num_return_sequences=oversample,
            temperature=temperature
        )
        
        if not isinstance(completions, list):
            completions = [completions]
        
        # Check each completion for correct answer
        for completion in completions:
            try:
                parsed_answer = model.parse_answer(completion)
                #check if answer matches (within small tolerance)
                if abs(parsed_answer - float(true_answer)) < 1e-6:
                    #store in format shown in README example
                    dataset.append([
                        question,
                        float(true_answer),
                        completion
                    ])
                    break  # Found a correct answer, move to next question
            except (ValueError, IndexError):
                continue  # Skip bad outputs
    
    # Create output directory if needed
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    
    #save the dataset with readme format
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=2)

def is_valid(output):
    answer = output.get("answer")
    reasoning = output.get("reasoning")
    return evaluate_answer(answer)        

        

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
