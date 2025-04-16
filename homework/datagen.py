from .cot import CoTModel
from .data import Dataset
import json
from pathlib import Path
import logging

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #initialize model and dataset
    model = CoTModel()
    trainset = Dataset("train")
    dataset = []
    
    print(f"Starting dataset generation with oversample={oversample}, temperature={temperature}")
    print(f"Processing {len(trainset)} questions from training set")

    # Process first few questions as a test
    for i, (question, true_answer) in enumerate(trainset):
        if i >= 2:  # Only process first 2 questions for debugging
            break
            
        print(f"\nQuestion {i+1}: {question}")
        print(f"True answer: {true_answer}")
        
        # Get the formatted prompt with chat template
        prompt = model.format_prompt(question)
        print("\nFormatted prompt:")
        print(prompt)
        
        # Generate multiple completions with temperature
        completions = model.batched_generate(
            [prompt],  # Use the formatted prompt instead of just the question
            num_return_sequences=oversample,
            temperature=temperature
        )
        
        # Handle both possible return types from batched_generate
        if isinstance(completions, list) and isinstance(completions[0], list):
            completions = completions[0]
            
        print("\nModel outputs:")
        for j, completion in enumerate(completions):
            print(f"\nCompletion {j+1}:")
            print("---")
            print(completion)
            print("---")
            try:
                parsed = model.parse_answer(completion)
                print(f"Parsed answer: {parsed}")
            except Exception as e:
                print(f"Parse error: {str(e)}")
        
        # Check each completion for correct answer
        for completion in completions:
            try:
                parsed_answer = model.parse_answer(completion)
                if abs(parsed_answer - float(true_answer)) < 1e-6:
                    print(f"Found correct answer! Adding to dataset.")
                    #store in format shown in README example
                    dataset.append([
                        question,
                        float(true_answer),
                        completion.strip()
                    ])
                    break
            except (ValueError, IndexError):
                continue
    
    print(f"\nDataset generation complete. Generated {len(dataset)} examples.")
    
    if len(dataset) == 0:
        print("ERROR: No valid examples were generated!")
        return
    
    # Create output directory if needed
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    
    #save the dataset with readme format
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset saved to {output_json}")

def is_valid(output):
    answer = output.get("answer")
    reasoning = output.get("reasoning")
    return evaluate_answer(answer)        

        

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
