from .cot import CoTModel
from .data import Dataset
import json
from pathlib import Path
import logging

def generate_dataset(output_json: str, oversample: int = 20, temperature: float = 0.8):
    #initialize model and dataset
    model = CoTModel()
    trainset = Dataset("train")
    dataset = []
    
    print(f"Starting dataset generation with oversample={oversample}, temperature={temperature}")
    print(f"Processing {len(trainset)} questions from training set")

    # Process each question in the training set
    for i, (question, true_answer) in enumerate(trainset):
        print(f"\nProcessing question {i+1}/{len(trainset)}: {question}")
        print(f"True answer: {true_answer}")
        
        # Generate multiple completions with temperature
        completions = model.batched_generate(
            [question],  # Needs to be a list for batched_generate
            num_return_sequences=oversample,
            temperature=temperature
        )
        
        # Handle both possible return types from batched_generate
        if isinstance(completions, list) and len(completions) > 0:
            if isinstance(completions[0], list):
                completions = completions[0]  # Get the list of completions for this question
        else:
            print(f"Warning: Unexpected completions format: {completions}")
            continue
            
        print(f"Generated {len(completions)} completions")
        
        # Check each completion for correct answer
        found_valid = False
        for j, completion in enumerate(completions):
            if not isinstance(completion, str):
                print(f"Warning: Completion {j+1} is not a string: {completion}")
                continue
                
            try:
                parsed_answer = model.parse_answer(completion)
                if not isinstance(parsed_answer, (int, float)):
                    print(f"Warning: Parsed answer is not a number: {parsed_answer}")
                    continue
                    
                print(f"Completion {j+1}: parsed answer = {parsed_answer}")
                
                #check if answer matches (within small tolerance)
                if abs(parsed_answer - float(true_answer)) < 1e-6:
                    print(f"Found correct answer! Adding to dataset.")
                    #store in format shown in README example
                    dataset.append([
                        question,
                        float(true_answer),
                        completion.strip()
                    ])
                    found_valid = True
                    break  # Found a correct answer, move to next question
            except (ValueError, IndexError, AttributeError) as e:
                print(f"Completion {j+1}: Failed to parse answer - {str(e)}")
                continue  # Skip bad outputs
        
        if not found_valid:
            print(f"Warning: No valid completion found for question {i+1}")
    
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
