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

    # Process questions in batches of 10 for efficiency
    batch_size = 10
    total_correct = 0
    total_processed = 0
    
    for batch_start in range(0, len(trainset), batch_size):
        batch_end = min(batch_start + batch_size, len(trainset))
        batch = list(trainset)[batch_start:batch_end]
        
        for question, true_answer in batch:
            total_processed += 1
            
            # Get the formatted prompt with chat template
            prompt = model.format_prompt(question)
            
            # Generate multiple completions with temperature
            completions = model.batched_generate(
                [prompt],
                num_return_sequences=oversample,
                temperature=temperature
            )
            
            # Handle both possible return types from batched_generate
            if isinstance(completions, list) and isinstance(completions[0], list):
                completions = completions[0]
            
            # Check each completion for correct answer
            found_valid = False
            for completion in completions:
                try:
                    parsed_answer = model.parse_answer(completion)
                    if abs(parsed_answer - float(true_answer)) < 1e-6:
                        dataset.append([
                            question,
                            float(true_answer),
                            completion.strip()
                        ])
                        total_correct += 1
                        found_valid = True
                        break
                except (ValueError, IndexError):
                    continue
            
            # Print progress
            if total_processed % 10 == 0:
                success_rate = (total_correct / total_processed) * 100
                print(f"Processed {total_processed}/{len(trainset)} questions. Success rate: {success_rate:.1f}%")
            
            if not found_valid:
                print(f"\nWarning: No valid completion found for question: {question}")
                print(f"True answer: {true_answer}")
    
    print(f"\nDataset generation complete. Generated {len(dataset)} examples.")
    print(f"Overall success rate: {(total_correct / total_processed) * 100:.1f}%")
    
    if len(dataset) == 0:
        print("ERROR: No valid examples were generated!")
        return
    
    # Create output directory if needed
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the dataset
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
