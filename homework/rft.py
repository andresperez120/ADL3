from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    import json
    from pathlib import Path
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments, Trainer
    from .sft import TokenizedDataset, format_example
    from .base_llm import BaseLLM

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    #load RFT dataset
    with open("data/rft.json", "r") as f:
        rft_data = json.load(f)
    
    llm = BaseLLM()
    
    lora_config = LoraConfig(
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
        r=16,  # Doubled from SFT for better performance
        lora_alpha=64  # 4x the rank as recommended
    )
    
    model = get_peft_model(llm.model, lora_config)
    model.train()
    model.enable_input_require_grads()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        gradient_checkpointing=True,
        logging_dir=output_dir,
        learning_rate=1e-4,
        warmup_steps=100,
        save_steps=500,
        eval_steps=100,
        save_total_limit=1,
        weight_decay=0.01
    )
    
    #create dataset class for RFT data
    class RFTDataset(TokenizedDataset):
        def __init__(self, tokenizer, data):
            self.tokenizer = tokenizer
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            question, _, completion = self.data[idx]
            #use the completion (which includes reasoning) as the target
            return format_example(question, completion)
    
    train_dataset = RFTDataset(llm.tokenizer, rft_data)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Train and save
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
