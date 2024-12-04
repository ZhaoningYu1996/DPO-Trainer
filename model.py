from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str, device: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the supervised fine-tuned language model and tokenizer from huggingface.

    Args:
        model_name: The name or path of the sft model.
        device: The device to load the model onto.

    Returns:
        model: The loaded language model.
        tokenizer: The corresponding tokenizer.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure the tokenizer has a padding token (important for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the pre-trained language model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move the model to the specified device
    model.to(device)
    
    return model, tokenizer
