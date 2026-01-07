from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print(f"Tokenizer model_max_length: {tokenizer.model_max_length}")

# Create a long text
long_text = "test " * 1000
print(f"Text length: {len(long_text)}")

# Try to tokenize with the same parameters as in the code
try:
    encoded = tokenizer(long_text, return_tensors="pt", truncation=True, max_length=512)
    print(f"Encoded shape: {encoded['input_ids'].shape}")
    
    if encoded['input_ids'].shape[1] > 514:
        print("FAIL: Tokenizer did not truncate correctly!")
    else:
        print("SUCCESS: Tokenizer truncated correctly.")
        
    # Try forward pass
    output = model(**encoded)
    print("Forward pass successful.")
    
except Exception as e:
    print(f"Error: {e}")
