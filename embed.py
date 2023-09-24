import torch
from torch.nn import CosineSimilarity
from transformers import BertModel, BertTokenizer

# Load models and tokenizer
base_model = BertModel.from_pretrained("bert-base-uncased")
fine_tuned_model = BertModel.from_pretrained("fine_tuned_bert_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Ensure both models are in evaluation mode
base_model.eval()
fine_tuned_model.eval()

# Ensure models are on the right device (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = base_model.to(device)
fine_tuned_model = fine_tuned_model.to(device)


def get_embedding(text, model):
    """Get the BERT embeddings for a given text using the specified model."""
    inputs = tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True, padding="max_length"
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :]


def evaluate_similarity(text, query, model):
    """Evaluate similarity between a text and a query using cosine similarity and the specified model."""
    text_embedding = get_embedding(text, model)
    query_embedding = get_embedding(query, model)

    cosine_similarity = CosineSimilarity(dim=1)
    similarity = cosine_similarity(text_embedding, query_embedding)
    return similarity.item()


# Example
text = "Good"
query = "Bad"

base_similarity_score = evaluate_similarity(text, query, base_model)
fine_tuned_similarity_score = evaluate_similarity(text, query, fine_tuned_model)

print(f"Base BERT similarity score: {base_similarity_score:.4f}")
print(f"Fine-tuned BERT similarity score: {fine_tuned_similarity_score:.4f}")
