"""finetune.py"""

import torch
from torch.nn import CosineSimilarity, MSELoss
from torch.optim import Adam
from transformers import BertModel, BertTokenizer

from dataset import load_google_data

# Hyperparameters
EPOCHS = 3
LR = 1e-5
BATCH_SIZE = 32

# Load data
train_data = load_google_data("train")
valid_data = load_google_data("validation")

# Only use training data if score > 0.8
train_data = train_data[train_data["score"] > 0.8]
print(f"Training data size: {len(train_data)}")

# Initialize BERT model and tokenizer
model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
model = model.to(device)


# Tokenize data
train_encodings = tokenizer(
    list(train_data["anchor"]),
    list(train_data["target"]),
    truncation=True,
    padding=True,
    return_tensors="pt",
    max_length=128,
)
valid_encodings = tokenizer(
    list(valid_data["anchor"]),
    list(valid_data["target"]),
    truncation=True,
    padding=True,
    return_tensors="pt",
    max_length=128,
)

# Initialize optimizer, similarity measure, and loss function
optimizer = Adam(model.parameters(), lr=LR)
cosine_similarity = CosineSimilarity(dim=1)
loss_fn = MSELoss()

# Training loop
print("Fine-tuning BERT...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    # We'll use batches to train the model
    num_batches = len(train_data) // BATCH_SIZE
    for batch in range(num_batches):
        optimizer.zero_grad()

        # Get embeddings for anchor and target for the current batch
        start_idx = batch * BATCH_SIZE
        end_idx = (batch + 1) * BATCH_SIZE

        # 3. Move the data to the specified device
        input_ids = train_encodings["input_ids"][start_idx:end_idx].to(device)
        attention_mask = train_encodings["attention_mask"][start_idx:end_idx].to(device)

        input_ids_val = valid_encodings["input_ids"][start_idx:end_idx].to(device)
        attention_mask_val = valid_encodings["attention_mask"][start_idx:end_idx].to(
            device
        )

        anchor_embeddings = model(input_ids, attention_mask).last_hidden_state[:, 0, :]
        target_embeddings = model(input_ids_val, attention_mask_val).last_hidden_state[
            :, 0, :
        ]

        # Compute cosine similarity
        similarities = cosine_similarity(anchor_embeddings, target_embeddings)

        # 4. Move the target scores to the device
        scores = (
            torch.tensor(train_data["score"][start_idx:end_idx].values)
            .float()
            .to(device)
        )

        # Compute loss
        loss = loss_fn(similarities, scores)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    # Print average loss for the epoch
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / num_batches}")

    # Can add a validation loop here to evaluate the model on the validation set after each epoch

# Save the fine-tuned model
model.save_pretrained("fine_tuned_bert_model")

print("Fine-tuning completed!")
