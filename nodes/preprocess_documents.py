import json
import numpy as np
import nltk
from transformers import AutoTokenizer, AutoModel
import torch
from promptflow import tool

nltk.download('punkt', quiet=True)

def create_embeddings(sentences):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding.tolist())  # Convert numpy array to list
    return embeddings

@tool
def preprocess_documents(documents: list) -> str:
    processed_docs = []
    for doc in documents:
        sentences = nltk.sent_tokenize(doc)
        sentence_embeddings = create_embeddings(sentences)
        processed_docs.append({
            'sentences': sentences,
            'embeddings': sentence_embeddings
        })
    return json.dumps(processed_docs)