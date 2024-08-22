import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def preprocess_documents(documents):
    # Initialize the BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    processed_docs = []
    for doc in documents:
        # Tokenize the document into sentences
        sentences = sent_tokenize(doc)
        
        # Create sentence embeddings using BERT
        sentence_embeddings = []
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            sentence_embeddings.append(embedding)
        
        processed_docs.append({
            'text': doc,
            'sentences': sentences,
            'embeddings': sentence_embeddings
        })
    
    return processed_docs

# Node function
def process_documents(documents):
    processed_docs = preprocess_documents(documents)
    print(f"Debug: Processed {len(processed_docs)} documents")
    for i, doc in enumerate(processed_docs):
        print(f"Debug: Document {i} has {len(doc['sentences'])} sentences and {len(doc['embeddings'])} embeddings")
    return processed_docs