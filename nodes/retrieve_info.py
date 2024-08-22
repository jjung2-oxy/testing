import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from promptflow import tool

def get_question_embedding(question):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    
    inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

@tool
def retrieve_relevant_info(processed_docs: str, question: str, top_k: int = 15) -> list:
    processed_docs = json.loads(processed_docs)
    question_embedding = get_question_embedding(question)
    
    relevant_sentences = []
    for doc_index, doc in enumerate(processed_docs):
        for sent_index, (sentence, embedding) in enumerate(zip(doc['sentences'], doc['embeddings'])):
            similarity = cosine_similarity([question_embedding], [np.array(embedding)])[0][0]
            relevant_sentences.append((sentence, similarity, doc_index, sent_index))
    
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = relevant_sentences[:top_k]
    
    result = [f"Document {doc_index + 1}: {sentence}" for sentence, _, doc_index, _ in top_sentences]
    return result