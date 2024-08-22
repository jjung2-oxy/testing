from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def get_question_embedding(question):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    
    inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def retrieve_relevant_info(processed_docs, question, top_k=15):  # Increased from 5 to 15
    question_embedding = get_question_embedding(question)
    
    relevant_sentences = []
    for doc_index, doc in enumerate(processed_docs):
        for sent_index, (sentence, embedding) in enumerate(zip(doc['sentences'], doc['embeddings'])):
            similarity = cosine_similarity([question_embedding], [embedding])[0][0]
            relevant_sentences.append((sentence, similarity, doc_index, sent_index))
    
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = relevant_sentences[:top_k]
    
    # Group sentences by document and maintain original order
    grouped_sentences = {}
    for sentence, _, doc_index, sent_index in top_sentences:
        if doc_index not in grouped_sentences:
            grouped_sentences[doc_index] = []
        grouped_sentences[doc_index].append((sent_index, sentence))
    
    # Sort sentences within each document by their original order
    result = []
    for doc_index in sorted(grouped_sentences.keys()):
        sorted_sentences = sorted(grouped_sentences[doc_index], key=lambda x: x[0])
        result.extend([f"Document {doc_index + 1}: {sentence}" for _, sentence in sorted_sentences])
    
    return result

# Node function
def get_relevant_info(processed_docs, question):
    return retrieve_relevant_info(processed_docs, question)