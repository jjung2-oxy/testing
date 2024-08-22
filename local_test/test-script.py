import json
import os
from preprocess import process_documents
from retrieve_info import get_relevant_info
from nodes.analyze_sentiment import get_sentiments
from transformers import pipeline
from openai import AzureOpenAI
from dotenv import load_dotenv
from docx import Document

# Load environment variables
load_dotenv()

# Set up Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def simulate_chat(system_message, user_message, relevant_info, sentiments):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"""
            Context:
            {relevant_info}

            Sentiment Analysis:
            {json.dumps(sentiments)}

            User Question: {user_message}

            Please provide a comprehensive response that addresses the question in detail. Include specific quotes from the provided context to support your analysis. Organize your response into clear sections covering different aspects of the question. Aim for a thorough analysis that captures the nuances and complexities of the healthcare industry based on the interview data.
            """}
                ]
    
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=messages,
        max_tokens=2048,  
        temperature=0.7,
    )
    
    return response.choices[0].message.content

def read_interviews(folder_path):
    documents = []
    print(f"Debug: Attempting to read from folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"Debug: Error - Folder does not exist: {folder_path}")
        return documents
    
    if not os.path.isdir(folder_path):
        print(f"Debug: Error - Path is not a directory: {folder_path}")
        return documents
    
    files = os.listdir(folder_path)
    print(f"Debug: Total files in directory: {len(files)}")
    
    docx_files = [f for f in files if f.endswith('.docx')]
    print(f"Debug: Total .docx files in directory: {len(docx_files)}")
    
    for filename in docx_files:
        file_path = os.path.join(folder_path, filename)
        try:
            doc = Document(file_path)
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            documents.append(content)
            print(f"Debug: Successfully read file {filename}, length: {len(content)} characters")
        except Exception as e:
            print(f"Debug: Error reading file {filename}: {str(e)}")
    
    print(f"Debug: Total documents successfully read: {len(documents)}")
    return documents

def main():
    # Use an absolute path to the interviews folder
    folder_path = '/Users/jordanjung/Desktop/HA/interviews'  # Replace with the actual path to your folder
    print("Reading interview documents...")
    documents = read_interviews(folder_path)
    
    if not documents:
        print("Error: No documents were read. Please check the folder path and ensure it contains .docx files.")
        return

    # Preprocess documents
    print("Preprocessing documents...")
    processed_docs = process_documents(documents)
    
    # Sample question related to Health Advances interviews
    question = "What are the key challenges and opportunities in the healthcare industry according to the interviews?"
    
    # Retrieve relevant information
    print("Retrieving relevant information...")
    relevant_info = get_relevant_info(processed_docs, question)
    
    # Perform sentiment analysis
    print("Analyzing sentiment...")
    sentiments = get_sentiments(relevant_info)
    
    # Simulate chat
    print("Generating response...")
    system_message = """
        You are an expert analyst in the healthcare industry with a focus on oncology and advanced medical technologies. Your task is to provide comprehensive and accurate insights based on the given interview data. Use industry-specific terms where appropriate and provide a balanced view of the information presented.

        When responding:
        1. Use specific quotes from the provided context to support your analysis.
        2. Organize your response into clear sections covering different aspects of the question.
        3. Provide detailed explanations and interpretations of the quoted material.
        4. Discuss both challenges and opportunities in the healthcare industry.
        5. Consider the implications of the information for various stakeholders (e.g., patients, healthcare providers, technology companies).
        6. If there are conflicting viewpoints in the data, acknowledge and analyze them.
        7. Aim for a thorough analysis that captures the nuances and complexities of the healthcare industry based on the interview data.
        """
    response = simulate_chat(system_message, question, "\n".join(relevant_info), sentiments)
    
    print("\nQuestion:", question)
    print("\nRelevant Information:")
    if relevant_info:
        for info in relevant_info:
            print(f"- {info}")
    else:
        print("No relevant information found.")
    
    print("\nSentiment Analysis:")
    if sentiments:
        for sent in sentiments:
            print(f"- Label: {sent['label']}, Score: {sent['score']:.4f}")
    else:
        print("No sentiment analysis results available.")
    
    print("\nGenerated Response:")
    print(response)

if __name__ == "__main__":
    main()
