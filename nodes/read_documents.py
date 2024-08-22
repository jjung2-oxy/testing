import os
from docx import Document
from promptflow import tool

@tool
def read_documents(folder_path: str) -> list:
    documents = []
    print(f"Attempting to read documents from: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"Error: Folder path does not exist: {folder_path}")
        return documents
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):
            file_path = os.path.join(folder_path, filename)
            try:
                doc = Document(file_path)
                content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                documents.append(content)
                print(f"Successfully read: {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
    
    print(f"Total documents read: {len(documents)}")
    return documents