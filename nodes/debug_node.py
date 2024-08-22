from promptflow import tool

@tool
def debug_output(read_documents_output, preprocess_documents_output, retrieve_info_output, analyze_sentiment_output):
    return {
        "read_documents_output": read_documents_output[:500] if read_documents_output else None,  # Limit to first 500 chars
        "preprocess_documents_output": preprocess_documents_output[:500] if preprocess_documents_output else None,
        "retrieve_info_output": retrieve_info_output,
        "analyze_sentiment_output": analyze_sentiment_output
    }