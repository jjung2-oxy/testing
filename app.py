from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from promptflow.client import PFClient

app = FastAPI()

# Initialize the Prompt Flow client
client = PFClient()

class Question(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_healthcare(question: Question):
    try:
        # Assuming your flow is in the current directory and named 'flow.dag.yaml'
        result = client.test(flow=".", inputs={"question": question.text})
        return {"response": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)