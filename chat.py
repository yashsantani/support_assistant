import os
import uuid
import datetime
from typing import Dict, List, Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Company Policy Chatbot", description="AI-powered chatbot for company policy inquiries")

# In-memory session storage (use Redis/database for production)
sessions: Dict[str, List[Dict]] = {}

HF_TOKEN = os.getenv("HF_TOKEN")
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_DB_PATH = "vector_store/faiss_index"

def get_hf_llm(HF_LLM_MODEL: str):
    """Get Hugging Face LLM model."""
    llm = HuggingFaceEndpoint(
        repo_id=HF_LLM_MODEL,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm)

PROMPT = """
You are a virtual customer care agent for our company. Your primary role is to assist customers by referencing our company policy documents.

Company Policy Context: {context}

{question}

Instructions:
1. If you can answer the query using the policy documents, provide a friendly and helpful response.
2. If you cannot answer based on available policies, respond with: "I understand your question, but I'll need to transfer you to a human agent who can better assist with this specific inquiry. Please hold while I connect you with one of our customer care specialists."
3. Use a warm, conversational tone like a helpful customer service representative.
4. Only provide information that's explicitly mentioned in the policy documents.
5. Address the customer respectfully and thank them for their patience.

Your response:
"""

def setup_prompt(prompt: str = PROMPT) -> PromptTemplate:
    """Set up the prompt template."""
    return PromptTemplate(
        template=prompt,
        input_variables=["context", "question"]
    )

# load vector store
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# since the db is created by us we can allow dangerous deserialization
db = FAISS.load_local(FAISS_DB_PATH, embedding_model, allow_dangerous_deserialization=True)


# create the qa chain
qa_chain = RetrievalQA.from_chain_type(
    llm=get_hf_llm(HF_LLM_MODEL),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": setup_prompt(PROMPT)},
)

# user_question = input("Enter your question: ")
# result = qa_chain.invoke({'query': user_question})
# print("RESULT: ", result['result'])
# print("SOURCE DOCUMENTS: ", result['source_documents'])

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    ai_message: str
    session_id: str
    source_documents: List[str]

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Chat endpoint with multi-conversation support"""
    # Generate session ID if not provided
    if not request.session_id:
        request.session_id = str(uuid.uuid4())
    
    # Initialize session if new
    if request.session_id not in sessions:
        sessions[request.session_id] = []
        is_new_session = True
    
    # Format question with chat history context
    enhanced_question = request.question
    if sessions[request.session_id]:
        chat_context = "Previous conversation context:\n"
        for i, exchange in enumerate(sessions[request.session_id][-2:]):  # Only use last 2 exchanges for context
            chat_context += f"Q: {exchange['question']}\n"
            chat_context += f"A: {exchange['answer']}\n\n"
        enhanced_question = f"{chat_context}Current question: {request.question}"
    
    # Get response from QA chain
    result = qa_chain.invoke({'query': enhanced_question})
    
    # Store conversation in session
    sessions[request.session_id].append({
        "question": request.question,
        "answer": result['result'],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Format source documents
    source_docs = [doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content 
                   for doc in result.get('source_documents', [])]
    
    return ChatResponse(
        ai_message=result['result'],
        session_id=request.session_id,
        source_documents=source_docs
    )

@app.get("/sessions/{session_id}")
def get_session_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": sessions[session_id]}

@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    """Clear a conversation session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session cleared successfully"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Company Policy Chatbot"}
