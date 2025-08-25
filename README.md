# 🎧 Virtual Customer Care Agent 🤖

### An intelligent virtual agent that assists customers with policy-related questions and seamlessly transfers to human agents when needed. Built with LangChain, FAISS, and HuggingFace models, it provides a natural customer service experience with conversation memory.

### 🔍 Key Features

  - **Smart Human Handoff**: Transfers to human agents when unable to answer from policy documents
  - **Conversation Memory**: Maintains context of the last 2 exchanges for natural conversations
  - **Multi-session Support**: Handles multiple concurrent customer conversations with unique session IDs
  - **Warm, Conversational Tone**: Responds like a real customer service representative
  - **Policy Knowledge Base**: Leverages company policies to provide accurate information
  - **REST API**: Simple FastAPI endpoints for easy integration with any customer service platform
  - **Docker Ready**: Optimized container with persistent knowledge base

### 🚀 Quick Start (Local)

  1. **Install dependencies**
       ```bash
       pip install -r requirements.txt
       ```
  
  2. **Add your policy documents**
       ```bash
       # Create policy_store directory if it doesn't exist
       mkdir -p policy_store
       # Place all policy PDF files in this directory
       cp your-policies/*.pdf policy_store/
       ```
  
  3. **Create a .env file**
      ```bash
      # Include your Hugging Face token
      echo "HF_TOKEN=your_huggingface_token_here" > .env
      ```
  
  4. **Generate embeddings**
      ```bash
      python emedding.py
      ```
  
  5. **Start the API server**
      ```bash
      uvicorn chat:app --host 0.0.0.0 --port 8000 --reload
      ```

### 🐳 Docker Deployment

  1. **Build and start the container**
     ```bash
     docker compose up --build
     ```

  2. **Access the API**
     ```
     The API will be available at http://localhost:8000
     ```

### 🏗️ Architecture Diagram
  ![Architecture Diagram](https://github.com/user-attachments/assets/c90b7e04-fdc9-4be5-9840-f96bc8280f6d)

### 📡 API Endpoints

- **POST /chat**
  ```json
  {
    "question": "How do I request a refund under the 30-day policy?",
    "session_id": "optional-session-id"
  }
  ```
  Provides customer assistance with automatic human handoff when needed

- **GET /sessions/{session_id}**  
  Retrieves customer conversation history for a specific support session

- **DELETE /sessions/{session_id}**  
  Clears a customer support conversation

- **GET /health**  
  Service health check endpoint

### ✅ Future Enhancements

 - Add customer authentication and ticket creation
 - Implement persistent conversation storage (e.g., Redis)
 - Create a customer-facing chat widget interface
 - Support for additional knowledge base formats
 - Integration with CRM and ticketing systems
 - Analytics dashboard for customer service metrics
