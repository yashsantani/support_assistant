FROM python:3.12-slim

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy policy documents and embedding script
COPY policy_store/ /app/policy_store/
COPY emedding.py /app/emedding.py

# Create vector_store directory for embeddings (mount volume here)
RUN mkdir -p /app/vector_store

# Build embeddings only if policy_store changed
RUN python emedding.py

# Copy application files
COPY chat.py /app/chat.py

# Expose the application port
EXPOSE 8000

# Start the application directly
CMD ["uvicorn", "chat:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
