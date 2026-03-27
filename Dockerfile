FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the API port
EXPOSE 8000

# Start the FastAPI server via uvicorn
CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "8000"]
