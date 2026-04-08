# Use the official, lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the "box"
WORKDIR /app

# Copy the rest of our SmartMess code into the "box"
COPY . .

# Install dependencies using pyproject.toml
RUN pip install --no-cache-dir .

# Set up the huggingface user (UID 1000)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

# Set the default environment variables (Hackers will override these!)
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o"
ENV HF_TOKEN=""

USER user

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
