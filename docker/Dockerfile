# ---------------------------
# 1. Base Image
# ---------------------------
FROM python:3.10-slim

# ---------------------------
# 2. Set working directory
# ---------------------------
WORKDIR /app

# ---------------------------
# 3. Copy project files
# ---------------------------
COPY api/ ./api/
COPY models/ ./models/
COPY src/ ./src/
COPY data/ ./data/

# ---------------------------
# 4. Install dependencies
# ---------------------------
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------
# 5. Expose port for API
# ---------------------------
EXPOSE 8000

# ---------------------------
# 6. Run FastAPI app
# ---------------------------
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

