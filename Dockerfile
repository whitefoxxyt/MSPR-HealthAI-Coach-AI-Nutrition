FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY app/ ./app/

# Durcissement (mise en production MSPR3) : execution en utilisateur non-root.
# HF_HOME pointe vers un cache inscriptible : le modele HuggingFace est telecharge au runtime.
ENV HF_HOME=/app/.cache/huggingface
RUN useradd --create-home appuser \
    && mkdir -p "$HF_HOME" \
    && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health', timeout=8).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
