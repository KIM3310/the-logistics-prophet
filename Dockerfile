FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8501

CMD ["/bin/sh", "-lc", "python scripts/run_pipeline.py && python -m streamlit run app/dashboard.py --server.port=8501 --server.address=0.0.0.0"]
