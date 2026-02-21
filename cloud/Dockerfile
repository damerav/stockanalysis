# Combined: relay + dashboards in one container
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn requests streamlit plotly pandas

COPY src/ /app/src/

EXPOSE 8000 8501 8502

COPY cloud/start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
