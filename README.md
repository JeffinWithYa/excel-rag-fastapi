# excel-rag-fastapi

docker build -t insurance-rag .
docker run -p 8000:8000 --env-file .env insurance-rag