web: uvicorn app_api:app --host 0.0.0.0 --port 8001
worker: PYTHONPATH=. celery -A celery_app worker --loglevel=info --pool=threads --concurrency=4
redis: redis-server
