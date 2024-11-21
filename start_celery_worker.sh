# start_celery_worker.sh

export PYTHONPATH=.
celery -A celery_app worker --loglevel=info --pool=threads --concurrency=4
