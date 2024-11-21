from celery import Celery
# Initialize Celery
celery_app = Celery(
    'celery_app',
    broker='redis://localhost:6379/0',  # Update if using a different broker
    backend='redis://localhost:6379/0',  # Update if using a different backend
    include=['app_api']  # Directly include the 'app_api' module
)

# Load task modules from all registered Django app configs.
celery_app.autodiscover_tasks(['app_api'])
