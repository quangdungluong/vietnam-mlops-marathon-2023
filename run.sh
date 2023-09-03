# gunicorn app:server --workers 9 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:5040
# uvicorn app:api.app --host 0.0.0.0 --port 5040 --workers 1
hypercorn app:api.app --bind 0.0.0.0:5040 --workers 8