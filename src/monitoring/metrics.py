from prometheus_client import Counter, Histogram, start_http_server
import time

# Métriques de base
REQUEST_COUNT = Counter('housing_model_requests_total', 'Total number of requests')
REQUEST_LATENCY = Histogram('housing_model_request_latency_seconds', 'Request latency')
ERROR_COUNT = Counter('housing_model_errors_total', 'Total number of errors')

def track_request():
    """Décorateur pour tracker les requêtes"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            REQUEST_COUNT.inc()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                ERROR_COUNT.inc()
                raise e
            finally:
                latency = time.time() - start_time
                REQUEST_LATENCY.observe(latency)
        return wrapper
    return decorator

# Démarrez le serveur Prometheus
start_http_server(8001)
