"""Voice Task A model loading and inference.

Load models lazily with a thread-safe cache — copy the `_load_once` idiom from
app/services/model_loader_service.py rather than editing that shared file
(transformers weight loading is not thread-safe; see commit 0c21127).

Example skeleton:

    import threading

    _LOAD_LOCK = threading.Lock()
    _MODEL_CACHE: dict = {}

    def _load_once(key: str, factory):
        with _LOAD_LOCK:
            if key not in _MODEL_CACHE:
                _MODEL_CACHE[key] = factory()
            return _MODEL_CACHE[key]

    def get_model():
        return _load_once("task-a-model", lambda: ...)
"""
