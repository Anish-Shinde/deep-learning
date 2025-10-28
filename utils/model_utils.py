import io
import tempfile
import os
import joblib

def model_pickle_bytes(model):
    """Helper to pickle a model into bytes."""
    with io.BytesIO() as buffer:
        joblib.dump(model, buffer)
        return buffer.getvalue()

def model_weights_bytes(model):
    """Helper to save model weights to bytes."""
    tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        try:
            model.save_weights(tmp_path, save_format='h5')
        except TypeError:
            model.save_weights(tmp_path)
        with open(tmp_path, 'rb') as f:
            data = f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return data