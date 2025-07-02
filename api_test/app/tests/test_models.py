from app.utils import model_loader

def test_get_model():
    model = model_loader.get_model()
    assert model is not None

def test_get_preprocessor():
    preprocessor = model_loader.get_preprocessor()
    assert preprocessor is not None

def test_get_model_metadata():
    metadata = model_loader.get_model_metadata()
    assert metadata is not None