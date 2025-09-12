from src.preprocess import load_and_clean

def test_data_loading():
    X, y = load_and_clean()
    assert not X.empty
    assert len(X) == len(y)
