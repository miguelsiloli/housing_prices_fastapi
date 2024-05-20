from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict():
    # ['municipality', 'parish', 'neighborhood', 'garage', 'home_type', 'home_size', 'home_area', 'floor', 'elevator']
    data = {
    "home_type": "Apartamento",
    "garage": True,
    "home_size": 'T1',
    "floor": 1,
    "elevator": True,
    "municipality": "albufeira",
    'parish': 'albufeira e olhos de agua',    
    "neighborhood": "quinta da palmeira",
    "home_area": 80.0,
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    print(response.json())

client = TestClient(app)

def test_get():
    response = client.get("/features")
    assert response.status_code == 200
    print(response.json())

if __name__ == "__main__":
    test_get()
    test_predict()