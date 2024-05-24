from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict():
    # ['municipality', 'parish', 'neighborhood', 'garage', 'home_type', 'home_size', 'home_area', 'floor', 'elevator']
    data = {
    "home_type": "Apartamento",
    "garage": True,
    "home_size": 'T2',
    "floor": 2,
    "elevator": False,
    "municipality": "Albufeira",
    'parish': 'Albufeira e olhos de agua',    
    "neighborhood": "Branqueira brejos",
    "home_area": 158.0,
    }
    response = client.post("/predict", 
                           json=data) # 2800
    assert response.status_code == 200
    print(response.json())

"""
# {"Unnamed: 0":148,"district":null,"municipality":"Albufeira","parish":"Albufeira e olhos de agua","neighborhood":"Branqueira brejos","neighborhood_link":
    "https://www.idealista.pt/arrendar-casas/albufeira-e-olhos-de-agua/albufeira/branqueira-brejos/",
    "title":"Apartamento T2 em Branqueira - Brejos, Albufeira, Albufeira e Olhos de Água","link":"https://www.idealista.pt/imovel/33378178/",
    "description":"Indulge in opulence within this 2022-built, 2-bedroom luxury penthouse boasting high-quality construction and upscale finishes. 
    Immerse yourself in modern elegance with a private rooftop terrace featuring a Jacuzzi, overlooking ocean, garden, pool, and coastal views. 
    Furnished with exquisite luxury furnishings, this ha","garage":true,"price":2800,"home_type":"Apartamento","date":1716163200000,"source_link":
    "https://www.idealista.pt/arrendar-casas/albufeira-e-olhos-de-agua/albufeira/branqueira-brejos/","home_size":"T2","home_area":158,"floor":2,"elevator":
    false,"price_per_sqr_meter":17.72151898734177,"__index_level_0__":"Albufeira","__index_level_1__":0}
"""

client = TestClient(app)

def test_get():
    response = client.get("/features")
    assert response.status_code == 200
    print(response.json())

if __name__ == "__main__":
    # test_get()
    test_predict()