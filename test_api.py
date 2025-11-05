import requests
import json

def test_api():
    """Ejemplo de cómo usar la API desde Python"""
    
    # URL base de la API
    base_url = "http://localhost:8000"
    
    # 1. Health check
    print("🔍 Verificando estado de la API...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()
    
    # 2. Información general
    print("ℹ️ Información de la API...")
    response = requests.get(f"{base_url}/")
    print(f"Response: {response.json()}")
    print()
    
    # 3. Predicción (necesitas una imagen)
    # Descomenta y ajusta la ruta de la imagen
    """
    print("🔮 Realizando predicción...")
    
    # Reemplaza con la ruta a tu imagen
    image_path = "ruta/a/tu/imagen.jpg"
    
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(f"{base_url}/predict", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Clase predicha: {result['class_name']} (clase {result['class']})")
        print(f"Confianza: {result['confidence']:.4f}")
        print(f"Probabilidades: {result['probabilities']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    """

if __name__ == "__main__":
    test_api()