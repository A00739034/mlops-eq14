"""
Script para probar la API de análisis de riesgo crediticio
"""

import requests
import json
from typing import Dict, Any

# URL base de la API
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Probar el health check"""
    print("\n🔍 Probando Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_model_info():
    """Probar endpoint de información del modelo"""
    print("\n📊 Probando Model Info...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_single_prediction():
    """Probar predicción individual"""
    print("\n🎯 Probando Predicción Individual...")
    
    # Caso 1: Solicitud con alta probabilidad de aprobación
    data_high_risk = {
        "age": 35,
        "gender": 1,
        "income": 7500.0,
        "employment_type": 2,
        "credit_history": 4,
        "loan_amount": 10000.0,
        "loan_term": 36,
        "existing_loans": 0,
        "debt_to_income_ratio": 0.15
    }
    
    print("\n📝 Caso 1: Perfil de BAJO riesgo")
    response = requests.post(f"{BASE_URL}/predict", json=data_high_risk)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predicción: {'APROBADO' if result['prediction'] == 1 else 'RECHAZADO'}")
        print(f"Probabilidad: {result['probability']:.2%}")
        print(f"Nivel de Riesgo: {result['risk_level']}")
        print(f"Recomendación: {result['recommendation']}")
    
    # Caso 2: Solicitud con baja probabilidad de aprobación
    data_low_risk = {
        "age": 22,
        "gender": 0,
        "income": 1500.0,
        "employment_type": 0,
        "credit_history": 0,
        "loan_amount": 30000.0,
        "loan_term": 60,
        "existing_loans": 3,
        "debt_to_income_ratio": 0.85
    }
    
    print("\n📝 Caso 2: Perfil de ALTO riesgo")
    response = requests.post(f"{BASE_URL}/predict", json=data_low_risk)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predicción: {'APROBADO' if result['prediction'] == 1 else 'RECHAZADO'}")
        print(f"Probabilidad: {result['probability']:.2%}")
        print(f"Nivel de Riesgo: {result['risk_level']}")
        print(f"Recomendación: {result['recommendation']}")
    
    # Caso 3: Solicitud moderada
    data_medium = {
        "age": 28,
        "gender": 1,
        "income": 4000.0,
        "employment_type": 1,
        "credit_history": 2,
        "loan_amount": 15000.0,
        "loan_term": 48,
        "existing_loans": 1,
        "debt_to_income_ratio": 0.40
    }
    
    print("\n📝 Caso 3: Perfil MODERADO")
    response = requests.post(f"{BASE_URL}/predict", json=data_medium)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predicción: {'APROBADO' if result['prediction'] == 1 else 'RECHAZADO'}")
        print(f"Probabilidad: {result['probability']:.2%}")
        print(f"Nivel de Riesgo: {result['risk_level']}")
        print(f"Recomendación: {result['recommendation']}")
    
    return True

def test_batch_prediction():
    """Probar predicción por lote"""
    print("\n📦 Probando Predicción por Lote...")
    
    applications = [
        {
            "age": 35,
            "gender": 1,
            "income": 5000.0,
            "employment_type": 2,
            "credit_history": 3,
            "loan_amount": 15000.0,
            "loan_term": 36,
            "existing_loans": 1,
            "debt_to_income_ratio": 0.3
        },
        {
            "age": 28,
            "gender": 0,
            "income": 3500.0,
            "employment_type": 1,
            "credit_history": 2,
            "loan_amount": 10000.0,
            "loan_term": 24,
            "existing_loans": 0,
            "debt_to_income_ratio": 0.25
        },
        {
            "age": 45,
            "gender": 1,
            "income": 8000.0,
            "employment_type": 3,
            "credit_history": 4,
            "loan_amount": 25000.0,
            "loan_term": 60,
            "existing_loans": 2,
            "debt_to_income_ratio": 0.35
        }
    ]
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=applications)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nTotal de predicciones: {result['total_predictions']}")
        
        for i, pred in enumerate(result['predictions'], 1):
            print(f"\n--- Solicitud {i} ---")
            print(f"Edad: {pred['input']['age']}, Ingreso: ${pred['input']['income']:,.2f}")
            print(f"Predicción: {'APROBADO' if pred['prediction'] == 1 else 'RECHAZADO'}")
            print(f"Probabilidad: {pred['probability']:.2%}")
            print(f"Riesgo: {pred['risk_level']}")
    
    return True

def test_validation_errors():
    """Probar validación de datos"""
    print("\n⚠️ Probando Validación de Errores...")
    
    # Datos inválidos: edad fuera de rango
    invalid_data = {
        "age": 150,  # Edad inválida
        "gender": 1,
        "income": -1000,  # Ingreso negativo
        "employment_type": 2,
        "credit_history": 3,
        "loan_amount": 15000.0,
        "loan_term": 36,
        "existing_loans": 1,
        "debt_to_income_ratio": 1.5  # Ratio mayor a 1
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
    print(f"Status Code: {response.status_code}")
    print(f"Error esperado (validación): {response.status_code == 422}")
    
    if response.status_code == 422:
        print("✅ Validación funcionando correctamente")
        print(f"Errores: {json.dumps(response.json(), indent=2)}")
    
    return True

def run_all_tests():
    """Ejecutar todas las pruebas"""
    print("=" * 60)
    print("🚀 INICIANDO PRUEBAS DE LA API")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Predicción Individual", test_single_prediction),
        ("Predicción por Lote", test_batch_prediction),
        ("Validación de Errores", test_validation_errors)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "✅ PASSED" if success else "❌ FAILED"))
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Error: No se pudo conectar a la API en {BASE_URL}")
            print("   Asegúrate de que el servidor esté ejecutándose:")
            print("   uvicorn api.main:app --reload")
            return
        except Exception as e:
            results.append((test_name, f"❌ ERROR: {str(e)}"))
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)
    for test_name, result in results:
        print(f"{test_name:.<50} {result}")
    print("=" * 60)
    
    # Estadísticas
    passed = sum(1 for _, result in results if "PASSED" in result)
    total = len(results)
    print(f"\n✅ Pruebas exitosas: {passed}/{total} ({passed/total*100:.1f}%)")

if __name__ == "__main__":
    run_all_tests()
