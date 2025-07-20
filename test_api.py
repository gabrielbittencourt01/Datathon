# Script para testar a API FastAPI localmente
import requests

url = "http://localhost:8000/analisar_candidato"

payload = {
    "cv_pt": "Experiência com desenvolvimento backend em Python, APIs REST e banco de dados PostgreSQL.",
    "nivel_academico": "Superior Completo",
    "nivel_ingles": "Avançado",
    "nivel_espanhol": "Intermediário",
    "tipo_contratacao": "CLT",
    "area_atuacao": "Desenvolvimento"
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
