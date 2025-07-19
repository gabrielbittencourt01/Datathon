# 🤖 Recrutamento Inteligente com IA — Projeto Datathon FIAP

Este projeto foi desenvolvido para o **Datathon de Machine Learning Engineering da FIAP**, com o objetivo de aplicar técnicas de Inteligência Artificial para **otimizar o processo de recrutamento** da empresa fictícia **Decision**, especializada em bodyshop e alocação de talentos na área de TI.

## 🧠 Objetivo

Criar uma solução de IA que:

* Estime a **probabilidade de contratação** de um(a) candidato(a)
* Faça **recomendações de vagas** com base no conteúdo do CV e perfis similares anteriores
* Disponibilize tudo isso via **API (FastAPI)** e **aplicação interativa (Streamlit)**

---

## 🗂️ Estrutura do Projeto

```
├── api.py                     # API com FastAPI para predição e recomendação
├── app_streamlit.py           # Interface visual para uso da IA
├── model_train.py             # Modelo com TF-IDF tradicional
├── modelo_train_multimodal.py # Modelo final com embeddings + categorias
├── modelo_train_sbert.py      # Modelo baseado só em SBERT
├── gerar_embeddings_vagas.py # Gera embeddings das vagas
├── vaga_recomendada.py        # Script isolado para recomendação
├── obtecao.py                 # Ingestão e transformação dos arquivos JSON
├── processamento.py           # Pré-processamento e feature cleaning
├── feature_engineering.py     # Criação de pipelines de transformação
├── test_api.py                # Teste com requests da API
├── test_pipeline.py           # Testes unitários simples
├── monitoramento.py           # Detecção de drift com evidently
├── Dockerfile                 # Containerização da API
├── requirements.txt           # Dependências do projeto
├── models/                    # Modelos salvos (.pkl, .npy)
├── base_tratada.csv/parquet   # Dados finais tratados
├── vagas_unicas.csv           # Base única de vagas
```

---

## 🚀 Como Executar Localmente

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/recrutamento-ia.git
cd recrutamento-ia
```

### 2. Crie um ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Inicie a API FastAPI

```bash
uvicorn api:app --reload
```

Acesse: [http://localhost:8000/docs](http://localhost:8000/docs) para testar via Swagger.

### 5. Teste a API via script

```bash
python test_api.py
```

---

## 🐳 Como Rodar com Docker

### Build da imagem:

```bash
docker build -t recrutamento-api .
```

### Executar o container:

```bash
docker run -p 8000:8000 recrutamento-api
```

---

## 📊 Como Rodar o Monitoramento (Drift)

> ⚠️ Você precisa gerar ou simular um arquivo `dados_recentes.csv` com as mesmas colunas que `base_tratada.csv`.

```bash
python monitoramento.py
```

Isso gera um arquivo `drift_report.html` que pode ser aberto no navegador.

---

## 🌐 Interface Interativa com Streamlit

Para usar o app visual:

```bash
streamlit run app_streamlit.py
```

Você verá uma página como:

```
📋 Análise de Candidato e Recomendação de Vagas
[ Preencha os dados do candidato ]
[ CV ]
[ Botão: Analisar Candidato ]
```

---

## ✅ Funcionalidades Completas

* [x] Pipeline de Machine Learning Multimodal
* [x] Embeddings com SBERT
* [x] Recomendação de Vagas com Cosine Similarity
* [x] API com FastAPI
* [x] App interativo com Streamlit
* [x] Dockerfile para empacotamento
* [x] Monitoramento com `evidently`
* [x] Testes unitários simples com `pytest`
* [x] Testes de API com `requests`

---

## 📌 Tecnologias Utilizadas

* Python 3.11
* FastAPI
* Streamlit
* Scikit-Learn
* XGBoost
* Sentence-Transformers (SBERT)
* Docker
* Evidently AI

---

## 🏁 Considerações Finais

Este projeto simula um cenário real de aplicação de Machine Learning em um contexto de negócio com foco em **produtização** e **eficiência em recrutamento**. É possível expandir a solução com feedbacks humanos, banco vetorial (FAISS) e integração com plataformas reais de recrutamento.
