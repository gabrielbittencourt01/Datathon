# 🤖 Recrutamento Inteligente com IA — Projeto Datathon FIAP

Este projeto foi desenvolvido para o **Datathon de Machine Learning Engineering da FIAP**, com o objetivo de aplicar técnicas de Inteligência Artificial para **otimizar o processo de recrutamento** da empresa fictícia **Decision**, especializada em bodyshop e alocação de talentos na área de TI.

## 🧠 Objetivo

Criar uma solução de IA que:

* Estime a **probabilidade de contratação** de um(a) candidato(a)
* Faça **recomendações de vagas** com base no conteúdo do CV e perfis similares anteriores
* Disponibilize tudo isso via **API (FastAPI)** e **aplicação interativa (Streamlit)**

---

## 🗂️ Estrutura do Projeto e Ordem de Execução

### 1. `obtecao.py`

Realiza a ingestão e transformação dos arquivos JSON (`applicants`, `prospects`, `jobs`) e gera `dados_unificados.xlsx`.

```bash
python obtecao.py
```

### 2. `processamento.py`

Faz a limpeza e pré-processamento dos dados. Gera `base_tratada.parquet`.

```bash
python processamento.py
```

### 3. `feature_engineering.py`

Aplica TF-IDF e OneHotEncoder. Salva os dados vetorizados para treino/teste.

```bash
python feature_engineering.py
```

### 4. `modelo_train_multimodal.py`

Treina o modelo final combinando embeddings SBERT + dados categóricos. Salva artefatos na pasta `models/`.

```bash
python modelo_train_multimodal.py
```

### 5. `gerar_embeddings_vagas.py`

Gera embeddings vetoriais para recomendação de vagas.

```bash
python gerar_embeddings_vagas.py
```

### 6. `api.py`

Inicia a API com FastAPI. Contém o endpoint `/analisar_candidato`.

```bash
uvicorn api:app --reload
```

Acesse em: [http://localhost:8000/docs](http://localhost:8000/docs)

### 7. `test_api.py`

Testa a API localmente enviando uma requisição de exemplo.

```bash
python test_api.py
```

### 8. `test_pipeline.py`

Executa testes unitários no modelo salvo.

```bash
pytest test_pipeline.py -v
```

### 9. `monitoramento.py`

Compara dados recentes com os dados originais e gera um relatório de drift.

```bash
python monitoramento.py
```

### 10. `app_streamlit.py`

Executa o app interativo de análise de candidatos e recomendação de vagas.

```bash
streamlit run app_streamlit.py
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
