# Music Genre Classifier 🎵

API REST que classifica o gênero musical de arquivos de áudio usando Machine Learning.

## Demo

API em produção: https://music-genre-classifier-os8j.onrender.com/docs

## Como funciona

O modelo foi treinado com o dataset GTZAN (1000 músicas, 10 gêneros). Para cada arquivo de áudio, o Librosa extrai 57 features — MFCCs, spectral bandwidth, chroma, tempo, entre outras — que alimentam um classificador XGBoost.

## Resultados

| Modelo | Acurácia |
|--------|----------|
| XGBoost | **73.5%** |
| SVM | 73.0% |
| Random Forest | 68.5% |

### Análise de erros

O modelo acerta bem gêneros com características acústicas distintas (classical: F1 0.95, metal: F1 0.89). Os maiores erros ocorrem em gêneros com sobreposição histórica — rock confunde com blues e country (F1 0.47), disco confunde com hiphop e pop (F1 0.52). Esses erros são musicalmente esperados.

## Endpoints

- `POST /predict` — recebe arquivo `.wav`, retorna gênero e confiança
- `GET /predictions` — histórico de predições
- `GET /health` — status da API e do modelo

## Stack

Python · FastAPI · XGBoost · Librosa · SQLAlchemy · Docker · Render

## Decisões de arquitetura

**Paridade treino/inferência:** O modelo foi treinado no CSV do GTZAN. Para garantir que a API reproduz exatamente o mesmo vetor de 57 features, foi criado o módulo `src/features.py` com a função `extract_features()` — usada tanto na validação quanto na inferência. Diferenças validadas contra o CSV original ficaram abaixo de 0.3%.

## Como rodar localmente

```bash
git clone https://github.com/Leonardomachad0/music-genre-classifier
cd music-genre-classifier
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Acesse http://localhost:8000/docs