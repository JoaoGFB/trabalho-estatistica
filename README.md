# Análise Estatística — Top 5000 TV Shows IMDb

A aplicação analisa estatisticamente um conjunto de dados de séries de televisão do IMDb,
aplicando medidas de posição e dispersão, identificação de outliers, teste Qui-Quadrado de
associação e análise de correlação e regressão linear entre número de votos e avaliação
média. A interface é construída com [Streamlit](https://streamlit.io/).

Os dados foram obtidos do dataset [IMDb Top 5000 TV Shows](https://www.kaggle.com/datasets/tiagoadrianunes/imdb-top-5000-tv-shows),
disponível na plataforma Kaggle.

## Instalação

1. Clone o repositório e entre na pasta do projeto:

   ```bash
   git clone https://github.com/JoaoGFB/trabalho-estatistica
   cd trabalho-estatistica
   ```

2. Crie e ative um ambiente virtual (venv):

   ```bash
   python -m venv .venv

   # Linux / macOS
   source .venv/bin/activate

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

## Como usar

Com o ambiente virtual ativado, execute:

```bash
streamlit run app.py
```

O Streamlit abrirá a aplicação no navegador (por padrão em `http://localhost:8501`).