import streamlit as st
import pandas as pd
from calculos_estatisticos import calcular_media, calcular_desvio_padrao

st.set_page_config(page_title="Análise IMDB", layout="wide")

st.title("Análise Estatística - Top 5000 Tv Shows IMDB")

@st.cache_data
def carregar_dados():
    df = pd.read_csv('data/imdb_top_5000_tv_shows.csv') 
    df = df.dropna(subset=['averageRating'])#limpa as linhas vazias
    return df

df = carregar_dados()

if st.checkbox("Mostrar dados brutos", value="True"):
    st.dataframe(df.head())

st.header("1. Medidas de Posição e Dispersão")

#a coluna notas (rating) convertida para lista do python
notas = df['averageRating'].tolist()

media_notas = calcular_media(notas)
desvio_notas = calcular_desvio_padrao(notas)

col1, col2 = st.columns(2)
col1.metric("Média das Notas", f"{media_notas:.2f}")
col2.metric("Desvio-Padrão", f"{desvio_notas:.2f}")

st.write(f"Conclusão: A média das notas dos melhores programas é {media_notas:.2f}, com uma variação média de {desvio_notas:.2f}")