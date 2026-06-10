import streamlit as st
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from calculos_estatisticos import (
    calcular_media,
    calcular_desvio_padrao,
    identificar_limites_outliers,
    calcular_assimetria_pearson,
    calcular_correlacao_pearson_xy,
    calcular_regressao_linear,
    calcular_qui_quadrado,
    encontrar_voto_menor
)

st.set_page_config(page_title="Análise IMDB", layout="wide")

st.title("Análise Estatística - Top 5000 Tv Shows IMDB")

@st.cache_data
def carregar_dados():
    df = pd.read_csv('data/imdb_top_5000_tv_shows.csv') 
    df = df.dropna(subset=['averageRating']) 
    return df

df = carregar_dados()

if st.checkbox("Visualização do Dataset bruto", value=False):
    st.dataframe(df.head())

st.header("1. Medidas de Posição e Dispersão")

notas = df['averageRating'].tolist()

media_notas = calcular_media(notas)
desvio_notas = calcular_desvio_padrao(notas)
assimetria = calcular_assimetria_pearson(notas)

col1, col2, col3 = st.columns(3)
col1.metric("Média das Notas", f"{media_notas:.2f}")
col2.metric("Desvio Padrão", f"{desvio_notas:.2f}")
col3.metric("Assimetria Pearson", f"{assimetria:.3f}")

st.write(f"Conclusão: A média das notas dos melhores programas é {media_notas:.2f}, com uma variação média de {desvio_notas:.2f} para mais ou para menos.")

top_100_df = df.head(100).copy()
top_100_df['Rank'] = range(1, 101)

top_100_explodido = top_100_df.assign(genres=top_100_df['genres'].str.split(',')).explode('genres')
top_100_explodido = top_100_explodido.reset_index(drop=True)
top_100_explodido['genres'] = top_100_explodido['genres'].str.strip()

st.header("Análise do Top 100 Séries")

st.subheader("Gêneros mais frequentes no Top 100")
contagem_generos = top_100_explodido['genres'].value_counts()
st.dataframe(contagem_generos)

notas_top_100 = top_100_df['averageRating'].tolist()
media_top100 = calcular_media(notas_top_100)
desvio_top100 = calcular_desvio_padrao(notas_top_100)

col_t1, col_t2 = st.columns(2)
col_t1.metric("Média do Top 100", f"{media_top100:.2f}")
col_t2.metric("Desvio Padrão do Top 100", f"{desvio_top100:.2f}")

limite_inf_100, limite_sup_100 = identificar_limites_outliers(notas_top_100)

outliers_top100 = top_100_df[
    (top_100_df['averageRating'] > limite_sup_100) | 
    (top_100_df['averageRating'] < limite_inf_100)
]

st.subheader("Outliers no Top 100")
st.dataframe(outliers_top100[['primaryTitle', 'averageRating', 'genres']])

st.divider()
st.header("Curiosidades do Top 100")
st.subheader("Série com o menor número de votos no Top 100")
votos_top100 = top_100_df['numVotes'].tolist()
indice_voto_menor = encontrar_voto_menor(votos_top100)

st.dataframe(
    top_100_df.iloc[[indice_voto_menor]][
        ['Rank', 'primaryTitle', 'numVotes', 'averageRating']
    ]
)

st.divider()
st.header("Associação: Gênero Principal vs Avaliação (Teste Qui-Quadrado)")

df_chi = df.copy()
df_chi['PrimaryGenre'] = df_chi['genres'].str.split(',').str[0].str.strip()

generos_alvo = ['Action', 'Comedy', 'Drama', 'Crime', 'Adventure', 'Biography', 'Animation', 'Documentary']
df_chi = df_chi[df_chi['PrimaryGenre'].isin(generos_alvo)]

def categorizar_nota(nota):
    if nota < 7.0: return 'Baixa (< 7.0)'
    elif nota <= 8.0: return 'Média (7.0 a 8.0)'
    else: return 'Alta (> 8.0)'

df_chi['Faixa_Avaliacao'] = df_chi['averageRating'].apply(categorizar_nota)

tabela_contingencia = pd.crosstab(df_chi['PrimaryGenre'], df_chi['Faixa_Avaliacao'])

ordem_colunas = ['Baixa (< 7.0)', 'Média (7.0 a 8.0)', 'Alta (> 8.0)']
tabela_contingencia = tabela_contingencia.reindex(columns=ordem_colunas)

st.write("**Tabela de Contingência (Frequências Observadas):**")
st.dataframe(tabela_contingencia)

matriz_para_calculo = tabela_contingencia.values.tolist()
valor_qui_quadrado, graus_liberdade = calcular_qui_quadrado(matriz_para_calculo)

p_valor_chi = 1 - stats.chi2.cdf(valor_qui_quadrado, graus_liberdade)

col_chi1, col_chi2, col_chi3 = st.columns(3)
col_chi1.metric("Valor do Qui-Quadrado (χ²)", f"{valor_qui_quadrado:.2f}")
col_chi2.metric("Graus de Liberdade (gl)", f"{graus_liberdade}")
col_chi3.metric("P-Valor", f"{p_valor_chi:.2e}")

st.subheader("Interpretação:")
if p_valor_chi < 0.05:
    st.success(f"Como o p-valor ({p_valor_chi:.2e}) é menor que 0.05, **rejeita-se a Hipótese Nula (H₀)**. Fica provada a associação estatisticamente significativa entre o gênero principal da série e sua faixa de avaliação.")
else:
    st.warning("Falha em rejeitar H₀. Não há associação significativa.")

st.divider()
st.header("Correlação e Regressão Global (Votos vs Notas)")
st.write("Para uma análise de popularidade vs qualidade, foi testada a base completa de séries cruzando a quantidade de votos (X) com a nota média (Y).")

votos_global = df['numVotes'].tolist()
notas_global = df['averageRating'].tolist()

r_global = calcular_correlacao_pearson_xy(votos_global, notas_global)
a_global, b_global = calcular_regressao_linear(votos_global, notas_global)

r_quadrado = r_global ** 2
_, p_valor_reg = stats.pearsonr(votos_global, notas_global)

col_r1, col_r2, col_r3 = st.columns(3)
col_r1.metric("Correlação (r)", f"{r_global:.4f}")
col_r2.metric("Coeficiente de Determinação (R²)", f"{r_quadrado:.4f}")
col_r3.metric("P-Valor", f"{p_valor_reg:.2e}")

st.subheader("Discussão:")
st.info(f"Observou-se uma correlação positiva fraca entre o número de votos e a avaliação média das séries. O coeficiente de determinação (R² = {r_quadrado:.4f}) indica que cerca de {r_quadrado*100:.2f}% da variação das notas pode ser explicada pela popularidade medida pelo número de votos. Apesar da força da relação ser baixa, o p-valor pequeno ({p_valor_reg:.2e}) indica que a relação dificilmente ocorreu ao acaso.")

st.divider()
st.subheader("Predição (Votos + Gênero)")
st.write("Aplicação da regressão linear ($y = a + bx$) filtrada por gênero para prever a nota e a posição no ranking da categoria selecionada.")

lista_generos = sorted(top_100_explodido['genres'].unique().tolist())

col_gen, col_vot = st.columns(2)
genero_escolhido = col_gen.selectbox("1. Escolha o Gênero da Série:", lista_generos)
votos_simulados = col_vot.number_input("2. Digite a estimativa de Votos:", min_value=0, value=500000, step=50000)

df_genero = top_100_explodido[top_100_explodido['genres'] == genero_escolhido]

votos_gen = df_genero['numVotes'].tolist()
notas_gen = df_genero['averageRating'].tolist()
rank_gen = df_genero['Rank'].tolist()

if len(votos_gen) > 1:
    r_teste_confiabilidade = calcular_correlacao_pearson_xy(votos_gen, notas_gen)
    
    if abs(r_teste_confiabilidade) < 0.6:
        st.warning("Não é aconselhável a utilização da reta para fazer projeções quando o coeficiente r é muito baixo, pois as estimativas perdem confiabilidade. O resultado abaixo está sendo exibido apenas para demonstração algorítmica e visual.")
    
    a_nota, b_nota = calcular_regressao_linear(votos_gen, notas_gen)
    a_rank, b_rank = calcular_regressao_linear(votos_gen, rank_gen)
    
    nota_prevista = min((a_nota + (b_nota * votos_simulados)), 10.0)
    rank_previsto = max(1, min(100, int(a_rank + (b_rank * votos_simulados))))
    
    st.success(f"**Previsão concluída**")
    
    col_res1, col_res2 = st.columns(2)
    col_res1.metric("Nota Aproximada", f"{nota_prevista:.2f}")
    col_res2.metric("Posição Estimada no Rank", f"{rank_previsto}º")
    
    st.write("### Visualização da Regressão Linear")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.6, color='#D0E1F9')
    
    ax.scatter(votos_gen, notas_gen, color='#00E5FF', s=60, alpha=0.8, edgecolor='#007BFF', label='Séries Reais')
    
    x_reta = [min(votos_gen) * 0.9, max(votos_gen) * 1.1]
    y_reta = [a_nota + (b_nota * x) for x in x_reta]
    ax.plot(x_reta, y_reta, color='#FF007F', linewidth=2.5, label=f'Reta de Regressão\n(y = {a_nota:.2f} + {b_nota:.6f}x)')
    
    ax.scatter([votos_simulados], [nota_prevista], color='#39FF14', s=150, marker='*', edgecolor='black', zorder=5, label='Previsão')
    
    ax.set_title(f"Votos vs Notas - Gênero: {genero_escolhido}", color='#003366', fontweight='bold')
    ax.set_xlabel("Número de Votos", color='#003366')
    ax.set_ylabel("Nota Média", color='#003366')
    ax.legend()
    
    st.pyplot(fig)
    
    with st.expander("Detalhes do cálculo"):
        st.write(f"Coeficiente r do subgrupo: {r_teste_confiabilidade:.4f}")
        st.write(f"Equação da Nota: $y = {a_nota:.4f} + {b_nota:.8f}x$")
        st.write(f"Equação do Rank: $y = {a_rank:.4f} + {b_rank:.8f}x$")
else:
    st.error(f"Amostragem insuficiente (n = {len(votos_gen)}) para o gênero '{genero_escolhido}'. É impossível traçar uma reta de regressão.")