import streamlit as st
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import Counter
from calculos_estatisticos import (
    calcular_media,
    calcular_desvio_padrao,
    identificar_limites_outliers,
    calcular_assimetria_pearson,
    calcular_correlacao_pearson_xy,
    calcular_regressao_linear,
    calcular_qui_quadrado,
    encontrar_voto_menor,
    encontrar_voto_maior,
)

st.set_page_config(
    page_title="Análise IMDb",
    page_icon="🎬",
    layout="wide",
)

OURO = "#F5C518"
ROSA = "#FF4B6E"
CIANO = "#3DD6D0"
VERDE = "#5CE65C"

def secao(titulo, descricao=None):
    corpo = f"<div class='secao'><h2>{titulo}</h2>"
    if descricao:
        corpo += f"<p>{descricao}</p>"
    corpo += "</div>"
    st.markdown(corpo, unsafe_allow_html=True)


st.markdown(
    """
    <div class="hero">
        <h1>Análise Estatística — Top 5000 TV Shows IMDb</h1>
        <p>Este trabalho analisa estatisticamente um conjunto de dados contendo séries de
        televisão do IMDb. Foram aplicadas medidas de posição e dispersão, identificação de
        outliers, análise de associação utilizando o teste Qui-Quadrado e análise de
        correlação e regressão linear entre número de votos e avaliação média.</p>
        <p>Os dados foram obtidos do dataset <strong>IMDb Top 5000 TV Shows</strong>,
        disponível na plataforma Kaggle.</p>
        <p><a href="https://www.kaggle.com/datasets/tiagoadrianunes/imdb-top-5000-tv-shows" target="_blank">Acessar o dataset no Kaggle</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def carregar_dados():
    df = pd.read_csv("data/imdb_top_5000_tv_shows.csv")
    df = df.dropna(subset=["averageRating"])
    return df


@st.cache_data
def carregar_ranking_diretores():
    df = pd.read_csv("data/imdb_top_100_tv_shows.csv")
    rows = []
    for _, row in df.iterrows():
        if pd.isna(row["directors"]):
            continue
        for d in [n.strip() for n in row["directors"].split(",") if n.strip()]:
            rows.append({
                "director": d,
                "show": row["primaryTitle"],
                "rating": row["averageRating"],
                "rank": row["rank"],
            })
    ddf = pd.DataFrame(rows).drop_duplicates(subset=["director", "show"])
    summary = ddf.groupby("director").agg(
        num_shows=("show", "count"),
        avg_rating=("rating", "mean"),
        shows=("show", list),
        ratings=("rating", list),
    ).reset_index()
    result = (
        summary[summary["num_shows"] >= 5]
        .sort_values("avg_rating", ascending=False)
        .reset_index(drop=True)
    )
    result.index += 1
    return result


@st.cache_data
def carregar_top_100():
    return pd.read_csv("data/imdb_top_100_tv_shows.csv")

df = carregar_dados()

top_100_df = df.head(100).copy()
top_100_df["Rank"] = range(1, 101)

top_100_explodido = top_100_df.assign(
    genres=top_100_df["genres"].str.split(",")
).explode("genres").reset_index(drop=True)
top_100_explodido["genres"] = top_100_explodido["genres"].str.strip()

tab_geral, tab_top100, tab_testes, tab_talentos = st.tabs(
    ["Visão Geral", "Top 100", "Testes Estatísticos", "Talentos"]
)

with tab_geral:
    secao(
        "Medidas de Posição e Dispersão",
        "Estatísticas da base completa de 5000 séries.",
    )

    notas = df["averageRating"].tolist()
    media_notas = calcular_media(notas)
    desvio_notas = calcular_desvio_padrao(notas)
    assimetria = calcular_assimetria_pearson(notas)

    col1, col2, col3 = st.columns(3)
    col1.metric("Média das Notas", f"{media_notas:.2f}")
    col2.metric("Desvio Padrão", f"{desvio_notas:.2f}")
    col3.metric("Assimetria de Pearson", f"{assimetria:.3f}")

    st.info(
        f"A média das notas dos melhores programas é **{media_notas:.2f}**, com uma "
        f"variação média de **{desvio_notas:.2f}** para mais ou para menos."
    )

    with st.expander("Visualizar o dataset bruto"):
        st.dataframe(df.head(5000), width='stretch')

    st.divider()
    secao(
        "Duração das Séries Programa Mais Longo e Mais Curto",
        "Tempo no ar (anos entre o início e o fim) na base completa de séries. "
        "Séries sem ano de término informado (ainda no ar) não entram no cálculo.",
    )

    df_duracao = df.dropna(subset=["startYear", "endYear"]).copy()
    df_duracao["Duracao"] = (df_duracao["endYear"] - df_duracao["startYear"]).astype(int)
    df_duracao = df_duracao.reset_index(drop=True)

    duracoes = df_duracao["Duracao"].tolist()
    indice_mais_longo = encontrar_voto_maior(duracoes)
    indice_mais_curto = encontrar_voto_menor(duracoes)

    serie_longa = df_duracao.iloc[indice_mais_longo]
    serie_curta = df_duracao.iloc[indice_mais_curto]

    col_long, col_curt = st.columns(2)
    with col_long:
        st.markdown("#### Programa Mais Longo")
        st.metric(serie_longa["primaryTitle"], f"{int(serie_longa['Duracao'])} anos no ar")
        st.caption(
            f"No ar de {int(serie_longa['startYear'])} a {int(serie_longa['endYear'])} "
            f"· Nota {serie_longa['averageRating']}"
        )
    with col_curt:
        st.markdown("#### Programa Mais Curto")
        st.metric(serie_curta["primaryTitle"], f"{int(serie_curta['Duracao'])} anos no ar")
        st.caption(
            f"No ar de {int(serie_curta['startYear'])} a {int(serie_curta['endYear'])} "
            f"· Nota {serie_curta['averageRating']}"
        )

with tab_top100:
    notas_top_100 = top_100_df["averageRating"].tolist()
    media_top100 = calcular_media(notas_top_100)
    desvio_top100 = calcular_desvio_padrao(notas_top_100)

    secao(
        "Análise do Top 100 Séries",
        "Recorte das 100 séries mais bem avaliadas.",
    )

    col_t1, col_t2 = st.columns(2)
    col_t1.metric("Média do Top 100", f"{media_top100:.2f}")
    col_t2.metric("Desvio Padrão do Top 100", f"{desvio_top100:.2f}")

    col_gen, col_out = st.columns(2)

    with col_gen:
        st.markdown("#### Gêneros mais frequentes")
        contagem_generos = top_100_explodido["genres"].value_counts()
        st.dataframe(contagem_generos, width='stretch')

    with col_out:
        st.markdown("#### Outliers no Top 100")
        limite_inf_100, limite_sup_100 = identificar_limites_outliers(notas_top_100)
        outliers_top100 = top_100_df[
            (top_100_df["averageRating"] > limite_sup_100)
            | (top_100_df["averageRating"] < limite_inf_100)
        ]
        st.dataframe(
            outliers_top100[["primaryTitle", "averageRating", "genres"]],
            width='stretch',
        )

    st.divider()
    secao("Curiosidades do Top 100")

    votos_top100 = top_100_df["numVotes"].tolist()
    indice_voto_menor = encontrar_voto_menor(votos_top100)
    indice_voto_maior = encontrar_voto_maior(votos_top100)

    col_menor, col_maior = st.columns(2)
    with col_menor:
        st.markdown("#### 🔻 Menor número de votos")
        st.dataframe(
            top_100_df.iloc[[indice_voto_menor]][
                ["Rank", "primaryTitle", "numVotes", "averageRating"]
            ],
            width='stretch',
        )
    with col_maior:
        st.markdown("#### 🔺 Maior número de votos")
        st.dataframe(
            top_100_df.iloc[[indice_voto_maior]][
                ["Rank", "primaryTitle", "numVotes", "averageRating"]
            ],
            width='stretch',
        )

with tab_testes:
    secao(
        "Associação: Gênero Principal vs Avaliação",
        "Teste Qui-Quadrado (χ²) de independência.",
    )

    df_chi = df.copy()
    df_chi["PrimaryGenre"] = df_chi["genres"].str.split(",").str[0].str.strip()

    generos_alvo = [
        "Action", "Comedy", "Drama", "Crime",
        "Adventure", "Biography", "Animation", "Documentary",
    ]
    df_chi = df_chi[df_chi["PrimaryGenre"].isin(generos_alvo)]

    def categorizar_nota(nota):
        if nota < 7.0:
            return "Baixa (< 7.0)"
        elif nota <= 8.0:
            return "Média (7.0 a 8.0)"
        else:
            return "Alta (> 8.0)"

    df_chi["Faixa_Avaliacao"] = df_chi["averageRating"].apply(categorizar_nota)

    ordem_colunas = ["Baixa (< 7.0)", "Média (7.0 a 8.0)", "Alta (> 8.0)"]
    tabela_contingencia = pd.crosstab(
        df_chi["PrimaryGenre"], df_chi["Faixa_Avaliacao"]
    ).reindex(columns=ordem_colunas)

    st.markdown("**Tabela de Contingência (frequências observadas):**")
    st.dataframe(tabela_contingencia, width='stretch')

    matriz_para_calculo = tabela_contingencia.values.tolist()
    valor_qui_quadrado, graus_liberdade = calcular_qui_quadrado(matriz_para_calculo)
    p_valor_chi = 1 - stats.chi2.cdf(valor_qui_quadrado, graus_liberdade)

    col_chi1, col_chi2, col_chi3 = st.columns(3)
    col_chi1.metric("Qui-Quadrado (χ²)", f"{valor_qui_quadrado:.2f}")
    col_chi2.metric("Graus de Liberdade (gl)", f"{graus_liberdade}")
    col_chi3.metric("P-Valor", f"{p_valor_chi:.2e}")

    if p_valor_chi < 0.05:
        st.success(
            f"Como o p-valor ({p_valor_chi:.2e}) é menor que 0.05, "
            "**rejeita-se a Hipótese Nula (H₀)**. Fica provada a associação "
            "estatisticamente significativa entre o gênero principal da série e "
            "sua faixa de avaliação."
        )
    else:
        st.warning("Falha em rejeitar H₀. Não há associação significativa.")

    st.divider()

    # --- Correlação e Regressão Global -----------------------------------
    secao(
        "Correlação e Regressão Global (Votos vs Notas)",
        "Popularidade vs qualidade na base completa de séries.",
    )

    votos_global = df["numVotes"].tolist()
    notas_global = df["averageRating"].tolist()

    r_global = calcular_correlacao_pearson_xy(votos_global, notas_global)
    a_global, b_global = calcular_regressao_linear(votos_global, notas_global)
    r_quadrado = r_global ** 2
    _, p_valor_reg = stats.pearsonr(votos_global, notas_global)

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Correlação (r)", f"{r_global:.4f}")
    col_r2.metric("Determinação (R²)", f"{r_quadrado:.4f}")
    col_r3.metric("P-Valor", f"{p_valor_reg:.2e}")

    st.info(
        "Observou-se uma correlação positiva fraca entre o número de votos e a "
        f"avaliação média das séries. O coeficiente de determinação (R² = "
        f"{r_quadrado:.4f}) indica que cerca de {r_quadrado * 100:.2f}% da "
        "variação das notas pode ser explicada pela popularidade medida pelo "
        f"número de votos. Apesar da força da relação ser baixa, o p-valor pequeno "
        f"({p_valor_reg:.2e}) indica que a relação dificilmente ocorreu ao acaso."
    )

    st.divider()

    # Predição (Votos + Gênero)
    secao(
        "Predição (Votos + Gênero)",
        "Regressão linear (y = a + bx) filtrada por gênero para prever nota e posição.",
    )

    lista_generos = sorted(top_100_explodido["genres"].unique().tolist())

    col_g, col_v = st.columns(2)
    genero_escolhido = col_g.selectbox("1. Escolha o gênero da série:", lista_generos)
    votos_simulados = col_v.number_input(
        "2. Digite a estimativa de votos:",
        min_value=0, value=500000, step=50000,
    )

    df_genero = top_100_explodido[top_100_explodido["genres"] == genero_escolhido]
    votos_gen = df_genero["numVotes"].tolist()
    notas_gen = df_genero["averageRating"].tolist()
    rank_gen = df_genero["Rank"].tolist()

    if len(votos_gen) > 1:
        r_teste_confiabilidade = calcular_correlacao_pearson_xy(votos_gen, notas_gen)

        if abs(r_teste_confiabilidade) < 0.6:
            st.warning(
                "Não é aconselhável usar a reta para fazer projeções quando o "
                "coeficiente r é muito baixo, pois as estimativas perdem "
                "confiabilidade. O resultado abaixo é exibido apenas para "
                "demonstração algorítmica e visual."
            )

        a_nota, b_nota = calcular_regressao_linear(votos_gen, notas_gen)
        a_rank, b_rank = calcular_regressao_linear(votos_gen, rank_gen)

        nota_prevista = min((a_nota + (b_nota * votos_simulados)), 10.0)
        rank_previsto = max(1, min(100, int(a_rank + (b_rank * votos_simulados))))

        st.success("**Previsão concluída**")

        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Nota Aproximada", f"{nota_prevista:.2f}")
        col_res2.metric("Posição Estimada no Rank", f"{rank_previsto}º")

        st.markdown("#### Visualização da Regressão Linear")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#171A23")
        for borda in ax.spines.values():
            borda.set_color("#3A3F4B")
        ax.tick_params(colors="#C9CDD6")
        ax.grid(True, linestyle="--", alpha=0.3, color="#3A3F4B")

        ax.scatter(
            votos_gen, notas_gen,
            color=CIANO, s=60, alpha=0.85, edgecolor="#0E1117",
            label="Séries Reais",
        )

        x_reta = [min(votos_gen) * 0.9, max(votos_gen) * 1.1]
        y_reta = [a_nota + (b_nota * x) for x in x_reta]
        ax.plot(
            x_reta, y_reta,
            color=ROSA, linewidth=2.5,
            label=f"Reta de Regressão\n(y = {a_nota:.2f} + {b_nota:.6f}x)",
        )

        ax.scatter(
            [votos_simulados], [nota_prevista],
            color=VERDE, s=180, marker="*", edgecolor="#0E1117", zorder=5,
            label="Previsão",
        )

        ax.set_title(
            f"Votos vs Notas — Gênero: {genero_escolhido}",
            color=OURO, fontweight="bold",
        )
        ax.set_xlabel("Número de Votos", color="#C9CDD6")
        ax.set_ylabel("Nota Média", color="#C9CDD6")
        legenda = ax.legend(facecolor="#171A23", edgecolor="#3A3F4B")
        for texto in legenda.get_texts():
            texto.set_color("#C9CDD6")

        st.pyplot(fig)

        with st.expander("Detalhes do cálculo"):
            st.write(f"Coeficiente r do subgrupo: {r_teste_confiabilidade:.4f}")
            st.write(f"Equação da Nota: $y = {a_nota:.4f} + {b_nota:.8f}x$")
            st.write(f"Equação do Rank: $y = {a_rank:.4f} + {b_rank:.8f}x$")
    else:
        st.error(
            f"Amostragem insuficiente (n = {len(votos_gen)}) para o gênero "
            f"'{genero_escolhido}'. É impossível traçar uma reta de regressão."
        )

with tab_talentos:
    df_result = carregar_ranking_diretores()

    secao(
        "Diretores com Melhor Média por Série",
        "Diretores das Top 100 séries do IMDb com participação em 5 ou mais séries.",
    )

    with st.expander(" Como o cálculo foi feito?", expanded=False):
        st.markdown(
            """
**Passos do cálculo:**

1. **Extração dos diretores** — cada série pode ter múltiplos diretores listados separados por vírgula na coluna `directors`. Cada nome foi separado individualmente.

2. **Deduplicação por série** — um diretor pode aparecer listado mais de uma vez na mesma série (episódios diferentes). Para evitar que isso influa na média, cada par *(diretor × série)* foi contado **apenas uma vez**.

3. **Filtro de participação** — foram mantidos apenas os diretores com participação em **5 ou mais séries distintas**.

4. **Cálculo da média** — para cada diretor, calculou-se a **média aritmética simples** das notas (`averageRating`) de todas as séries em que participou:

$$\\bar{x} = \\frac{\\sum_{i=1}^{n} r_i}{n}$$

onde $r_i$ é a nota da série $i$ e $n$ é o número de séries.

5. **Ordenação** — os diretores foram ordenados do maior para o menor valor de média.
            """
        )

    st.markdown("####  Pódio — Top 3")
    medals = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)
    for i, col in enumerate(cols):
        row = df_result.iloc[i]
        with col:
            st.markdown(f"### {medals[i]} {row['director']}")
            st.metric("Média de Avaliação", f"{row['avg_rating']:.3f}")
            st.caption(f"Participou de **{row['num_shows']}** séries")

    st.divider()
    st.markdown("#### Ranking Completo")

    display_df = df_result[["director", "num_shows", "avg_rating"]].copy()
    display_df.columns = ["Diretor", "Nº de Séries", "Média de Avaliação"]
    display_df["Média de Avaliação"] = display_df["Média de Avaliação"].map("{:.3f}".format)
    display_df.index.name = "Posição"
    st.dataframe(display_df, width='stretch')

    st.divider()
    st.markdown("#### Detalhe por Diretor")
    selected = st.selectbox(
        "Selecione um diretor para ver as séries em que participou:",
        options=df_result["director"].tolist(),
        format_func=lambda x: f"{df_result[df_result['director'] == x].index[0]}º — {x}",
    )

    row = df_result[df_result["director"] == selected].iloc[0]
    detail_df = pd.DataFrame({
        "Série": row["shows"],
        "Nota IMDb": [f"{r:.1f}" for r in row["ratings"]],
    }).sort_values("Nota IMDb", ascending=False).reset_index(drop=True)
    detail_df.index += 1

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.dataframe(detail_df, width='stretch')
    with col_b:
        st.metric("Média geral", f"{row['avg_rating']:.3f}")
        st.metric("Total de séries", int(row["num_shows"]))
        st.metric("Nota mais alta", f"{max(row['ratings']):.1f}")
        st.metric("Nota mais baixa", f"{min(row['ratings']):.1f}")

    st.divider()
    secao(
        "Top 5 Diretores e Escritores",
        "Os profissionais mais presentes nas séries mais bem avaliadas.",
    )

    df_top100_raw = carregar_top_100()

    def count_names(series):
        all_names = []
        for entry in series.dropna():
            names = [n.strip() for n in entry.split(",") if n.strip()]
            all_names.extend(names)
        return Counter(all_names)

    director_counts = count_names(df_top100_raw["directors"])
    writer_counts = count_names(df_top100_raw["writers"])

    top5_directors = director_counts.most_common(5)
    top5_writers = writer_counts.most_common(5)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top 5 Diretores")
        for i, (name, count) in enumerate(top5_directors, start=1):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i - 1]
            st.markdown(f"**{medal} {i}. {name}**")
            st.progress(count / top5_directors[0][1])
            st.caption(f"{count} aparição{'ões' if count > 1 else ''}")

    with col2:
        st.markdown("#### Top 5 Escritores")
        for i, (name, count) in enumerate(top5_writers, start=1):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i - 1]
            st.markdown(f"**{medal} {i}. {name}**")
            st.progress(count / top5_writers[0][1])
            st.caption(f"{count} aparição{'ões' if count > 1 else ''}")

    with st.expander(" Ver dados brutos"):
        st.dataframe(
            df_top100_raw[
                ["primaryTitle", "startYear", "averageRating", "directors", "writers"]
            ].reset_index(drop=True),
            width='stretch',
        )

st.markdown(
    "<div class='rodape'> Análise Estatística IMDb — Trabalho de Estatística</div>",
    unsafe_allow_html=True,
)