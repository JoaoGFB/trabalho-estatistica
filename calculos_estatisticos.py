import math

def calcular_media(valores):
    #média aritmética simples
    n = len(valores)
    if n == 0:
        return 0
    soma = sum(valores)
    return soma / n

def calcular_desvio_padrao(valores):
    #desvio-padrão amostral
    n = len(valores)
    if n <= 1:
        return 0
    
    media = calcular_media(valores)
    
    #somatório de (Xi - Media)^2
    soma_quadrados = sum((x - media) ** 2 for x in valores)
    
    #desvio padrão (amostral usa n-1) 
    variancia = soma_quadrados / (n - 1)
    return math.sqrt(variancia)

def calcular_quartil(valores, q):
    #quartil (1, 2 ou 3)
    n = len(valores)
    if n == 0:
        return 0
    
    ordenados = sorted(valores)#ordenação dos dados
    
    #posição EQi = i(N+1)/4
    posicao = q * (n + 1) / 4
    
    #usar math.floor para arredondar para baixo para o índice ficar exato
    idx_base = math.floor(posicao) - 1 
    fracao = posicao - math.floor(posicao)#guarda também o que pode sobrar
    
    if idx_base < 0:
        return ordenados[0]
    if idx_base >= n - 1:
        return ordenados[-1]
        
    valor_base = ordenados[idx_base]
    proximo_valor = ordenados[idx_base + 1]
    
    #ex: posição 4.5, ele pega o 4º valor + 50% da diferença para o 5º valor
    return valor_base + fracao * (proximo_valor - valor_base)


def identificar_limites_outliers(valores):
    #faixa inter quartil (FIQ), retorna os limites inferior e superior
    q1 = calcular_quartil(valores, 1)
    q3 = calcular_quartil(valores, 3)
    
    fiq = q3 - q1 
    
    limite_inferior = q1 - 1.5 * fiq
    limite_superior = q3 + 1.5 * fiq
    
    return limite_inferior, limite_superior


def calcular_mediana(valores):
    return calcular_quartil(valores, 2)

def calcular_assimetria_pearson(valores):
    n = len(valores)
    if n <= 1:
        return 0
        
    media = calcular_media(valores)
    mediana = calcular_mediana(valores)
    desvio = calcular_desvio_padrao(valores)
    
    if desvio == 0:
        return 0
        
    return (3 * (media - mediana)) / desvio


def calcular_correlacao_pearson_xy(x, y):
    #coeficiente de correlação r entre duas variáveis
    n = len(x)
    if n != len(y) or n <= 1:
        return 0
        
    soma_x = sum(x)
    soma_y = sum(y)
    soma_xy = sum(x[i] * y[i] for i in range(n))
    soma_x_quad = sum(xi ** 2 for xi in x)
    soma_y_quad = sum(yi ** 2 for yi in y)
    
    numerador = soma_xy - ((soma_x * soma_y) / n)
    
    termo_x = soma_x_quad - ((soma_x ** 2) / n)
    termo_y = soma_y_quad - ((soma_y ** 2) / n)
    denominador = math.sqrt(termo_x * termo_y)
    
    if denominador == 0:
        return 0
        
    return numerador / denominador


def calcular_regressao_linear(x, y):
    #coeficientes 'a' e 'b' da equação da reta y = a + bx
    n = len(x)
    if n != len(y) or n <= 1:
        return 0, 0
        
    soma_x = sum(x)
    soma_y = sum(y)
    soma_xy = sum(x[i] * y[i] for i in range(n))
    soma_x_quad = sum(xi ** 2 for xi in x)
    
    media_x = calcular_media(x)
    media_y = calcular_media(y)
    
    #coeficiente angular 'b'
    numerador_b = soma_xy - ((soma_x * soma_y) / n)
    denominador_b = soma_x_quad - ((soma_x ** 2) / n)
    
    if denominador_b == 0:
        b = 0
    else:
        b = numerador_b / denominador_b
        
    #coeficiente linear 'a'
    a = media_y - (b * media_x)
    
    return a, b



def calcular_qui_quadrado(matriz_observada):
    #calculo do teste qui quadrado (X^2)
    linhas = len(matriz_observada)
    colunas = len(matriz_observada[0])
    
    #totais
    total_geral = sum(sum(linha) for linha in matriz_observada)
    totais_linhas = [sum(linha) for linha in matriz_observada]
    
    totais_colunas = []
    for j in range(colunas):
        soma_coluna = sum(matriz_observada[i][j] for i in range(linhas))
        totais_colunas.append(soma_coluna)
        
    #somatório de (O - E)^2 / E
    qui_quadrado = 0
    for i in range(linhas):
        for j in range(colunas):
            #frequência esperada (Ei)
            esperado = (totais_linhas[i] * totais_colunas[j]) / total_geral
            
            if esperado == 0:
                continue
                
            #frequência observada (Oi)
            observado = matriz_observada[i][j]
            
            qui_quadrado += ((observado - esperado) ** 2) / esperado
            
    #grau de liberdade
    gl = (linhas - 1) * (colunas - 1)
    
    return qui_quadrado, gl