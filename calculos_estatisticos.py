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