import pandas as pd
import yfinance as yf
import numpy as np
import selenium
import time
from selenium.webdriver.common.by import By
import re
from scipy.stats import norm

def processLine(linha):
    # Processa a primeira parte (antes da vírgula)
    parte1 = linha[0].strip()
    # Extrai o índice (ex: "269ª")
    indice = re.match(r'^\d+ª', parte1).group()
    # Extrai as duas primeiras datas (ex: "19/03/2025" e "20/03/2025")
    datas_parte1 = re.findall(r'\d{2}/\d{2}/\d{4}', parte1)
    data1, data2 = datas_parte1[0], datas_parte1[1] if len(datas_parte1) > 1 else 'n/a'
    
    # Processa a segunda parte (depois da vírgula)
    parte2 = linha[1].strip()
    # Extrai a terceira data (ex: "07/05/2025")
    data3 = re.search(r'\d{2}/\d{2}/\d{4}', parte2).group()
    # Extrai os números (ex: "14,25", "1,69", "14,15")
    numeros = re.findall(r'\d+,\d{2}', parte2)
    num1 = numeros[0] if len(numeros) > 0 else 'n/a'
    num2 = numeros[1] if len(numeros) > 1 else 'n/a'
    num3 = numeros[2] if len(numeros) > 2 else 'n/a'
    
    return [indice, data1, data2, data3, num1, 'n/a', num2, num3]

def downloadData(): return yf.download("PETR4.SA", start="2024-12-01", end="2025-05-31")

def historicalVol():
    # baixando os dados da petr4
    data = downloadData()
    returns = np.log(data["Close"] / data["Close"].shift(1)) # log dos retornos
    volDaily = returns.std() # desv pad dos retornos 
    volAnnual = volDaily * np.sqrt(252) # anual = diario * sqrt(dias uteis)... desv. pad med. so 

    print(f"Volatilidade histórica anual: {volAnnual.iloc[0]:.2%}")

    return volAnnual

def intitialPrice(data):

    close = np.asarray([data.loc[date].iloc[0] for date in data.index]) # valores no close 
    open = np.asarray([data.loc[date].iloc[3] for date in data.index]) # valores no open

    time = len(close) # "tempo"... utilizado pra tirar media, so
    avgClose = np.sum(close)/time
    avgOpen = np.sum(open)/time # medias 

    S0 = np.mean([avgClose, avgOpen]) # S0 foi tomado como media da abertura e fechamento

    return S0

def selic():
    driver = selenium.webdriver.Chrome() # inicializacao do driver pra fazer scraping com uma pagina com js

    driver.get("https://www.bcb.gov.br/controleinflacao/historicotaxasjuros") # obtendo pagina 

    time.sleep(2) # delay necessario pra que a pagina carregue antes do scraping
    
    element = driver.find_element(by=By.CLASS_NAME, value = "table") # procurando tabela na pagina
    
    # processamento da tabela/dados 
    splitting = [i.split("-") for i in element.text.split("\n")[7:]] # removendo partes desnecessarias da tabela 
    splitting = [processLine(line) for line in splitting[0:50]] # nao extreaindo todos os dados
    splitting = np.strings.replace(splitting, ",", ".")
    df = pd.DataFrame(splitting)
    df[[4, 6, 7]] = df[[4, 6, 7]].replace(',', '.', regex=True).astype(np.float32)
    splitting = df.values.tolist()

    
    foo = []
    for line in splitting[0:3]: # ultimos 6 meses 
        foo.append(line[-1])
    driver.close()
    return np.mean(foo) # retorna media da selic nos ultimos ~6 meses

def blackScholesCall(S, K, sigma, r, t): # modelo de black scholes
    d1 = (np.log(S/K) + (r + sigma**2/2)*t)/(sigma * t**0.5)
    d2 = d1 - (sigma*t**0.5)
    call = S*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)

    return call