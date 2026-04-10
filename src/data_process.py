#On commence par aller chercher les données Deribit via son API

import requests #Pour la connection avec l'API de Deribit
import pandas as pd

def get_active_options(currency="BTC"):
    url = f"https://www.deribit.com/api/v2/public/get_instruments?currency={currency}&kind=option&expired=false"
    response = requests.get(url).json()
    
    # On transforme le résultat en DataFrame pour plus de facilité
    df_instruments = pd.DataFrame(response['result'])
    return df_instruments[['instrument_name', 'strike', 'expiration_timestamp', 'option_type']]

def get_order_book(instrument_name):
    #Récupère le meilleur cours acheteur (bid) et vendeur (ask)
    url = f"https://www.deribit.com/api/v2/public/get_order_book?instrument_name={instrument_name}"
    response = requests.get(url).json()
    
    data = response['result']
    bid = data.get('best_bid_price', 0)
    ask = data.get('best_ask_price', 0)
    
    return bid, ask

def process_market_data(df_instruments):
    #Parcourt les instruments pour calculer le mid-price et filtrer selon le spread.
    results = []
    
    for _, row in df_instruments.iterrows():
        name = row['instrument_name']
        bid, ask = get_order_book(name) # Utilise ta fonction précédente
        
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
            spread = (ask - bid) / mid
            
            # Conservation des données pour filtrage
            results.append({
                'instrument_name': name,
                'strike': row['strike'],
                'expiry': row['expiration_timestamp'],
                'type': row['option_type'],
                'bid': bid,
                'ask': ask,
                'mid_price': mid,
                'spread': spread
            })
    
    df_market = pd.DataFrame(results)
    
    # Filtre : Éliminer si spread > 25% 
    df_cleaned = df_market[df_market['spread'] <= 0.25].copy()
    
    return df_cleaned

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Liste de 6 valeurs représentatives du CAC 40
tickers = ['MC.PA', 'TTE.PA', 'SAN.PA', 'AIR.PA', 'OR.PA', 'BNP.PA']
data = yf.download(tickers, period="1y", interval="1d")['Close']

# 2. Calcul des rendements logarithmiques et nettoyage
returns = np.log(data / data.shift(1)).dropna()

# 3. Standardisation (moyenne = 0, écart-type = 1)
# Très important en ACP pour ne pas que les actions chères dominent
scaler = StandardScaler()
scaled_returns = scaler.fit_transform(returns)

# 4. Calcul de l'ACP
pca = PCA(n_components=2) # On garde les 2 axes principaux
pca_data = pca.fit_transform(scaled_returns)

# --- VISUALISATION ---
plt.figure(figsize=(14, 6))

# A. Graphique des Éboulis (Scree Plot)
plt.subplot(1, 2, 1)
plt.bar(range(1, 3), pca.explained_variance_ratio_, color='skyblue', label='Variance Expliquée')
plt.ylabel('Ratio de Variance')
plt.xlabel('Composantes Principales')
plt.title('Importance des Composantes')
plt.xticks([1, 2], ['PC1', 'PC2'])

# B. Cercle des Corrélations (Loadings)
# Il montre comment chaque action contribue aux composantes
plt.subplot(1, 2, 2)
for i, ticker in enumerate(tickers):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
              head_width=0.03, head_length=0.03, fc='red', ec='red')
    plt.text(pca.components_[0, i]*1.2, pca.components_[1, i]*1.2, ticker, fontsize=12)

circle = plt.Circle((0,0), 1, color='blue', fill=False, linestyle='--')
plt.gca().add_artist(circle)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.title('Cercle des Corrélations (Loadings)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

plt.tight_layout()
plt.show()