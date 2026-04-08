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