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
    """Récupère le meilleur cours acheteur (bid) et vendeur (ask)."""
    url = f"https://www.deribit.com/api/v2/public/get_order_book?instrument_name={instrument_name}"
    response = requests.get(url).json()
    
    data = response['result']
    bid = data.get('best_bid_price', 0)
    ask = data.get('best_ask_price', 0)
    
    return bid, ask