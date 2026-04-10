import requests
import pandas as pd
import numpy as np

def get_deribit_data(currency="BTC"):
    #Récupération des instruments actifs
    instr_url = f"https://www.deribit.com/api/v2/public/get_instruments?currency={currency}&expired=false"
    instruments = pd.DataFrame(requests.get(instr_url).json()['result'])
    
    #Récupération des prix
    book_url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={currency}"
    books = pd.DataFrame(requests.get(book_url).json()['result'])

    #Ajustement du noms des colonnes sur Déribit
    rename_dict = {
        'bid_price': 'best_bid_price',
        'ask_price': 'best_ask_price'
    }
    books = books.rename(columns=rename_dict)
    
    # On définit les colonnes qu'on veut garder pour le merge
    available_cols = [c for c in ['instrument_name', 'best_bid_price', 'best_ask_price', 'mid_price'] if c in books.columns]
    
    # Fusion des données
    df = pd.merge(instruments, books[available_cols], on='instrument_name')
    
    df_options = df[df['kind'] == 'option'].copy()
    df_futures = df[df['kind'] == 'future'].copy()
    
    return df_options, df_futures

def process_and_filter_options(df_options, spot_price):
    """
    Nettoyage du dataset et vérification d'arbitrage.
    """
    # Calcul du mid-price si absent et du spread bid-ask relatif
    df_options['mid_price'] = (df_options['best_bid_price'] + df_options['best_ask_price']) / 2
    df_options['spread'] = (df_options['best_ask_price'] - df_options['best_bid_price']) / df_options['mid_price']
    
    # Filtre 1 : Exclusion des spreads > 25% 
    df_filtered = df_options[df_options['spread'] <= 0.25].copy()
    
    # Calcul de la valeur intrinsèque pour l'arbitrage statistique 
    # Call : max(0, S - K) | Put : max(0, K - S)
    df_filtered['intrinsic_val'] = np.where(
        df_filtered['option_type'] == 'call',
        (spot_price - df_filtered['strike']).clip(lower=0),
        (df_filtered['strike'] - spot_price).clip(lower=0)
    )
    
    # Filtre 2 : Vérification arbitrage (Prime >= Valeur Intrinsèque) 
    df_filtered = df_filtered[df_filtered['mid_price'] >= df_filtered['intrinsic_val']]
    
    # Filtre 3 : Élimination des options trop loin hors/dans la monnaie [cite: 15]
    # On garde les strikes entre 50% et 150% du spot (ajustable)
    df_filtered = df_filtered[
        (df_filtered['strike'] > spot_price * 0.5) & 
        (df_filtered['strike'] < spot_price * 1.5)
    ]
    
    return df_filtered
