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
    Nettoyage complet : Spread, Moneyness et Arbitrage Statistique.
    """
    # A. Calcul du mid-price et du spread [cite: 14, 15]
    df_options['mid_price'] = (df_options['best_bid_price'] + df_options['best_ask_price']) / 2
    df_options['spread'] = (df_options['best_ask_price'] - df_options['best_bid_price']) / df_options['mid_price']
    
    # B. Filtre 1 : Exclusion des spreads > 25% du mid-price 
    df_filtered = df_options[df_options['spread'] <= 0.25].copy()
    
    # C. Calcul de la valeur intrinsèque (Vérification d'arbitrage) 
    # Attention : sur Deribit, si le mid_price est en BTC, multiplie-le par spot_price 
    # pour comparer des USD avec des USD.
    
    # Pour les Calls : max(0, Spot - Strike) 
    is_call = df_filtered['option_type'] == 'call'
    df_filtered.loc[is_call, 'intrinsic_val'] = (spot_price - df_filtered['strike']).clip(lower=0)
    
    # Pour les Puts : max(0, Strike - Spot) 
    is_put = df_filtered['option_type'] == 'put'
    df_filtered.loc[is_put, 'intrinsic_val'] = (df_filtered['strike'] - spot_price).clip(lower=0)
    
    # D. Filtre 2 : Vérification arbitrage (Prime >= Valeur Intrinsèque) 
    # On convertit ici la prime en USD pour la comparaison
    df_filtered['mid_price_usd'] = df_filtered['mid_price'] * spot_price
    df_filtered = df_filtered[df_filtered['mid_price_usd'] >= df_filtered['intrinsic_val']]
    
    return df_filtered
