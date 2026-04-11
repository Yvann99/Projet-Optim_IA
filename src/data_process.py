import requests
import pandas as pd
import numpy as np
import time

def get_deribit_data(currency="BTC"):
    instr_url = f"https://www.deribit.com/api/v2/public/get_instruments?currency={currency}&expired=false"
    instruments = pd.DataFrame(requests.get(instr_url).json()['result'])
    
    book_url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={currency}"
    books = pd.DataFrame(requests.get(book_url).json()['result'])

    rename_dict = {'bid_price': 'best_bid_price', 'ask_price': 'best_ask_price'}
    books = books.rename(columns=rename_dict)
    
    available_cols = [c for c in ['instrument_name', 'best_bid_price', 'best_ask_price', 'mid_price'] if c in books.columns]
    df = pd.merge(instruments, books[available_cols], on='instrument_name')
    
    df_options = df[df['kind'] == 'option'].copy()
    df_futures = df[df['kind'] == 'future'].copy()
    
    return df_options, df_futures

def process_and_filter_options(df_options, spot_price):
    # 1. Calcul du Temps restant (T) en années
    # Deribit donne l'expiration en millisecondes (ms)
    now = time.time() * 1000 
    df_options['T'] = (df_options['expiration_timestamp'] - now) / (1000 * 60 * 60 * 24 * 365)
    
    # Sécurité : on supprime les options déjà expirées ou trop proches ( < 1h)
    df_options = df_options[df_options['T'] > 0.0001].copy()

    # 2. Calcul du mid-price et du spread
    df_options['mid_price'] = (df_options['best_bid_price'] + df_options['best_ask_price']) / 2
    df_options['spread'] = (df_options['best_ask_price'] - df_options['best_bid_price']) / df_options['mid_price']
    
    # 3. Filtre 1 : Spread (25%)
    df_filtered = df_options[df_options['spread'] <= 0.25].copy()
    
    # 4. Calcul de la valeur intrinsèque
    is_call = df_filtered['option_type'] == 'call'
    df_filtered.loc[is_call, 'intrinsic_val'] = (spot_price - df_filtered['strike']).clip(lower=0)
    
    is_put = df_filtered['option_type'] == 'put'
    df_filtered.loc[is_put, 'intrinsic_val'] = (df_filtered['strike'] - spot_price).clip(lower=0)
    
    # 5. Conversion USD et Filtre 2 : Arbitrage
    df_filtered['mid_price_usd'] = df_filtered['mid_price'] * spot_price
    df_filtered = df_filtered[df_filtered['mid_price_usd'] >= df_filtered['intrinsic_val']]
    
    return df_filtered
