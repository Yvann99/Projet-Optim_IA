def main():
    print("Hello from projet-optim-ia!")


if __name__ == "__main__":
    main()

from src.data_process import get_active_options, get_order_book, calculate_mid_price

def main():
    print("Démarrage de la récupération des données BTC...")
    
    # 1. Liste des instruments
    options_list = get_active_options("BTC")
    
    # 2. On ne prend que les 10 premières pour tester le code
    test_subset = options_list.head(10).copy()
    
    bids, asks, mids = [], [], []
    
    for name in test_subset['instrument_name']:
        print(f"Récupération des prix pour : {name}")
        bid, ask = get_order_book(name)
        mid = (bid + ask) / 2 # Calcul du mid-price demandé 
        
        bids.append(bid)
        asks.append(ask)
        mids.append(mid)
    
    test_subset['bid'] = bids
    test_subset['ask'] = asks
    test_subset['mid_price'] = mids
    
    print("\nAperçu des données récupérées :")
    print(test_subset.head())

if __name__ == "__main__":
    main()