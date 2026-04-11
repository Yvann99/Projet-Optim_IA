
# Projet Optim_IA : Plateforme d'Analyse et de Calibration de Dérivés Crypto

Ce projet vise à construire un pipeline complet de récupération, de traitement et de modélisation de données d'options sur crypto-actifs (BTC) en provenance de l'exchange Deribit. L'objectif final est la calibration de la surface de volatilité via le modèle SSVI.

## Architecture du Projet

Le projet est structuré de manière modulaire pour garantir la robustesse des calculs :

* **src/data_process.py** : Pipeline d'ingestion et de nettoyage des données.
* **src/rates_model.py** : Extraction des taux implicites et modèle Nelson-Siegel.
* **main.py** : Point d'entrée orchestrant les différentes phases du projet.

---

## Phase 1 : Pipeline de Données et Nettoyage Robustes

La première phase a consisté à transformer les flux de données bruts de l'API Deribit en un dataset structuré exploitable pour la finance quantitative.

### Optimisations et Corrections Techniques
* **Migration vers get_book_summary_by_currency** : Optimisation des appels API pour récupérer l'ensemble du carnet d'ordres en une seule requête, évitant ainsi les limitations de débit (Rate Limiting).
* **Standardisation du Mid-Price** : Calcul systématique du prix moyen entre le Best Bid et le Best Ask pour neutraliser le biais du Market Maker et obtenir une valeur théorique juste.
* **Filtrage par Liquidité** : Exclusion stricte des instruments présentant un spread bid-ask supérieur à 25% du mid-price, garantissant que les modèles ne sont pas calibrés sur du bruit de marché.
* **Validation de Non-Arbitrage Statistique** : Implémentation d'un filtre vérifiant que la prime de l'option est supérieure ou égale à sa valeur intrinsèque. Cette étape est critique pour la convergence des algorithmes de calcul de volatilité implicite.

---

## Phase 2 : Reconstruction de la Structure par Terme des Taux

En l'absence de taux sans risque officiel pour les crypto-actifs, nous reconstruisons une courbe de taux implicite basée sur les prix du marché.

### Méthodologie et Résolution des Anomalies
* **Parité Call-Put** : Extraction des taux pour chaque maturité disponible en utilisant la relation de non-arbitrage entre les prix des Calls, des Puts et des Futures.
* **Correction d'Homogénéité des Unités** : Les options Deribit étant cotées en BTC et les Strikes en USD, nous avons intégré une conversion dynamique des primes en USD. Cette correction a permis de stabiliser l'échelle des taux, passant de valeurs brutes erronées à des pourcentages cohérents.
* **Modèle de Nelson-Siegel** : Application d'un lissage paramétrique pour transformer des points de données discrets en une courbe de taux continue. Cela permet l'interpolation des taux pour n'importe quelle maturité intermédiaire.
* **Visualisation Empirique** : Génération de graphiques comparant les taux bruts du marché aux valeurs lissées du modèle pour valider la pertinence de la calibration.

---

## Guide d'Utilisation

### Prérequis
* Python 3.10+
* Bibliothèques : pandas, numpy, scipy, matplotlib, requests

### Exécution
Pour lancer le pipeline complet et générer la courbe des taux :
```bash
python main.py
```

Les graphiques de diagnostic sont automatiquement sauvegardés à la racine du projet sous le nom courbe_taux_nelson_siegel.png.

Analyse de Performance et Indicateurs de Risque
L'analyse d'une ligne d'option extraite du pipeline (Strike 71 000 USD, type Call) permet de valider la robustesse du modèle et d'en extraire les métriques de risque opérationnelles.

1. Convergence et Fidélité du Modèle

L'écart entre le prix de marché (1670.78 USD) et le prix théorique calculé via la surface SSVI (1670.86 USD) est inférieur à 0.01%. Cette convergence démontre que :

Le modèle Nelson-Siegel fournit une structure de taux cohérente avec le forward du marché.

La calibration des paramètres ρ (asymétrie) et ϕ (courbure) du modèle SSVI capture précisément la dynamique locale de la volatilité.

L'extraction de la volatilité implicite (IV) par l'algorithme hybride (Newton-Raphson/Dichotomie) assure une précision élevée (IV Market 198.80% vs IV SSVI 198.81%).

2. Interprétation des Sensibilités (Grecques)

Le calcul des grecques sur une surface lissée permet d'obtenir des indicateurs de gestion de risque stables, essentiels pour une stratégie de hedging :

Delta (Δ=0.38) : L'instrument présente une sensibilité directionnelle de 38%. Mathématiquement, cela indique une probabilité approchée de 38% que l'option expire "dans la monnaie". L'option est actuellement Out-Of-The-Money (OTM), le prix spot étant inférieur au strike ajusté du coût de portage.

Gamma (Γ=5.76×10 
−5
 ) : La faible valeur nominale est corrélée au prix élevé du sous-jacent (BTC). Toutefois, cette courbure indique qu'une variation de 1000 USD du Bitcoin modifierait le Delta de 0.05, nécessitant un rééquilibrage fréquent du portefeuille de couverture.

Vega (ν=12.68) : Une augmentation de 1% de la volatilité implicite accroît la valeur de l'option de 12.68 USD. Dans un régime de haute volatilité (~200%), la sensibilité au Vega est un facteur de risque prédominant.

Theta (Θ=−208.34) : L'érosion temporelle est particulièrement marquée. L'option perd environ 208 USD de valeur par jour, ce qui est caractéristique d'une option proche de son échéance (Gamma élevé, Theta élevé).

3. Conclusion Technique

L'alignement quasi parfait entre les données brutes et les données modélisées confirme la validité de l'approche paramétrique. Le passage d'une donnée transactionnelle (prix Deribit) à une donnée analytique (grecques SSVI) permet désormais d'envisager des stratégies de couverture dynamique (Delta-Hedging) ou d'arbitrage de volatilité.