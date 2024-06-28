# Projet MNIST FastAPI

Ce projet implémente un modèle de reconnaissance de chiffres manuscrits (MNIST) avec un backend FastAPI et un frontend Streamlit.

## Installation

1. Clonez ce dépôt
2. Installez les dépendances : `pip install -r requirements.txt`
3. Lancez le backend : `uvicorn src.app.main:app --reload`
4. Lancez le frontend : `streamlit run src/app/frontend.py`

## Utilisation avec Docker

1. Construisez les images : `docker-compose build`
2. Lancez les conteneurs : `docker-compose up`

Le frontend sera accessible sur `http://localhost:8501`
