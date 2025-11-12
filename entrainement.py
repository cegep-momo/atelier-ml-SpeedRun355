import numpy as npm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#Chargement du dataset avec pour séparateur des tabulations
dataset = pd.read_csv("apprentissage/datas/dataset.csv",delimiter='\t')

#Suppression des lignes ayant des valeurs manquantes
dataset = dataset.dropna(axis=0, how='any')

X = dataset.iloc[:, 5:12].values
y = dataset.iloc[:, 16].values

#Construction du jeu d'entrainement et du jeu de tests
from sklearn.model_selection import train_test_split
X_APPRENTISSAGE, X_VALIDATION, Y_APPRENTISSAGE, Y_VALIDATION = train_test_split(X, y, test_size = 0.2, random_state = 0)

#---- ALGORITHME 1: REGRESSION LINEAIRE -----
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

#Choix de l'algorithme
from sklearn.ensemble import RandomForestRegressor
algorithme = RandomForestRegressor()
algorithme.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = algorithme.predict(X_VALIDATION)
precision_apprentissage = algorithme.score(X_APPRENTISSAGE,Y_APPRENTISSAGE)
precision = r2_score(Y_VALIDATION, predictions)

print(">> ----------- FORETS ALEATOIRES -----------")
print(">> Precision = "+str(precision))
print("------------------------------------------")

fichier = 'modele/modele_pokemon.mod'
joblib.dump(algorithme, fichier)