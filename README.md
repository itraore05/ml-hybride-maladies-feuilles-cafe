# ☕ Détection des maladies des feuilles de café — Apprentissage hybride (non supervisé + supervisé)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-LightGBM-2E7D32?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-interprétabilité-8E24AA?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Master](https://img.shields.io/badge/Master%201-Big%20Data%20%26%20IA-blue?style=flat)

> Pipeline de Machine Learning **hybride** qui détecte la **rouille du caféier** à partir de données multispectrales : le clustering découvre d'abord la structure cachée des données, puis ses résultats deviennent de nouvelles variables pour les modèles de classification.
> Projet académique — Master 1 Big Data Analytics & IA (UVCI), UE Machine Learning, 2026.

---

## 🎯 Problématique

La détection précoce des maladies du caféier est un enjeu clé de l'agriculture de précision. Les données multispectrales (bandes Blue, Green, Red, RedEdge, NIR et indices de végétation) permettent de repérer la maladie avant qu'elle soit visible, mais :

- les signatures spectrales des feuilles **saines** et **malades** se chevauchent ;
- les variables sont **fortement corrélées** (multicolinéarité confirmée par le VIF) ;
- les classes sont **déséquilibrées**.

**Idée du projet :** utiliser l'apprentissage non supervisé comme « guide » pour les modèles supervisés.

## 🧭 Pipeline

```
Données multispectrales
   │
   ├─▶ 1. EDA : distributions, corrélations Pearson / Spearman, VIF, normalisation
   ├─▶ 2. Réduction de dimension : PCA, t-SNE, UMAP
   ├─▶ 3. Clustering multi-modèles : K-Means, GMM, Spectral Clustering, DBSCAN
   ├─▶ 4. Consensus : matrice de co-association + clustering hiérarchique
   ├─▶ 5. Feature engineering hybride : cluster de consensus, probabilités GMM,
   │      distances aux centroïdes, coordonnées PCA / UMAP
   ├─▶ 6. Classification : Random Forest, XGBoost, LightGBM, SVM RBF, MLP
   │      (scénario 1 : données brutes  vs  scénario 2 : données enrichies)
   ├─▶ 7. Pseudo-labeling (seuil de confiance > 95 %)
   └─▶ 8. Validation statistique (Friedman, Wilcoxon) et interprétabilité (SHAP)
```

## 📈 Résultats clés

**Clustering (sans utiliser les étiquettes)**

| Modèle | Point fort |
|---|---|
| **GMM** | Meilleure correspondance avec la réalité terrain : **ARI 0,70 · NMI 0,62** |
| Spectral Clustering | Silhouette très élevée (0,83) mais ARI proche de 0 : des groupes « beaux » géométriquement, sans lien avec la maladie |
| K-Means | Bon compromis géométrique, moins précis que le GMM |

Le cluster de consensus isole un groupe **100 % sain** (190 feuilles, 0 malade) : la machine retrouve la structure biologique sans avoir vu les étiquettes.

**Classification supervisée (données enrichies)**

| Modèle | Accuracy | MCC |
|---|---|---|
| **Random Forest** | **92,9 %** | **0,80** |
| **LightGBM** | **92,9 %** | **0,80** |
| XGBoost | 92,4 % | — |
| SVM RBF, MLP | Faibles sans optimisation des hyperparamètres | — |

- **Friedman** (p ≈ 0) : les écarts entre modèles sont statistiquement significatifs.
- **Wilcoxon** Random Forest vs LightGBM (p = 0,97) : performances équivalentes.

**Interprétabilité (SHAP et importance des variables)**

Les 3 variables issues du clustering (probabilités GMM et cluster de consensus) représentent **plus de 32 %** de l'importance totale. Le modèle décide d'abord grâce à la structure latente, puis affine avec les bandes spectrales et les indices de végétation.

## 🧠 Regard critique

- **Valeur de l'hybridation** : le modèle ne travaille plus sur des données brutes, mais sur des représentations déjà organisées, ce qui stabilise les prédictions.
- **Limites** : le modèle utilisé pour SHAP est entraîné sur toutes les données (léger biais d'interprétation) ; le pseudo-labeling n'a pas amélioré les scores, car le seuil strict ne réinjecte que les cas faciles.
- **Pistes** : descripteurs de texture (GLCM, Haralick), modèles tabulaires profonds (TabNet).

## 📁 Structure du dépôt

```
ml-hybride-maladies-feuilles-cafe/
├── notebook/
│   └── classification_maladies_feuilles_cafe.ipynb   # Pipeline complet
├── rapport/
│   └── rapport_scientifique.pdf                       # Rapport de 17 pages
└── README.md
```

> ℹ️ Le jeu de données a été fourni dans le cadre du cours et n'est pas publié ici. Pour exécuter le notebook, placer le fichier de données à côté du notebook.

## ▶️ Utilisation

```bash
git clone https://github.com/itraore05/ml-hybride-maladies-feuilles-cafe.git
pip install pandas numpy scikit-learn xgboost lightgbm umap-learn shap matplotlib seaborn scipy openpyxl
```

Ouvrir le notebook dans Jupyter ou Google Colab et exécuter les cellules dans l'ordre.

## 💡 Compétences démontrées

- Analyse exploratoire et diagnostic de multicolinéarité
- Clustering multi-modèles et **consensus clustering**
- **Feature engineering** à partir de modèles non supervisés
- Comparaison rigoureuse de 5 classifieurs et **tests statistiques**
- **Interprétabilité** avec SHAP, et lecture métier (agronomique) des résultats

## 👤 Auteur

**Ismaël Traoré** — Développeur Data & IA
📧 contact.traoreimael@gmail.com · 💼 [LinkedIn](https://www.linkedin.com/in/ismael-traor%C3%A9-m-4007a2225/)
