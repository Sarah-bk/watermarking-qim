# watermarking-qim
Sécurisation des images 2D par tatouage numérique invisible et robuste basé sur la méthode QIM dans le domaine DCT. Implémentation Python avec simulation d'attaques et évaluation de la qualité.

# 🖼️ Sécurisation d'images 2D par tatouage numérique – Méthode QIM

> Un système robuste de tatouage numérique d'images basé sur la **Quantification par Index Modulation (QIM)** dans le **domaine DCT**, implémenté en Python.  
> Conçu pour la protection de la propriété intellectuelle, la vérification de propriété et l'authentification d'images.

---

## 🎯 Contexte et problématique

Avec la diffusion massive d'images numériques :

- Le vol de propriété intellectuelle est fréquent  
- Les images sont facilement modifiables  
- L'authenticité est difficile à vérifier  

👉 **Comment insérer une marque invisible et robuste dans une image 2D en utilisant la quantification contrôlée (QIM) tout en conservant une bonne qualité visuelle ?**

---

## 🧠 Solution proposée

Ce projet implémente un système complet de **tatouage aveugle** (blind watermarking) :

- ✅ Insertion d'une marque binaire dans les coefficients DCT de moyenne fréquence  
- ✅ Utilisation d'une **clé secrète** pour la sélection pseudo-aléatoire des coefficients  
- ✅ Résistance aux attaques courantes (bruit, compression JPEG)  
- ✅ Extraction et évaluation de la marque sans l'image originale

---

## 🏗️ Architecture du système

1. Chargement de l'image hôte  
2. Application de la **Transformée en Cosinus Discrète (DCT) 2D**  
3. Insertion de la marque par **Quantification par Index Modulation (QIM)**  
4. Simulation d'attaques (bruit gaussien, JPEG, etc.)  
5. Extraction de la marque et évaluation  
6. Reconstruction de l'image tatouée

---

## 🧰 Technologies et bibliothèques

- **Python 3.10+**
- `opencv-python` – chargement et traitement d'images  
- `numpy` – calcul matriciel  
- `scipy` – calcul de la DCT  
- `matplotlib` – visualisation des résultats  
- `scikit-image` – calcul du PSNR  

```bash
pip install opencv-python numpy matplotlib scikit-image scipy
