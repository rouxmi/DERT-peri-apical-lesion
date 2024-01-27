# Fine-tunning de DETR pour la prédiction de lésion péri-apicale

## Description

Baser sur le travail de NOUAMANE TAZI: https://www.kaggle.com/code/nouamane/fine-tuning-detr-for-license-plates-detection

L'application est conçue pour convertir des fichiers DICOM au format PNG et lire les fichiers xml contenant le contourage des lésions pour déterminer des boundings boxs pour la détection de ses lésions sur des photos panoramiques dentaires. Le modèle DETR de facebook est ensuite entrainé pour mieux détecter ses lésions et sauvegarder en tant que `detr`

Le script montre aussi un exemple avant le fine-tune pour s'assurer que les données sont bien import et un résultat pour donner une idée des résultats.

## Fonctionnalités

- Conversion DICOM vers PNG
- Détection d'objets en utilisant un modèle YOLO custom
- Création de fichiers GSPS basés sur les prédictions de boîtes englobantes

## Configuration requise

- Python 3.7 ou version ultérieure
- Bibliothèques principales : pytorch, cv2, numpy, pydicom
- Détails des librairies et des versions dans `requirements.txt`

## Installation

1. Clonez le dépôt sur votre machine locale.
2. Installez les bibliothèques Python requises en utilisant la commande `pip install -r requirements.txt`.

## Utilisation

1. Placez vos fichiers DICOM dans le répertoire `/images` et les contourages en .xml dans `/contourages` ou dans un autre dossier si vous avez modifier le `config.ini`
2. Le script `convert_and_split_data.py` permet de générer le bon format des données donc éxécuter celui ci en premier.
3. Le script principal `python main.py` permet lui de fine-tune DETR avec les fichiers répartis par `convert_and_split_data.py`.
4. Le modèle final sera enregistré dans un dossier à la racine de l'éxécution avec le nom `detr`

## Configuration

Tout les paramètres de l'application peuvent être modifier dans le fichier `config.ini` comme le learning rate des différents éléments le nom des dossiers dans lequel se trouve les images et les contourages.

Dans le `main.py`, la condition d'arrêt d'entrainement peut être modifier à votre guise, l'explication est donné dans le fichier.

## Contribution

Les contributions sont les bienvenues. Veuillez soumettre une pull request avec vos modifications.

## Support

Si vous avez des questions ou des problèmes, veuillez ouvrir une issue sur le dépôt GitHub.
