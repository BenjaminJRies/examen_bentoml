# Examen BentoML

Ce repertoire contient l'architecture basique afin de rendre l'évaluation pour l'examen BentoML.

Vous êtes libres d'ajouter d'autres dossiers ou fichiers si vous jugez utile de le faire.

Voici comment est construit le dossier de rendu de l'examen:

```bash       
├── examen_bentoml          
│   ├── data       
│   │   ├── processed      
│   │   └── raw           
│   ├── models      
│   ├── src       
│   └── README.md
```

## Setup Instructions

Afin de pouvoir commencer le projet vous devez suivre les étapes suivantes:

- Forker le projet sur votre compte github
- Cloner le projet sur votre machine
- Créer un environnement virtuel:
  ```bash
  python3 -m venv bentoml_env
  source bentoml_env/bin/activate  # Linux/Mac
  # ou bentoml_env\Scripts\activate  # Windows
  ```
- Installer les dépendances:
  ```bash
  pip install -r requirements.txt
  ```
- Récuperer le jeu de données à partir du lien suivant: [Lien de téléchargement](https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv)

## Dataset

The dataset contains information about student admissions to universities with the following variables:

- **GRE Score**: Score obtained on the GRE test (scored out of 340)
- **TOEFL Score**: Score obtained on the TOEFL test (scored out of 120)
- **University Rating**: University rating (scored out of 5)
- **SOP**: Statement of Purpose (scored out of 5)
- **LOR**: Letter of Recommendation (scored out of 5)
- **CGPA**: Cumulative Grade Point Average (scored out of 10)
- **Research**: Research experience (0 or 1)
- **Chance of Admit**: Chance of admission (scored out of 1)

Bon travail!
