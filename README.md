# Black Face

Application Streamlit **100 % gratuite et locale** qui :

1. **Détecte les contenus générés par IA** : images, vidéos, textes, PDF et DOCX.
2. **Vérifie les informations** en les comparant à des sources à haute fiabilité
   (fact-checkeurs, agences de presse, médias de référence, institutions, Wikipédia).

Aucune API payante : tous les modèles tournent sur l'ordinateur, sans carte graphique.

## Comment ça marche

### Détection IA

| Indice | Images | Vidéos | Textes / documents |
|---|---|---|---|
| Métadonnées (IPTC « trainedAlgorithmicMedia », C2PA, nom d'outil IA, paramètres Stable Diffusion) | ✓ | ✓ | ✓ (propriétés PDF/DOCX) |
| Modèle spécialisé | `haywoodsloan/ai-image-detector-deploy` | même modèle sur 12 images extraites | `fakespot-ai/roberta-base-ai-text-detection-v1` |

Une trace explicite de génération IA dans les métadonnées l'emporte sur le reste. L'absence
de métadonnées ne prouve rien : elles sont souvent effacées par les réseaux sociaux.

### Vérification des faits

1. **Affirmations** : les phrases factuelles (chiffres, dates, noms propres, verbes comme
   « a annoncé ») sont repérées par des règles linguistiques. Les opinions sont écartées.
   On peut aussi saisir directement l'affirmation à vérifier.
2. **Recherche** dans des sources fiables uniquement :
   - **Wikipédia** : toujours actif, sans clé ;
   - **Tavily** : web limité à la liste blanche de
     [`sources.py`](blackface/verification/sources.py) (clé gratuite) ;
   - **Google Fact Check** : verdicts publiés par les fact-checkeurs, comme l'AFP ou Snopes (clé gratuite).
3. **Comparaison** : le modèle d'inférence textuelle multilingue
   `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` lit chaque source et décide si
   elle **confirme**, **contredit** ou **ne dit rien** de l'affirmation. Le verdict d'un
   fact-checkeur sur la même affirmation est prioritaire.
4. **Jugement** : les sources sont pondérées selon leur catégorie (un fact-checkeur compte plus
   qu'une encyclopédie). Verdicts possibles : Vrai, Plutôt vrai, Trompeur, Sources divergentes,
   Faux, Non vérifiable.

## Installation (Windows, PowerShell)

```powershell
uv venv --python 3.11 .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

Facultatif, pour élargir la recherche : copier `.env.example` en `.env`, puis renseigner les
clés gratuites.

- `TAVILY_API_KEY` : créer un compte sur https://app.tavily.com (1 000 recherches/mois, sans carte bancaire).
- `GOOGLE_FACTCHECK_API_KEY` : sur https://console.cloud.google.com, activer « Fact Check Tools API »,
  puis créer une clé dans « Identifiants ».

## Lancer

```powershell
streamlit run app.py
```

Au premier lancement, les trois modèles (environ 1,5 Go) sont téléchargés puis mis en cache.
Sur un ordinateur modeste, comptez quelques secondes par affirmation vérifiée.

## Tests

```powershell
python -m pytest tests
```

Les tests simulent les recherches web et les modèles : ils sont rapides et n'utilisent pas Internet.

## Structure

```
app.py                         interface Streamlit
blackface/
  config.py                    configuration (.env, modèles)
  language.py                  outils linguistiques (langue, mots-clés, noms propres)
  report.py                    export PDF / Word / JSON
  detection/
    analyzer.py                choix de l'analyse selon le type de fichier
    metadata.py                traces IA dans les métadonnées
    hf_models.py               modèles Hugging Face (image, texte, inférence textuelle)
    documents.py               extraction PDF / DOCX / TXT
    signals.py                 combinaison des indices en verdict
  verification/
    claims.py                  repérage des affirmations factuelles
    sources.py                 liste blanche des sources et poids par catégorie
    search.py                  Wikipédia, Tavily, Google Fact Check
    factcheck.py               comparaison des sources et verdict
tests/
```

## Limites

- Aucun détecteur d'IA n'est fiable à 100 %, surtout face aux générateurs les plus récents :
  les résultats sont des indices, pas des preuves.
- Le modèle de détection de textes est entraîné surtout sur l'anglais.
- Le modèle de comparaison donne les meilleurs résultats sur des **affirmations courtes et
  précises**. Il comprend moins bien l'ironie ou les nuances qu'un grand modèle de langage.
- Sans clé Tavily ni Google, seule Wikipédia est consultée : les sujets d'actualité récente
  seront souvent « Non vérifiables ».
- Une information absente des sources fiables est classée « Non vérifiable », pas « Fausse ».
