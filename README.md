# Cémantix Discord Bot

Bot Discord qui recrée le principe de [Cémantix](https://cemantix.certitudes.org/) (jeu à la Semantle : deviner un mot grâce à un score de proximité sémantique "hot/cold") pour jouer entre amis sur un serveur Discord.

## Fonctionnement

Le bot propose **deux modes de jeu** :

### Mode Local (par défaut)
- Chaque nuit à minuit, le bot tire automatiquement un nouveau nom commun français depuis une liste pré-filtrée (dico_mm.txt ou dico_ms.txt), en excluant les mots déjà utilisés.
- Les joueurs tapent un mot dans le salon configuré. Le bot calcule la similarité cosinus entre ce mot et le mot du jour via un modèle d'embeddings [word2vec français (frWac2Vec)](https://fauconnier.github.io/#data), et répond avec un score en % et une barre de progression.
- Un top 10 des mots les plus proches proposés dans la journée s'affiche à chaque tentative.
- Stats persistées par joueur (victoires, nombre moyen de coups).

### Mode API (Infomaniak)
- Utilise l'API Infomaniak pour obtenir des embeddings de mots en temps réel ou pré-calculés.
- Nécessite un vocabulaire local (fournis dans `data/vocab/`) et des identifiants API Infomaniak.
- Permet d'avoir un vocabulaire propre et contrôlé, sans les problèmes de bruit du modèle word2vec local.
- Le mode API doit être activé avec `/set_mode api` et les embeddings doivent être pré-calculés avec `/ingest_vocab`.

## Commandes

### Commandes communes
| Commande | Description |
|---|---|
| `/start` | État de la partie du jour (mode local) |
| `/start_api` | État de la partie du jour (mode API) |
| `/top` | Classement global des joueurs |
| `/top_local` | Classement des joueurs (mode local) |
| `/top_api` | Classement des joueurs (mode API) |
| `/profil` | Stats personnelles (tous modes) |
| `/profil_local` | Stats personnelles (mode local) |
| `/profil_api` | Stats personnelles (mode API) |

### Commandes de configuration
| Commande | Description |
|---|---|
| `/set_mode` | Change le mode de jeu (local/api) |
| `/mode_status` | Affiche le mode actuel et la disponibilité |
| `/ingest_vocab` | **[Admin]** Lance l'ingestion du vocabulaire API |

### Commandes de jeu
- Tapez simplement un mot dans le salon pour jouer dans le mode actuel.

## Préréquis

- Python 3.11+
- Un Raspberry Pi (ou toute machine Linux) allumé en continu
- Une application Discord Bot ([Developer Portal](https://discord.com/developers/applications)) avec l'intent **Message Content** activé

```bash
# Pour le mode local
pip install discord.py gensim numpy python-dotenv aiohttp

# Pour le mode API (aiohttp est nécessaire pour les appels API)
pip install discord.py numpy python-dotenv aiohttp
```

## Installation

1. Clone ce repo sur ta machine :
   ```bash
   git clone <url_du_repo> cemantix-bot
   cd cemantix-bot
   ```

2. **Pour le mode local** : Télécharge le modèle de vecteurs français depuis [fauconnier.github.io/#data](https://fauconnier.github.io/#data) : prends `frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin` (120 Mo, cbow, 200 dimensions, cutoff 100), place-le à la racine du repo. **Ce fichier n'est pas versionné dans git** (trop lourd), il faut le retélécharger sur chaque nouvelle machine.

3. **Pour le mode API** : Prépare ton vocabulaire dans `data/vocab/vocab.txt` (voir [Sourcing du Vocabulaire](#sourcing-du-vocabulaire) ci-dessous).

4. Copie `.env.example` vers `.env` et remplis les valeurs (ce fichier n'est jamais commité, voir `.gitignore`) :
   - `DISCORD_TOKEN` : le token de ton bot (Developer Portal → Bot → Reset Token)
   - `CHANNEL_ID` : l'ID du salon Discord où vous jouez (mode développeur activé → clic droit sur le salon → copier l'identifiant)
   - **Pour le mode API** :
     - `INFOMANIAK_PRODUCT_ID` : L'ID de ton produit Infomaniak AI
     - `INFOMANIAK_API_KEY` : Ta clé API Infomaniak

5. Invite le bot sur ton serveur via OAuth2 URL Generator (scopes `bot` + `applications.commands`, permissions `Send Messages`, `Read Message History`, `View Channel`).

6. Lance le bot :
   ```bash
   # Mode local (par défaut)
   python cemantix_bot.py
   
   # Mode API avec ingestion du vocabulaire
   python cemantix_bot.py --mode api --ingest-vocab
   
   # Mode API sans ingestion (utilise les embeddings existants)
   python cemantix_bot.py --mode api
   ```

   **Note** : La première exécution avec `--ingest-vocab` peut prendre plusieurs minutes selon la taille de ton vocabulaire et les limites de l'API Infomaniak.

## Sourcing du Vocabulaire

Pour le mode API, tu as besoin d'un fichier de vocabulaire français propre. Voici les options recommandées :

### Option 1: Lexique 3.83 (Recommandé)
- **Source**: https://www.lexique.org/
- **Fichier**: `Lexique383.tsv`
- **Format**: Tab-separated avec métadonnées
- **Extraction**:
  ```bash
  # Extraire uniquement les mots (colonne 1) et filtrer
  cut -f1 Lexique383.tsv | grep -E '^[a-z]+$' | grep -v -E '[ -]' > data/vocab/vocab.txt
  ```

### Option 2: Utiliser les fichiers existants
Copie simplement tes fichiers existants :
```bash
cp dico_mm.txt data/vocab/vocab.txt
# ou
cp dico_ms.txt data/vocab/vocab.txt
```

### Option 3: Wiktionnaire
Utilise l'API MediaWiki pour extraire des noms communs français.

### Format attendu
- Un mot par ligne
- Encodage UTF-8
- Mots en minuscules
- Pas d'espaces, tirets ou apostrophes (mots composés exclus)
- Caractères alphabétiques uniquement

Voir `data/vocab/README.md` pour plus de détails.

## Structure des données

### Fichiers d'état
- `state.json` : État du jeu pour le **mode local** (non versionné)
- `state_api.json` : État du jeu pour le **mode API** (non versionné)

Les deux fichiers contiennent :
- `target` : Mot du jour
- `current_date` : Date du mot actuel
- `found` : Si le mot a été trouvé
- `attempts_today` : Nombre de tentatives
- `players` : Stats des joueurs
- `guesses_today` : Scores des propositions du jour
- `mots_utilises` : Historique des mots utilisés
- `neighbors` : Liste des 100 mots les plus proches du mot du jour

### Fichiers du vocabulaire API
- `data/vocab/vocab.txt` : Liste des mots du vocabulaire (un mot par ligne)
- `data/vocab/embeddings.npy` : Embeddings pré-calculés (format NumPy float16)
- `data/vocab/vocab.json` : Métadonnées du vocabulaire
- `data/vocab/word_to_index.json` : Mappage mot → index

Voir `CLAUDE.md` pour le schéma détaillé du state.

## Déploiement en continu (systemd)

Pour que le bot tourne 24/7 et redémarre automatiquement après un reboot du Pi, voir le service systemd `cemantix.service` (non versionné, propre à chaque machine, voir `CLAUDE.md` pour le contenu).

```bash
sudo systemctl status cemantix.service   # vérifier l'état
sudo systemctl restart cemantix.service  # relancer après une modif
journalctl -u cemantix.service -n 50 --no-pager   # voir les derniers logs
```

## Mettre à jour le bot sur le Raspberry Pi

```bash
cd ~/cemantix-bot
git pull
sudo systemctl restart cemantix.service
```

## Accès distant

Le Pi est accessible en SSH via [Tailscale](https://tailscale.com/) depuis n'importe quel réseau (pas seulement le Wi-Fi local), utile pour se connecter avec VS Code Remote-SSH depuis un autre ordinateur.
