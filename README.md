# Cémantix Discord Bot

Bot Discord qui recrée le principe de [Cémantix](https://cemantix.certitudes.org/) — un jeu où il faut deviner un mot grâce à un score de proximité sémantique (style "hot/cold").

## Comment ça marche

- Chaque nuit à minuit, le bot tire automatiquement un nouveau nom commun français
- Les joueurs tapent un mot dans le salon Discord configuré
- Le bot répond avec un score de similarité sémantique et une barre de progression
- Un top 10 des mots les plus proches proposés s'affiche à chaque tentative
- Stats persistées par joueur (victoires, nombre moyen de coups)

## Prérequis

- Python 3.11+
- Une machine (PC, Raspberry Pi, serveur) allumée en continu
- Une application Discord Bot ([Developer Portal](https://discord.com/developers/applications)) avec l'intent **Message Content** activé

---

## Installation avec Docker (Recommandé)

La méthode la plus simple — **tout est automatique** !

### Étapes

1. **Clone le repo** :
   ```bash
   git clone <url_du_repo> cemantix-bot
   cd cemantix-bot
   ```

2. **Configure le bot** :
   ```bash
   cp .env.example .env
   # Édite .env avec ton éditeur préféré
   ```
   
   Dans `.env`, mets :
   ```ini
   DISCORD_TOKEN=ton_token_de_bot_discord
   CHANNEL_ID=l_id_du_salon_ou_tu_veux_jouer
   ```

3. **Lance le bot** :
   ```bash
   docker-compose up -d
   ```
   
   > ✨ **Magie** : Au premier lancement, le conteneur télécharge automatiquement :
   > - Le modèle word2vec **frWac 500-dim skip-gram** (298 Mo)
   > - Les dictionnaires français (dico_mm.txt, dico_ms.txt)
   > - Toutes les dépendances Python

4. **Vérifie que ça marche** :
   ```bash
   docker-compose logs -f cemantix
   ```
   Attends de voir `Connecté en tant que [nom_du_bot]` avant de tester sur Discord.

### Commandes Docker utiles

| Commande | Description |
|---|---|
| `docker-compose up -d` | Lance le bot en arrière-plan |
| `docker-compose down` | Arrête le bot |
| `docker-compose restart` | Redémarre le bot |
| `docker-compose logs -f` | Affiche les logs en temps réel |
| `docker-compose pull && docker-compose up -d --build` | Met à jour après un `git pull` |

---

## Installation manuelle (sans Docker)

Si tu préfères ne pas utiliser Docker :

1. **Clone le repo** :
   ```bash
   git clone <url_du_repo> cemantix-bot
   cd cemantix-bot
   ```

2. **Installe les dépendances** :
   ```bash
   pip install discord.py gensim numpy python-dotenv
   ```

3. **Télécharge le modèle** (298 Mo) :
   ```bash
   # Modèle 500-dim skip-gram avec cutoff 100
   wget https://embeddings.net/embeddings/frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin
   ```
   
   *Alternative* : Tu peux aussi utiliser un autre modèle de [fauconnier.github.io](https://fauconnier.github.io/#data). Dans ce cas, modifie `MODEL_PATH` dans `cemantix_bot.py`.

4. **Télécharge les dictionnaires** :
   ```bash
   wget https://raw.githubusercontent.com/TikSL/semanTikSl/main/dico_mm.txt
   wget https://raw.githubusercontent.com/TikSL/semanTikSl/main/dico_ms.txt
   ```

5. **Configure le bot** :
   ```bash
   cp .env.example .env
   # Édite .env avec DISCORD_TOKEN et CHANNEL_ID
   ```

6. **Invite le bot sur ton serveur** via OAuth2 URL Generator (scopes : `bot` + `applications.commands`, permissions : `Send Messages`, `Read Message History`, `View Channel`).

7. **Lance le bot** :
   ```bash
   python cemantix_bot.py
   ```

---

## Commandes Discord

| Commande | Description |
|---|---|
| `/start` | Affiche l'état de la partie du jour |
| `/top` | Classement des meilleurs joueurs |
| `/profil` | Tes stats personnelles |
| `/stats` | Stats globales du bot (modèle, mots utilisés, etc.) |

**Pour jouer** : Tape simplement un mot dans le salon configuré !

---

## Modèle utilisé

Par défaut, le bot utilise **frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin** :
- **Source** : [frWac2Vec](https://fauconnier.github.io/#data) (corpus frWac, 1.6 milliard de mots)
- **Architecture** : Skip-gram (meilleure que CBOW pour la similarité)
- **Dimensions** : 500 (vs 200 précédemment) → meilleure précision sémantique
- **Taille** : 298 Mo (vs 120 Mo) → vocabulaire plus riche
- **Cutoff** : 100 (filtre les mots trop rares)
- **MD5** : af38

### Changer de modèle

Tu peux utiliser un autre modèle depuis [embeddings.net](https://embeddings.net/embeddings/) :

1. Modifie `MODEL_PATH` dans `cemantix_bot.py`
2. Modifie `docker-entrypoint.sh` si tu utilises Docker :
   ```bash
   MODEL_NAME="ton_modele.bin"
   MODEL_URL="https://embeddings.net/embeddings/${MODEL_NAME}"
   MODEL_MD5="son_md5"
   ```
3. Rebuild le conteneur : `docker-compose down && docker-compose up -d --build`

---

## Déploiement 24/7

### Avec Docker (recommandé)
Le conteneur se relance automatiquement grâce à `restart: unless-stopped`.

### Avec systemd (manual)
Crée un service systemd (voir `CLAUDE.md` pour un exemple).

---

## Mise à jour

### Avec Docker
```bash
cd ~/cemantix-bot
git pull
docker-compose down
docker-compose up -d --build
```

### Sans Docker
```bash
cd ~/cemantix-bot
git pull
# Si le modèle ou les dépendances ont changé
pip install -r requirements.txt
# Redémarre le bot (Ctrl+C puis relance)
```
