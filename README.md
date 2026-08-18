# Cémantix Discord Bot

Bot Discord qui recrée le principe de [Cémantix](https://cemantix.certitudes.org/) (jeu à la Semantle : deviner un mot grâce à un score de proximité sémantique "hot/cold") pour jouer entre amis sur un serveur Discord.

## Fonctionnement

- Chaque nuit à minuit, le bot tire automatiquement un nouveau nom commun français depuis l'API du [Wiktionnaire](https://fr.wiktionary.org) (catégorie "Noms communs en français"), en excluant les mots déjà utilisés.
- Les joueurs tapent un mot dans le salon configuré. Le bot calcule la similarité cosinus entre ce mot et le mot du jour via un modèle d'embeddings [word2vec français (frWac2Vec)](https://fauconnier.github.io/#data), et répond avec un score en % et une barre de progression.
- Un top 10 des mots les plus proches proposés dans la journée s'affiche à chaque tentative.
- Stats persistées par joueur (victoires, nombre moyen de coups).

## Commandes

| Commande | Description |
|---|---|
| `/start` | État de la partie du jour (nombre de tentatives, trouvé ou non) |
| `/top` | Classement des joueurs par victoires |
| `/profil` | Stats personnelles |

## Prérequis

- Python 3.11+
- Un Raspberry Pi (ou toute machine Linux) allumé en continu
- Une application Discord Bot ([Developer Portal](https://discord.com/developers/applications)) avec l'intent **Message Content** activé

```bash
pip install discord.py gensim requests python-dotenv
```

## Installation

1. Clone ce repo sur ta machine :
   ```bash
   git clone <url_du_repo> cemantix-bot
   cd cemantix-bot
   ```

2. Télécharge le modèle de vecteurs français depuis [fauconnier.github.io/#data](https://fauconnier.github.io/#data) : prends `frWac_no_postag_no_phrase_200_cbow_cut100.bin` (120 Mo, cbow, 200 dimensions, cutoff 100), place-le à la racine du repo. **Ce fichier n'est pas versionné dans git** (trop lourd), il faut le retélécharger sur chaque nouvelle machine.

3. Copie `.env.example` vers `.env` et remplis les valeurs (ce fichier n'est jamais commité, voir `.gitignore`) :
   - `DISCORD_TOKEN` : le token de ton bot (Developer Portal → Bot → Reset Token)
   - `CHANNEL_ID` : l'ID du salon Discord où vous jouez (mode développeur activé → clic droit sur le salon → copier l'identifiant)

4. Invite le bot sur ton serveur via OAuth2 URL Generator (scopes `bot` + `applications.commands`, permissions `Send Messages`, `Read Message History`, `View Channel`).

5. Lance le bot :
   ```bash
   python cemantix_bot.py
   ```

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

## Structure des données

`state.json` (non versionné, généré au premier lancement) contient le mot du jour, l'historique des mots déjà utilisés, les scores du jour, et les stats des joueurs. Voir `CLAUDE.md` pour le schéma détaillé.
