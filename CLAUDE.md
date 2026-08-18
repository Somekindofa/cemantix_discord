# CLAUDE.md

Contexte pour Claude Code travaillant sur ce repo.

## Ce que fait ce projet

Un bot Discord mono-fichier (`cemantix_bot.py`) qui recrée Cémantix (Semantle en français) pour un unique salon Discord. Pas de multi-serveur, pas de base de données, tout est dans un seul fichier JSON local (`state.json`).

## Contraintes matérielles à respecter

Ce bot tourne en production sur un **Raspberry Pi 3B (1 Go de RAM)**. C'est la contrainte la plus importante du projet :

- Ne jamais proposer un modèle d'embeddings plus lourd que 200 dimensions / ~150 Mo. Le choix actuel (`frWac_no_postag_no_phrase_200_cbow_cut100.bin`, 120 Mo) est déjà proche de la limite raisonnable.
- Éviter d'ajouter des dépendances lourdes (pandas, torch, etc.) sans discuter d'abord de l'impact mémoire.
- Le chargement du modèle au démarrage prend 1-2 minutes sur ce matériel, c'est normal et attendu, ne pas essayer de l'optimiser sans raison.

## Architecture

- `discord.py` (Client + `app_commands.CommandTree`), pas de cogs, tout est dans un seul fichier volontairement (projet perso simple, pas une lib à maintenir).
- `gensim.models.KeyedVectors` pour charger le modèle word2vec et calculer `model.similarity(mot1, mot2)`.
- `discord.ext.tasks` pour la tâche planifiée quotidienne (`@tasks.loop(time=datetime.time(hour=0, minute=0))`) qui tire le nouveau mot du jour.
- Mot du jour tiré via l'API MediaWiki du Wiktionnaire français (`action=query&list=categorymembers&cmtitle=Catégorie:Noms communs en français`), avec un `cmstartsortkeyprefix` aléatoire pour obtenir un tirage pseudo-aléatoire dans la catégorie (~205 000 entrées). Le mot est filtré : doit exister dans le vocabulaire du modèle gensim, ne doit pas contenir d'espace/tiret/apostrophe (locutions composées écartées), ne doit jamais avoir été utilisé (`state["mots_utilises"]`).

## Schéma de `state.json`

```json
{
  "target": "mot_du_jour_actuel",
  "current_date": "2026-08-18",
  "found": false,
  "attempts_today": 0,
  "guesses_today": {"mot_proposé": score_pourcentage},
  "mots_utilises": ["liste", "de", "tous", "les", "mots", "déjà", "tombés"],
  "players": {
    "user_id_discord": {"victoires": 0, "coups_total": 0, "parties": 0}
  }
}
```

Ce fichier n'est **pas versionné** dans git (il est propre à chaque déploiement, contient les stats des vrais joueurs). Toute évolution du schéma doit passer par `load_state()` avec un pattern `setdefault` pour rester rétrocompatible avec les fichiers `state.json` déjà en place sur le Pi en production, ne jamais supposer qu'on peut le régénérer à zéro sans prévenir l'utilisateur (ça efface l'historique et les stats de ses amis).

## Déploiement

Le bot tourne comme service systemd (`cemantix.service`, non versionné). Après tout `git pull` ou modification manuelle de `cemantix_bot.py` sur le Pi, il faut explicitement faire :

```bash
sudo systemctl restart cemantix.service
```

Le service ne recharge jamais le code automatiquement. C'est un piège classique rencontré plusieurs fois pendant le développement : penser que le bot a pris en compte un changement alors qu'il tourne toujours sur l'ancienne version en mémoire.

## Fichiers volontairement absents du repo

- `*.bin` (modèle de vecteurs, trop lourd, à retélécharger sur chaque machine)
- `state.json` (données de production, spécifique à chaque déploiement)
- Le fichier de service systemd lui-même (contient des chemins absolus propres à chaque machine)
- `.env` (secrets : token Discord, ID du salon — voir `.env.example` pour le template)

Ces éléments sont couverts par `.gitignore`.

## Configuration (`.env`)

`TOKEN` et `CHANNEL_ID` ne sont plus en dur dans `cemantix_bot.py` : ils sont chargés depuis un fichier `.env` local via `python-dotenv` (`DISCORD_TOKEN`, `CHANNEL_ID`). Le template versionné est `.env.example` ; chacun copie ce fichier vers `.env` et le remplit, ce `.env` n'est jamais commité.

## Conventions de code

- Tout le code et les commentaires sont en français (le bot s'adresse à des joueurs francophones, cohérence avec les messages affichés dans Discord).
- Pas de framework de tests pour l'instant, projet perso à échelle d'un salon Discord entre amis, pas de CI/CD.
- Préférer la simplicité à l'extensibilité : ce n'est pas un projet pensé pour scaler à plusieurs serveurs (décision explicite prise en cours de route), ne pas réintroduire de complexité multi-guild sans qu'on le demande explicitement.
