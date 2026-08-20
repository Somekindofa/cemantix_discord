"""
Bot Discord type Cémantix, avec mot du jour tiré aléatoirement depuis
une liste pré-filtrée de noms communs (dico_mm.txt ou dico_ms.txt).

Deux modes disponibles:
1. Mode local: utilise un modèle word2vec local (frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin)
2. Mode API: utilise l'API Infomaniak pour les embeddings

Prérequis :
    pip install discord.py gensim numpy python-dotenv aiohttp

Modèle de vecteurs français à télécharger (une seule fois) pour le mode local:
    https://fauconnier.github.io/#data
    -> prends "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin" (120 Mo)
    -> place-le dans le même dossier que ce script

Fichiers de mots à télécharger (une seule fois) pour le mode local:
    https://raw.githubusercontent.com/TikSL/semanTikSl/main/dico_mm.txt
    https://raw.githubusercontent.com/TikSL/semanTikSl/main/dico_ms.txt
    -> place-les dans le même dossier que ce script

Pour le mode API:
    - Configurer INFOMANIAK_PRODUCT_ID et INFOMANIAK_API_KEY dans .env
    - Placer un fichier de vocabulaire dans data/vocab/vocab.txt
    - Lancer l'ingestion avec --ingest-vocab pour pré-calculer les embeddings

Usage :
    1. Copie .env.example vers .env et remplis TOKEN et CHANNEL_ID.
    2. Pour le mode API, configure aussi INFOMANIAK_PRODUCT_ID et INFOMANIAK_API_KEY
    3. python cemantix_bot.py [--mode api] [--ingest-vocab]
"""

import datetime
import json
import logging
import os
import random
import asyncio
import numpy as np
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv

# Import des utilitaires API (seulement si nécessaire)
try:
    from gensim.models import KeyedVectors
    LOCAL_MODE_AVAILABLE = True
except ImportError:
    LOCAL_MODE_AVAILABLE = False
    print("Warning: gensim not available, local mode disabled")

# Set up logging for journalctl
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---- Config chargée depuis .env (voir .env.example) ----
load_dotenv()
TOKEN = os.environ["DISCORD_TOKEN"]
CHANNEL_ID = int(os.environ["CHANNEL_ID"])
MODEL_PATH = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"
STATE_FILE = Path("state.json")
STATE_API_FILE = Path("state_api.json")

# Mode par défaut (peut être écrasé par argument CLI)
DEFAULT_MODE = "local"

# ---- Chargement du modèle local (si disponible) ----
model = None
if LOCAL_MODE_AVAILABLE:
    try:
        print("Chargement du modèle local (peut prendre 1-2 min sur un Pi)...")
        model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
        print("Modèle local chargé.")
        logger.info(f"Gensim model loaded: {MODEL_PATH}")
        logger.info(f"Vocab size: {len(model.key_to_index)}")
    except FileNotFoundError:
        print("Modèle local non trouvé, le mode local sera désactivé")
        model = None
    except Exception as e:
        print(f"Erreur de chargement du modèle local: {e}")
        model = None


# ---- Utilitaires API ----
from api_utils import (
    load_vocabulary, save_embeddings, load_embeddings,
    ingest_vocabulary, get_similarity, get_nearest_neighbors,
    get_rank, compress_similarity,
    INFOMANIAK_PRODUCT_ID, INFOMANIAK_API_KEY
)

# Variables globales pour le mode API
api_embeddings = None
api_words = None
api_word_to_index = None


async def load_api_data():
    """Charge les données d'embeddings pour le mode API."""
    global api_embeddings, api_words, api_word_to_index
    
    if api_embeddings is not None:
        return True
    
    embeddings, words, word_to_index = load_embeddings()
    if embeddings is not None and words is not None and word_to_index is not None:
        api_embeddings = embeddings
        api_words = words
        api_word_to_index = word_to_index
        logger.info(f"Données API chargées: {len(words)} mots")
        return True
    
    logger.warning("Aucune donnée API trouvée, le mode API ne sera pas disponible")
    return False


def is_api_mode_available():
    """Vérifie si le mode API est disponible."""
    return (INFOMANIAK_PRODUCT_ID is not None and 
            INFOMANIAK_API_KEY is not None and
            api_embeddings is not None and
            api_words is not None and
            api_word_to_index is not None)


# ---- État du jeu ----

def load_state(mode: str = "local") -> dict:
    """Charge l'historique depuis le disque, ou l'initialise."""
    state_file = STATE_API_FILE if mode == "api" else STATE_FILE
    
    defaults = {
        "target": None,
        "current_date": None,
        "found": False,
        "attempts_today": 0,
        "players": {},
        "guesses_today": {},
        "mots_utilises": [],
        "neighbors": [],
        "mode": mode,
    }
    
    if state_file.exists():
        with open(state_file, encoding="utf-8") as f:
            loaded = json.load(f)
        for key, value in defaults.items():
            loaded.setdefault(key, value)
        return loaded
    return defaults


def save_state(state: dict, mode: str = "local"):
    """Sauvegarde l'état du jeu."""
    state_file = STATE_API_FILE if mode == "api" else STATE_FILE
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# États séparés pour chaque mode
state_local = load_state("local")
state_api = load_state("api")


# ---- Logique du jeu - Mode Local ----

async def tirer_mot_wiktionnaire() -> str | None:
    """Sélectionne un mot aléatoire depuis une liste pré-filtrée de noms communs."""
    MIN_NORM = 10.0
    
    word_list_path = "dico_mm.txt"
    if not os.path.exists(word_list_path):
        word_list_path = "dico_ms.txt"
    
    try:
        with open(word_list_path, "r", encoding="utf-8") as f:
            words = [line.strip().lower() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Fichier de mots introuvable: {word_list_path}")
        return None
    
    valid_words = []
    for mot in words:
        if " " in mot or "-" in mot or "'" in mot:
            continue
        if mot not in model:
            continue
        if mot in state_local.get("mots_utilises", []):
            continue
        norm = np.linalg.norm(model[mot])
        if norm < MIN_NORM:
            continue
        valid_words.append(mot)
    
    if not valid_words:
        logger.error("Aucun mot valide trouvé")
        return None
    
    mot = random.choice(valid_words)
    logger.info(f"Selected word (local): '{mot}' (L2 norm: {np.linalg.norm(model[mot]):.2f})")
    return mot


async def nouveau_mot_du_jour_local():
    """Tire un nouveau mot pour le mode local."""
    mot = await tirer_mot_wiktionnaire()
    if mot is None:
        logger.error("Failed to select a word (local)")
        return

    today = str(datetime.date.today())
    state_local["target"] = mot
    state_local["current_date"] = today
    state_local["found"] = False
    state_local["attempts_today"] = 0
    state_local["guesses_today"] = {}
    
    try:
        neighbors = [word for word, _ in model.most_similar(mot, topn=100)]
        state_local["neighbors"] = neighbors
        logger.info(f"Precomputed 100 neighbors for '{mot}'")
    except KeyError as e:
        logger.error(f"Failed to compute neighbors for '{mot}': {e}")
        state_local["neighbors"] = []
    
    state_local["mots_utilises"].append(mot)
    save_state(state_local, "local")
    logger.info(f"Nouveau mot du jour (local): {mot}")


async def check_reset_local():
    """Vérifie si un nouveau mot doit être tiré (mode local)."""
    today = str(datetime.date.today())
    if state_local["current_date"] != today or state_local["target"] is None:
        await nouveau_mot_du_jour_local()
    elif state_local["target"] not in model:
        await nouveau_mot_du_jour_local()


def current_target_local() -> str:
    return state_local["target"]


def record_result_local(user_id: str, coups: int, gagne: bool):
    p = state_local["players"].setdefault(
        user_id, {"victoires": 0, "coups_total": 0, "parties": 0}
    )
    p["parties"] += 1
    p["coups_total"] += coups
    if gagne:
        p["victoires"] += 1
    save_state(state_local, "local")


def record_guess_local(word: str, temp: float):
    existing = state_local["guesses_today"].get(word)
    if existing is None or temp > existing:
        state_local["guesses_today"][word] = temp


# ---- Logique du jeu - Mode API ----

async def tirer_mot_api() -> str | None:
    """Sélectionne un mot aléatoire depuis le vocabulaire API."""
    global api_words, api_word_to_index
    
    if api_words is None or len(api_words) == 0:
        logger.error("Aucun vocabulaire API chargé")
        return None
    
    # Filtrer les mots déjà utilisés
    used_words = state_api.get("mots_utilises", [])
    valid_words = [word for word in api_words if word not in used_words]
    
    if not valid_words:
        logger.error("Tous les mots du vocabulaire API ont déjà été utilisés")
        return None
    
    mot = random.choice(valid_words)
    logger.info(f"Selected word (API): '{mot}'")
    return mot


async def nouveau_mot_du_jour_api():
    """Tire un nouveau mot pour le mode API."""
    global api_embeddings, api_words, api_word_to_index
    
    # Charger les données API si nécessaire
    if not await load_api_data():
        logger.error("Impossible de charger les données API")
        return
    
    mot = await tirer_mot_api()
    if mot is None:
        logger.error("Failed to select a word (API)")
        return

    today = str(datetime.date.today())
    state_api["target"] = mot
    state_api["current_date"] = today
    state_api["found"] = False
    state_api["attempts_today"] = 0
    state_api["guesses_today"] = {}
    
    # Pré-calculer les 100 plus proches voisins
    try:
        neighbors = get_nearest_neighbors(mot, api_embeddings, api_word_to_index, api_words, topn=100)
        state_api["neighbors"] = [word for word, _ in neighbors]
        logger.info(f"Precomputed 100 neighbors for '{mot}' (API)")
    except Exception as e:
        logger.error(f"Failed to compute neighbors for '{mot}': {e}")
        state_api["neighbors"] = []
    
    state_api["mots_utilises"].append(mot)
    save_state(state_api, "api")
    logger.info(f"Nouveau mot du jour (API): {mot}")


async def check_reset_api():
    """Vérifie si un nouveau mot doit être tiré (mode API)."""
    today = str(datetime.date.today())
    if state_api["current_date"] != today or state_api["target"] is None:
        await nouveau_mot_du_jour_api()
    elif state_api["target"] not in api_word_to_index:
        await nouveau_mot_du_jour_api()


def current_target_api() -> str:
    return state_api["target"]


def record_result_api(user_id: str, coups: int, gagne: bool):
    p = state_api["players"].setdefault(
        user_id, {"victoires": 0, "coups_total": 0, "parties": 0}
    )
    p["parties"] += 1
    p["coups_total"] += coups
    if gagne:
        p["victoires"] += 1
    save_state(state_api, "api")


def record_guess_api(word: str, temp: float):
    existing = state_api["guesses_today"].get(word)
    if existing is None or temp > existing:
        state_api["guesses_today"][word] = temp


# ---- Fonctions communes ----

def make_bar(temp: float, length: int = 10) -> str:
    """Construit une barre de progression en blocs Unicode."""
    pct = max(0.0, min(100.0, temp))
    filled = round(pct / 100 * length)
    return "\u2588" * filled + "\u2591" * (length - filled)


def build_proches_embed(state: dict, mode: str = "local") -> discord.Embed:
    """Construit l'embed du top 10 des mots les plus proches."""
    embed = discord.Embed(
        title="\ud83d\udd25 Top 10 des mots les plus proches",
        color=discord.Color.orange(),
    )
    guesses = state["guesses_today"]
    if not guesses:
        embed.description = "Aucune proposition pour l'instant."
        return embed

    classement = sorted(guesses.items(), key=lambda kv: -kv[1])[:10]
    lignes = []
    for i, (word, temp) in enumerate(classement, start=1):
        rang = f"{i:>2}."
        mot = word[:12].ljust(12)
        pct_str = f"{temp:>6.1f}%"
        lignes.append(f"{rang} {mot} {make_bar(temp)} {pct_str}")

    embed.description = "```\n" + "\n".join(lignes) + "\n```"
    return embed


def get_current_state(mode: str):
    """Retourne l'état actuel pour le mode spécifié."""
    return state_api if mode == "api" else state_local


# ---- Bot Discord ----
intents = discord.Intents.default()
intents.message_content = True


class CemantixClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.mode = DEFAULT_MODE

    async def setup_hook(self):
        await self.tree.sync()


client = CemantixClient()


@tasks.loop(time=datetime.time(hour=0, minute=0))
async def tirage_minuit():
    """Tire un nouveau mot à minuit pour les deux modes."""
    await nouveau_mot_du_jour_local()
    await nouveau_mot_du_jour_api()
    channel = client.get_channel(CHANNEL_ID)
    if channel is not None:
        await channel.send("\ud83c\udf05 Nouveau mot du jour disponible, à vous de jouer !")


@client.event
async def on_ready():
    await check_reset_local()
    await check_reset_api()
    if not tirage_minuit.is_running():
        tirage_minuit.start()
    print(f"Connecté en tant que {client.user}")
    logger.info(f"Bot connected as {client.user}")
    
    # Charger les données API en arrière-plan
    await load_api_data()


@client.event
async def on_message(message: discord.Message):
    if message.author.bot or message.channel.id != CHANNEL_ID:
        return

    # Déterminer le mode actuel (par défaut local, mais peut être changé)
    mode = client.mode
    state = get_current_state(mode)
    
    await check_reset_local()
    await check_reset_api()
    
    if state["found"]:
        return

    guess = message.content.strip().lower()
    if not guess or " " in guess:
        return

    target = current_target_local() if mode == "local" else current_target_api()
    state["attempts_today"] += 1

    if guess == target:
        state["found"] = True
        save_state(state, mode)
        
        if mode == "local":
            record_result_local(str(message.author.id), state["attempts_today"], gagne=True)
        else:
            record_result_api(str(message.author.id), state["attempts_today"], gagne=True)
        
        await message.reply(
            f"\ud83c\udf89 Trouvé en {state['attempts_today']} coups ! "
            f"Le mot était **{target}**"
        )
        embed = build_proches_embed(state, mode)
        await message.channel.send(embed=embed)
        return

    # Vérifier si le mot existe dans le vocabulaire
    if mode == "local":
        if guess not in model:
            await message.reply("\u274c Mot inconnu du dictionnaire.")
            return
        
        # Calculer la similarité
        sim = model.similarity(guess, target)
        neighbors = state.get("neighbors", [])
        rank = neighbors.index(guess) + 1 if guess in neighbors else None
        
        # Appliquer l'échelle logarithmique
        compressed_temp = compress_similarity(sim)
        
        # Formater la réponse
        if rank is not None and 1 <= rank <= 100:
            emoji = "\ud83d\udd25" if compressed_temp > 60 else ("\u2601\ufe0f" if compressed_temp > 30 else "\u2744\ufe0f")
            response = f"{emoji} `{make_bar(compressed_temp)}` {compressed_temp}% (Rank: {rank}/100)"
        else:
            response = f"{sim:.2f}"
        
        record_guess_local(guess, compressed_temp)
        save_state(state, mode)
        
    else:  # Mode API
        if guess not in api_word_to_index:
            await message.reply("\u274c Mot inconnu du vocabulaire API.")
            return
        
        # Calculer la similarité via API
        try:
            sim = get_similarity(guess, target, api_embeddings, api_word_to_index)
            neighbors = state.get("neighbors", [])
            rank = neighbors.index(guess) + 1 if guess in neighbors else None
            
            # Appliquer l'échelle logarithmique
            compressed_temp = compress_similarity(sim)
            
            # Formater la réponse
            if rank is not None and 1 <= rank <= 100:
                emoji = "\ud83d\udd25" if compressed_temp > 60 else ("\u2601\ufe0f" if compressed_temp > 30 else "\u2744\ufe0f")
                response = f"{emoji} `{make_bar(compressed_temp)}` {compressed_temp}% (Rank: {rank}/100)"
            else:
                response = f"{sim:.2f}"
            
            record_guess_api(guess, compressed_temp)
            save_state(state, mode)
        except Exception as e:
            logger.error(f"Erreur lors du calcul de similarité API: {e}")
            await message.reply("\u274c Erreur lors du calcul de la similarité.")
            return
    
    embed = build_proches_embed(state, mode)
    await message.reply(content=response, embed=embed)


# ---- Commandes Discord ----

@client.tree.command(name="start", description="Affiche l'état de la partie du jour (mode local)")
async def start(interaction: discord.Interaction):
    await check_reset_local()
    state = state_local
    if state["found"]:
        msg = f"Le mot du jour a déjà été trouvé en {state['attempts_today']} coups !"
    else:
        msg = (
            f"Partie en cours. {state['attempts_today']} propositions faites "
            f"aujourd'hui. Tape un mot dans ce salon pour jouer."
        )
    await interaction.response.send_message(msg)


@client.tree.command(name="start_api", description="Affiche l'état de la partie du jour (mode API)")
async def start_api(interaction: discord.Interaction):
    await check_reset_api()
    
    # Vérifier que le mode API est disponible
    if not is_api_mode_available():
        await interaction.response.send_message(
            "\u26a0\ufe0f Le mode API n'est pas disponible. "
            "Vérifie que INFOMANIAK_PRODUCT_ID et INFOMANIAK_API_KEY sont configurés, "
            "et que les embeddings ont été générés avec --ingest-vocab."
        )
        return
    
    state = state_api
    if state["found"]:
        msg = f"Le mot du jour (API) a déjà été trouvé en {state['attempts_today']} coups !"
    else:
        msg = (
            f"Partie en cours (mode API). {state['attempts_today']} propositions faites "
            f"aujourd'hui. Tape un mot dans ce salon pour jouer."
        )
    await interaction.response.send_message(msg)


@client.tree.command(name="set_mode", description="Change le mode de jeu (local/api)")
@app_commands.choices(mode=[
    app_commands.Choice(name="local", value="local"),
    app_commands.Choice(name="api", value="api"),
])
async def set_mode(interaction: discord.Interaction, mode: app_commands.Choice[str]):
    client.mode = mode.value
    await interaction.response.send_message(
        f"Mode changé vers **{mode.value}**. "
        f"Les prochaines propositions utiliseront ce mode."
    )


@client.tree.command(name="top", description="Classement des meilleurs joueurs")
async def top(interaction: discord.Interaction):
    # Fusionner les stats des deux modes
    all_players = {}
    
    for mode in ["local", "api"]:
        state = state_local if mode == "local" else state_api
        for user_id, stats in state["players"].items():
            if user_id not in all_players:
                all_players[user_id] = {
                    "victoires": 0,
                    "coups_total": 0,
                    "parties": 0
                }
            all_players[user_id]["victoires"] += stats["victoires"]
            all_players[user_id]["coups_total"] += stats["coups_total"]
            all_players[user_id]["parties"] += stats["parties"]
    
    if not all_players:
        await interaction.response.send_message("Personne n'a encore joué.")
        return

    classement = sorted(
        all_players.items(), key=lambda kv: -kv[1]["victoires"]
    )[:10]

    lignes = []
    for i, (user_id, stats) in enumerate(classement, start=1):
        moyenne = stats["coups_total"] / stats["parties"] if stats["parties"] else 0
        lignes.append(
            f"{i}. <@{user_id}> — {stats['victoires']} victoires "
            f"(moyenne {moyenne:.1f} coups)"
        )

    await interaction.response.send_message("\ud83c\udfc6 **Classement**\n" + "\n".join(lignes))


@client.tree.command(name="top_local", description="Classement des meilleurs joueurs (mode local)")
async def top_local(interaction: discord.Interaction):
    players = state_local["players"]
    if not players:
        await interaction.response.send_message("Personne n'a encore joué en mode local.")
        return

    classement = sorted(
        players.items(), key=lambda kv: -kv[1]["victoires"]
    )[:10]

    lignes = []
    for i, (user_id, stats) in enumerate(classement, start=1):
        moyenne = stats["coups_total"] / stats["parties"] if stats["parties"] else 0
        lignes.append(
            f"{i}. <@{user_id}> — {stats['victoires']} victoires "
            f"(moyenne {moyenne:.1f} coups)"
        )

    await interaction.response.send_message("\ud83c\udfc6 **Classement (Local)**\n" + "\n".join(lignes))


@client.tree.command(name="top_api", description="Classement des meilleurs joueurs (mode API)")
async def top_api(interaction: discord.Interaction):
    players = state_api["players"]
    if not players:
        await interaction.response.send_message("Personne n'a encore joué en mode API.")
        return

    classement = sorted(
        players.items(), key=lambda kv: -kv[1]["victoires"]
    )[:10]

    lignes = []
    for i, (user_id, stats) in enumerate(classement, start=1):
        moyenne = stats["coups_total"] / stats["parties"] if stats["parties"] else 0
        lignes.append(
            f"{i}. <@{user_id}> — {stats['victoires']} victoires "
            f"(moyenne {moyenne:.1f} coups)"
        )

    await interaction.response.send_message("\ud83c\udfc6 **Classement (API)**\n" + "\n".join(lignes))


@client.tree.command(name="profil", description="Consulter tes stats de jeu")
async def profil(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    
    # Fusionner les stats des deux modes
    all_stats = {
        "victoires": 0,
        "coups_total": 0,
        "parties": 0
    }
    
    for mode in ["local", "api"]:
        state = state_local if mode == "local" else state_api
        stats = state["players"].get(user_id)
        if stats:
            all_stats["victoires"] += stats["victoires"]
            all_stats["coups_total"] += stats["coups_total"]
            all_stats["parties"] += stats["parties"]
    
    if all_stats["parties"] == 0:
        await interaction.response.send_message("Tu n'as pas encore joué.")
        return

    moyenne = all_stats["coups_total"] / all_stats["parties"] if all_stats["parties"] else 0
    await interaction.response.send_message(
        f"**Profil de {interaction.user.display_name}**\n"
        f"Parties jouées : {all_stats['parties']}\n"
        f"Victoires : {all_stats['victoires']}\n"
        f"Moyenne de coups : {moyenne:.1f}"
    )


@client.tree.command(name="profil_local", description="Consulter tes stats de jeu (mode local)")
async def profil_local(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    stats = state_local["players"].get(user_id)
    if not stats:
        await interaction.response.send_message("Tu n'as pas encore joué en mode local.")
        return

    moyenne = stats["coups_total"] / stats["parties"] if stats["parties"] else 0
    await interaction.response.send_message(
        f"**Profil de {interaction.user.display_name} (Local)**\n"
        f"Parties jouées : {stats['parties']}\n"
        f"Victoires : {stats['victoires']}\n"
        f"Moyenne de coups : {moyenne:.1f}"
    )


@client.tree.command(name="profil_api", description="Consulter tes stats de jeu (mode API)")
async def profil_api(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    stats = state_api["players"].get(user_id)
    if not stats:
        await interaction.response.send_message("Tu n'as pas encore joué en mode API.")
        return

    moyenne = stats["coups_total"] / stats["parties"] if stats["parties"] else 0
    await interaction.response.send_message(
        f"**Profil de {interaction.user.display_name} (API)**\n"
        f"Parties jouées : {stats['parties']}\n"
        f"Victoires : {stats['victoires']}\n"
        f"Moyenne de coups : {moyenne:.1f}"
    )


@client.tree.command(name="mode_status", description="Affiche le mode de jeu actuel")
async def mode_status(interaction: discord.Interaction):
    mode = client.mode
    api_available = is_api_mode_available()
    local_available = model is not None
    
    msg = f"Mode actuel: **{mode}**\n"
    msg += f"Mode local disponible: {'\u2705' if local_available else '\u274c'}\n"
    msg += f"Mode API disponible: {'\u2705' if api_available else '\u274c'}"
    
    if mode == "api" and not api_available:
        msg += "\n\u26a0\ufe0f Attention: le mode API est sélectionné mais n'est pas disponible"
    
    await interaction.response.send_message(msg)


@client.tree.command(name="ingest_vocab", description="[Admin] Lance l'ingestion du vocabulaire API")
async def ingest_vocab_command(interaction: discord.Interaction):
    """Commande pour déclencher l'ingestion du vocabulaire via l'API."""
    if not (INFOMANIAK_PRODUCT_ID and INFOMANIAK_API_KEY):
        await interaction.response.send_message(
            "\u274c Impossible de lancer l'ingestion: "
            "INFOMANIAK_PRODUCT_ID et INFOMANIAK_API_KEY doivent être configurés dans .env"
        )
        return
    
    await interaction.response.send_message(
        "\u23f0 Démarrage de l'ingestion du vocabulaire... "
        "Cela peut prendre plusieurs minutes selon la taille du vocabulaire."
    )
    
    try:
        success = await ingest_vocabulary()
        if success:
            await interaction.followup.send(
                "\u2705 Ingestion terminée avec succès! "
                "Les embeddings sont maintenant prêts pour le mode API."
            )
            # Recharger les données
            await load_api_data()
        else:
            await interaction.followup.send(
                "\u274c L'ingestion a échoué. "
                "Vérifie les logs pour plus de détails."
            )
    except Exception as e:
        logger.error(f"Erreur lors de l'ingestion: {e}")
        await interaction.followup.send(
            f"\u274c Erreur lors de l'ingestion: {e}"
        )


# ---- Point d'entrée ----

async def main():
    """Point d'entrée principal avec gestion des arguments CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bot Discord Cémantix")
    parser.add_argument("--mode", choices=["local", "api"], default=DEFAULT_MODE,
                        help="Mode de jeu par défaut (local ou api)")
    parser.add_argument("--ingest-vocab", action="store_true",
                        help="Lance l'ingestion du vocabulaire API avant de démarrer")
    
    args = parser.parse_args()
    
    # Mettre à jour le mode
    client.mode = args.mode
    
    # Lancer l'ingestion si demandé
    if args.ingest_vocab:
        if not (INFOMANIAK_PRODUCT_ID and INFOMANIAK_API_KEY):
            print("Erreur: INFOMANIAK_PRODUCT_ID et INFOMANIAK_API_KEY doivent être configurés")
            return
        
        print("Démarrage de l'ingestion du vocabulaire...")
        success = await ingest_vocabulary()
        if success:
            print("Ingestion terminée avec succès!")
            await load_api_data()
        else:
            print("L'ingestion a échoué")
            return
    
    # Démarrer le bot
    await client.start(TOKEN)


if __name__ == "__main__":
    import sys
    
    # Vérifier que les dépendances nécessaires sont installées
    try:
        import aiohttp
    except ImportError:
        print("Erreur: aiohttp est requis. Installez-le avec: pip install aiohttp")
        sys.exit(1)
    
    asyncio.run(main())
