"""
Bot Discord type Cémantix.

Le bot tire automatiquement un mot du jour à minuit parmi une liste de
noms communs filtrés depuis Lexique4.tsv (CDOrtho >= 20), et calcule la similarité
sémantique entre les propositions des joueurs et le mot cible.

Utilise un modèle word2vec local : frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin

Prérequis :
    pip install discord.py gensim numpy python-dotenv

Usage :
    1. Copie .env.example vers .env et remplis TOKEN et CHANNEL_ID
    2. Le modèle et les dictionnaires sont téléchargés automatiquement (Docker)
       ou doivent être placés manuellement à la racine du projet
    3. python cemantix_bot.py
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
from gensim.models import KeyedVectors

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---- Config ----
load_dotenv()
TOKEN = os.environ["DISCORD_TOKEN"]
CHANNEL_ID = int(os.environ["CHANNEL_ID"])

# Model configuration - 500-dim skip-gram, better than 200-dim cbow
MODEL_PATH = "frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin"
STATE_FILE = Path("data/state.json")
WORD_LIST_PATH = "data/noms_communs.txt"

# ---- Load word2vec model ----
print("Chargement du modèle word2vec (peut prendre 1-2 min)...")
model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
print(f"Modèle chargé: {len(model.key_to_index)} mots, {model.vector_size} dimensions")
logger.info(f"Gensim model loaded: {MODEL_PATH}")
logger.info(f"Vocab size: {len(model.key_to_index)}, Dimensions: {model.vector_size}")

# Load common noun set from pre-filtered list
noms_set = set()
try:
    with open(WORD_LIST_PATH, "r", encoding="utf-8") as f:
        noms_set = {line.strip().lower() for line in f if line.strip()}
    logger.info(f"Loaded {len(noms_set)} common nouns from {WORD_LIST_PATH}")
except FileNotFoundError:
    logger.warning(f"{WORD_LIST_PATH} not found - will use fallback word lists")
    noms_set = None

# ---- Similarity compression ----
def compress_similarity(sim: float) -> float:
    """
    Applique une échelle logarithmique pour compresser les scores de similarité.
    Transforme la similarité cosinus (0-1) en un score 0-100 plus "_player-friendly".
    """
    sim = max(0.0, min(1.0, sim))
    compressed_sim = np.log1p(sim * 10) / np.log1p(10)
    return round(float(compressed_sim) * 100, 1)


# ---- Game state ----
def load_state() -> dict:
    """Charge l'historique depuis le disque, ou l'initialise."""
    defaults = {
        "target": None,
        "current_date": None,
        "found": False,
        "attempts_today": 0,
        "players": {},
        "guesses_today": {},
        "mots_utilises": [],
        "neighbors": [],
    }
    
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            loaded = json.load(f)
        for key, value in defaults.items():
            loaded.setdefault(key, value)
        return loaded
    return defaults


def save_state(state: dict):
    """Sauvegarde l'état du jeu."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


state = load_state()


# ---- Game logic ----
# For 500-dim skip-gram model, norms are typically 2-6 (vs 10-20 for 200-dim cbow)
MIN_NORM = 2.0


async def tirer_mot() -> str | None:
    """Sélectionne un mot aléatoire depuis la liste des noms communs filtrés."""
    word_list_path = WORD_LIST_PATH
    
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
        # Words are pre-filtered, so we can skip the norm check
        # But still check if in model (in case model changed)
        if mot not in model:
            continue
        if mot in state.get("mots_utilises", []):
            continue
        norm = np.linalg.norm(model[mot])
        if norm < MIN_NORM:
            continue
        valid_words.append(mot)
    
    if not valid_words:
        logger.error("Aucun mot valide trouvé")
        return None
    
    mot = random.choice(valid_words)
    logger.info(f"Selected word: '{mot}' (L2 norm: {np.linalg.norm(model[mot]):.2f})")
    return mot


async def nouveau_mot_du_jour():
    """Tire un nouveau mot pour le jour."""
    mot = await tirer_mot()
    if mot is None:
        logger.error("Failed to select a word")
        return
    
    today = str(datetime.date.today())
    state["target"] = mot
    state["current_date"] = today
    state["found"] = False
    state["attempts_today"] = 0
    state["guesses_today"] = {}
    
    try:
        neighbors = [word for word, _ in model.most_similar(mot, topn=100)]
        state["neighbors"] = neighbors
        logger.info(f"Precomputed 100 neighbors for '{mot}'")
    except KeyError as e:
        logger.error(f"Failed to compute neighbors for '{mot}': {e}")
        state["neighbors"] = []
    
    state["mots_utilises"].append(mot)
    save_state(state)
    logger.info(f"Nouveau mot du jour: {mot}")


async def check_reset():
    """Vérifie si un nouveau mot doit être tiré."""
    today = str(datetime.date.today())
    if state["current_date"] != today or state["target"] is None:
        await nouveau_mot_du_jour()
    elif state["target"] not in model:
        await nouveau_mot_du_jour()


def current_target() -> str:
    return state["target"]


def record_result(user_id: str, coups: int, gagne: bool):
    p = state["players"].setdefault(
        user_id, {"victoires": 0, "coups_total": 0, "parties": 0}
    )
    p["parties"] += 1
    p["coups_total"] += coups
    if gagne:
        p["victoires"] += 1
    save_state(state)


def record_guess(word: str, temp: float):
    existing = state["guesses_today"].get(word)
    if existing is None or temp > existing:
        state["guesses_today"][word] = temp


# ---- Discord bot ----
intents = discord.Intents.default()
intents.message_content = True


class CemantixClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
    
    async def setup_hook(self):
        await self.tree.sync()


client = CemantixClient()


@tasks.loop(time=datetime.time(hour=0, minute=0))
async def tirage_minuit():
    """Tire un nouveau mot à minuit."""
    await nouveau_mot_du_jour()
    channel = client.get_channel(CHANNEL_ID)
    if channel is not None:
        await channel.send("🌅 Nouveau mot du jour disponible, à vous de jouer !")


@client.event
async def on_ready():
    await check_reset()
    if not tirage_minuit.is_running():
        tirage_minuit.start()
    print(f"Connecté en tant que {client.user}")
    logger.info(f"Bot connected as {client.user}")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot or message.channel.id != CHANNEL_ID:
        return
    
    await check_reset()
    
    if state["found"]:
        return
    
    guess = message.content.strip().lower()
    if not guess or " " in guess:
        return
    
    target = current_target()
    state["attempts_today"] += 1
    
    if guess == target:
        state["found"] = True
        save_state(state)
        
        record_result(str(message.author.id), state["attempts_today"], gagne=True)
        
        await message.reply(
            f"🎉 Trouvé en {state['attempts_today']} coups ! "
            f"Le mot était **{target}**"
        )
        embed = build_proches_embed()
        await message.channel.send(embed=embed)
        return
    
    # Vérifier si le mot est un nom commun
    if noms_set and guess not in noms_set:
        if guess in model:
            await message.reply("❌ Seuls les noms communs sont acceptés.")
        else:
            await message.reply("❌ Mot inconnu du vocabulaire.")
        return
    elif guess not in model:
        await message.reply("❌ Mot inconnu du dictionnaire.")
        return
    
    # Calculer la similarité
    sim = model.similarity(guess, target)
    neighbors = state.get("neighbors", [])
    rank = neighbors.index(guess) + 1 if guess in neighbors else None
    
    # Appliquer l'échelle logarithmique
    compressed_temp = compress_similarity(sim)
    
    # Formater la réponse
    if rank is not None and 1 <= rank <= 100:
        emoji = "🔥" if compressed_temp > 60 else ("☀️" if compressed_temp > 30 else "❄️")
        response = f"{emoji} `{make_bar(compressed_temp)}` {compressed_temp}% (Rank: {rank}/100)"
    else:
        response = f"{sim:.2f}"
    
    record_guess(guess, compressed_temp)
    save_state(state)
    
    embed = build_proches_embed()
    await message.reply(content=response, embed=embed)


# ---- Helper functions ----

def make_bar(temp: float, length: int = 10) -> str:
    """Construit une barre de progression en blocs Unicode."""
    pct = max(0.0, min(100.0, temp))
    filled = round(pct / 100 * length)
    return "█" * filled + "░" * (length - filled)


def build_proches_embed() -> discord.Embed:
    """Construit l'embed du top 10 des mots les plus proches."""
    embed = discord.Embed(
        title="🔥 Top 10 des mots les plus proches",
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


# ---- Discord commands ----

@client.tree.command(name="start", description="Affiche l'état de la partie du jour")
async def start(interaction: discord.Interaction):
    await check_reset()
    if state["found"]:
        msg = f"Le mot du jour a déjà été trouvé en {state['attempts_today']} coups !"
    else:
        msg = (
            f"Partie en cours. {state['attempts_today']} propositions faites "
            f"aujourd'hui. Tape un mot dans ce salon pour jouer."
        )
    await interaction.response.send_message(msg)


@client.tree.command(name="top", description="Classement des meilleurs joueurs")
async def top(interaction: discord.Interaction):
    players = state["players"]
    if not players:
        await interaction.response.send_message("Personne n'a encore joué.")
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
    
    await interaction.response.send_message("🏆 **Classement**\n" + "\n".join(lignes))


@client.tree.command(name="profil", description="Consulter tes stats de jeu")
async def profil(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    stats = state["players"].get(user_id)
    
    if not stats:
        await interaction.response.send_message("Tu n'as pas encore joué.")
        return
    
    moyenne = stats["coups_total"] / stats["parties"] if stats["parties"] else 0
    await interaction.response.send_message(
        f"**Profil de {interaction.user.display_name}**\n"
        f"Parties jouées : {stats['parties']}\n"
        f"Victoires : {stats['victoires']}\n"
        f"Moyenne de coups : {moyenne:.1f}"
    )


@client.tree.command(name="stats", description="Stats globales du bot")
async def stats_command(interaction: discord.Interaction):
    total_games = len(state["mots_utilises"])
    total_players = len(state["players"])
    total_guesses = state["attempts_today"]
    
    msg = (
        f"**Stats du bot**\n"
        f"Mots utilisés : {total_games}\n"
        f"Joueurs : {total_players}\n"
        f"Tentatives aujourd'hui : {total_guesses}\n"
        f"Modèle : {MODEL_PATH} ({len(model.key_to_index)} mots, {model.vector_size} dim)"
    )
    await interaction.response.send_message(msg)


# ---- Main ----
async def main():
    await client.start(TOKEN)


if __name__ == "__main__":
    import sys
    
    try:
        import gensim
    except ImportError:
        print("Erreur: gensim est requis. Installez-le avec: pip install gensim")
        sys.exit(1)
    
    asyncio.run(main())
