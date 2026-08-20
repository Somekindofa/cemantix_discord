"""
Bot Discord type Cémantix, avec mot du jour tiré aléatoirement depuis
une liste pré-filtrée de noms communs (dico_mm.txt ou dico_ms.txt).

Prérequis :
    pip install discord.py gensim numpy python-dotenv

Modèle de vecteurs français à télécharger (une seule fois) :
    https://fauconnier.github.io/#data
    -> prends "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin" (120 Mo)
    -> place-le dans le même dossier que ce script

Fichiers de mots à télécharger (une seule fois) :
    https://raw.githubusercontent.com/TikSL/semanTikSl/main/dico_mm.txt
    https://raw.githubusercontent.com/TikSL/semanTikSl/main/dico_ms.txt
    -> place-les dans le même dossier que ce script

Usage :
    1. Copie .env.example vers .env et remplis TOKEN et CHANNEL_ID.
    2. python cemantix_bot.py
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

# ---- Chargement du modèle ----
print("Chargement du modèle (peut prendre 1-2 min sur un Pi)...")
model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
print("Modèle chargé.")
logger.info(f"Gensim model loaded: {MODEL_PATH}")
logger.info(f"Vocab size: {len(model.key_to_index)}")


async def tirer_mot_wiktionnaire() -> str | None:
    """Sélectionne un mot aléatoire depuis une liste pré-filtrée de noms communs.
    
    Utilise les listes de mots de semanTikSl (dico_mm.txt ou dico_ms.txt) pour
    éviter les problèmes de l'API Wiktionary (403 Forbidden, mots bruités).
    
    Filtres appliqués:
    - Mots absents du modèle word2vec exclus.
    - Mots déjà utilisés exclus.
    - Mots avec une norme L2 trop faible (proxy pour la fréquence) exclus.
    """
    MIN_NORM = 10.0  # Seuil de norme L2 (plus élevé = mots plus fréquents)
    
    # Charger la liste de mots depuis le fichier local
    word_list_path = "dico_mm.txt"  # Liste moyenne (~50k mots)
    if not os.path.exists(word_list_path):
        word_list_path = "dico_ms.txt"  # Liste courte (~10k mots) si mm n'existe pas
    
    try:
        with open(word_list_path, "r", encoding="utf-8") as f:
            words = [line.strip().lower() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Fichier de mots introuvable: {word_list_path}")
        return None
    
    # Filtrer les mots valides (dans le modèle, pas déjà utilisés, norme L2 suffisante)
    valid_words = []
    for mot in words:
        # Écarte les mots composés/locutions
        if " " in mot or "-" in mot or "'" in mot:
            continue
        # Vérifier que le mot est dans le vocabulaire du modèle
        if mot not in model:
            continue
        # Vérifier que le mot n'a pas déjà été utilisé
        if mot in state.get("mots_utilises", []):
            continue
        # Filtre par norme L2
        norm = np.linalg.norm(model[mot])
        if norm < MIN_NORM:
            continue
        valid_words.append(mot)
    
    if not valid_words:
        logger.error("Aucun mot valide trouvé dans la liste. Vérifiez que dico_mm.txt ou dico_ms.txt existe et contient des mots valides.")
        return None
    
    # Sélectionner aléatoirement parmi les mots valides
    mot = random.choice(valid_words)
    logger.info(f"Selected word: '{mot}' (L2 norm: {np.linalg.norm(model[mot]):.2f})")
    return mot


def load_state() -> dict:
    """Charge l'historique (mot du jour + stats joueurs) depuis le disque,
    ou l'initialise s'il n'existe pas encore. Complète les clés manquantes
    si le fichier vient d'une version antérieure du script."""
    defaults = {
        "target": None,
        "current_date": None,
        "found": False,
        "attempts_today": 0,
        "players": {},
        "guesses_today": {},
        "mots_utilises": [],
        "neighbors": [],  # 100 plus proches voisins du mot cible
    }
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            loaded = json.load(f)
        for key, value in defaults.items():
            loaded.setdefault(key, value)
        return loaded
    return defaults


def save_state():
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


state = load_state()


async def nouveau_mot_du_jour():
    """Tire un nouveau mot, l'enregistre comme mot du jour, et réinitialise
    l'état de la partie. Précalcule également les 100 plus proches voisins."""
    mot = await tirer_mot_wiktionnaire()
    if mot is None:
        logger.error("Failed to select a word")
        return

    today = str(datetime.date.today())
    state["target"] = mot
    state["current_date"] = today
    state["found"] = False
    state["attempts_today"] = 0
    state["guesses_today"] = {}
    
    # Précalculer les 100 plus proches voisins
    try:
        neighbors = [word for word, _ in model.most_similar(mot, topn=100)]
        state["neighbors"] = neighbors
        logger.info(f"Precomputed 100 neighbors for '{mot}'")
    except KeyError as e:
        logger.error(f"Failed to compute neighbors for '{mot}': {e}")
        state["neighbors"] = []
    
    state["mots_utilises"].append(mot)
    save_state()
    print(f"Nouveau mot du jour tiré : {mot}")


async def check_reset():
    """Si aucun mot n'a encore été tiré aujourd'hui (premier lancement du
    bot ce jour-là, ou redémarrage après minuit sans que la tâche planifiée
    ait tourné), en tire un. Vérifie aussi que le mot cible est dans le modèle."""
    today = str(datetime.date.today())
    if state["current_date"] != today or state["target"] is None:
        await nouveau_mot_du_jour()
    elif state["target"] not in model:
        # Mot cible invalide, on en tire un nouveau
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
    save_state()


def record_guess(word: str, temp: float):
    """Garde le meilleur score obtenu pour ce mot aujourd'hui."""
    existing = state["guesses_today"].get(word)
    if existing is None or temp > existing:
        state["guesses_today"][word] = temp


def make_bar(temp: float, length: int = 10) -> str:
    """Construit une barre de progression en blocs Unicode."""
    pct = max(0.0, min(100.0, temp))
    filled = round(pct / 100 * length)
    return "█" * filled + "░" * (length - filled)


def build_proches_embed() -> discord.Embed:
    """Construit l'embed du top 10 des mots les plus proches proposés aujourd'hui."""
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


# ---- Bot Discord ----
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
        save_state()
        record_result(str(message.author.id), state["attempts_today"], gagne=True)
        await message.reply(
            f"🎉 Trouvé en {state['attempts_today']} coups ! "
            f"Le mot était **{target}**"
        )
        await message.channel.send(embed=build_proches_embed())
        return

    if guess not in model:
        await message.reply("❌ Mot inconnu du dictionnaire.")
        return

    # Calculer la similarité
    sim = model.similarity(guess, target)
    
    # Obtenir le rang dans les voisins précalculés (si disponible)
    neighbors = state.get("neighbors", [])
    rank = neighbors.index(guess) + 1 if guess in neighbors else None
    
    # Appliquer une échelle logarithmique pour compresser les scores
    # Clamp similarity to valid range [0, 1] to avoid log1p issues
    sim = max(0.0, min(1.0, sim))
    compressed_sim = np.log1p(sim * 10) / np.log1p(10)  # Échelle [0, 1]
    compressed_temp = round(float(compressed_sim) * 100, 1)
    
    # Formater la réponse
    if rank is not None and 1 <= rank <= 100:
        # Afficher la barre de progression + le rang
        emoji = "🔥" if compressed_temp > 60 else ("☁️" if compressed_temp > 30 else "❄️")
        response = f"{emoji} `{make_bar(compressed_temp)}` {compressed_temp}% (Rank: {rank}/100)"
    else:
        # Afficher uniquement la similarité en décimal (pas de barre)
        response = f"{sim:.2f}"
    
    record_guess(guess, compressed_temp)
    save_state()
    await message.reply(content=response, embed=build_proches_embed())


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


client.run(TOKEN)
