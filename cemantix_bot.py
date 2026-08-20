"""
Bot Discord type Cémantix, avec mot du jour tiré automatiquement du
Wiktionnaire (catégorie "Noms communs en français") chaque nuit à minuit.

Prérequis :
    pip install discord.py spacy requests python-dotenv
    python -m spacy download fr_core_news_sm

Usage :
    1. Copie .env.example vers .env et remplis TOKEN et CHANNEL_ID.
    2. python cemantix_bot.py
"""

import datetime
import json
import os
import random
from pathlib import Path

import discord
import requests
import spacy
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv

# ---- Config chargée depuis .env (voir .env.example) ----
load_dotenv()
TOKEN = os.environ["DISCORD_TOKEN"]
CHANNEL_ID = int(os.environ["CHANNEL_ID"])
STATE_FILE = Path("state.json")

WIKTIONNAIRE_API = "https://fr.wiktionary.org/w/api.php"
CATEGORIE = "Catégorie:Noms communs en français"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

# ---- Chargement du modèle spaCy ----
print("Chargement du modèle spaCy (peut prendre 1-2 min)...")
nlp = spacy.load("fr_core_news_sm")
print("Modèle chargé.")


def tirer_mot_wiktionnaire() -> str | None:
    """Interroge l'API du Wiktionnaire pour piocher un nom commun au hasard
    dans la catégorie dédiée. Retourne None si rien d'exploitable n'a été
    trouvé après plusieurs essais.
    
    Filtres appliqués:
    - Mots composés/locutions (espaces, tirets, apostrophes) exclus.
    - Mots absents du vocabulaire spaCy exclus.
    - Mots déjà utilisés exclus.
    - Seuls les noms (POS=NOUN) sont conservés.
    - Mots trop rares (prob > MIN_PROB) exclus.
    """
    MIN_PROB = -8.0  # Seuil de probabilité (plus bas = mots plus fréquents)
    for _ in range(15):
        lettre = random.choice(ALPHABET)
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": CATEGORIE,
            "cmlimit": 1,
            "cmstartsortkeyprefix": lettre,
            "format": "json",
        }
        try:
            resp = requests.get(WIKTIONNAIRE_API, params=params, timeout=10)
            resp.raise_for_status()
            membres = resp.json().get("query", {}).get("categorymembers", [])
        except requests.RequestException:
            continue

        if not membres:
            continue

        mot = membres[0]["title"].strip().lower()

        # On écarte les mots composés/locutions
        if " " in mot or "-" in mot or "'" in mot:
            continue
        
        # Vérifier que le mot est dans le vocabulaire spaCy
        if mot not in nlp.vocab:
            continue
        
        # Vérifier que le mot est un nom (POS=NOUN)
        doc = nlp(mot)
        # Accepter uniquement les mots à un seul token qui sont des noms
        if len(doc) == 1 and doc[0].pos_ != "NOUN":
            continue
        # Pour les mots multi-tokens (ex: "pomme de terre"), tous doivent être des noms
        if len(doc) > 1 and not all(token.pos_ == "NOUN" for token in doc):
            continue
        
        # Filtre par probabilité (proxy pour la fréquence)
        # Note: token.prob est la log-probabilité (plus bas = plus fréquent)
        if all(token.prob > MIN_PROB for token in doc):
            continue
        
        # Vérifier que le mot n'a pas déjà été utilisé
        if mot in state.get("mots_utilises", []):
            continue

        return mot

    return None


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


def nouveau_mot_du_jour():
    """Tire un nouveau mot, l'enregistre comme mot du jour, et réinitialise
    l'état de la partie."""
    mot = tirer_mot_wiktionnaire()
    if mot is None:
        print("⚠️  Impossible de tirer un nouveau mot, on garde l'ancien.")
        return

    today = str(datetime.date.today())
    state["target"] = mot
    state["current_date"] = today
    state["found"] = False
    state["attempts_today"] = 0
    state["guesses_today"] = {}
    state["mots_utilises"].append(mot)
    save_state()
    print(f"Nouveau mot du jour tiré : {mot}")


def check_reset():
    """Si aucun mot n'a encore été tiré aujourd'hui (premier lancement du
    bot ce jour-là, ou redémarrage après minuit sans que la tâche planifiée
    ait tourné), en tire un. Vérifie aussi que le mot cible est dans le vocabulaire."""
    today = str(datetime.date.today())
    if state["current_date"] != today or state["target"] is None:
        nouveau_mot_du_jour()
    elif state["target"] not in nlp.vocab:
        # Mot cible invalide, on en tire un nouveau
        nouveau_mot_du_jour()


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
    """Construit une barre de progression en blocs Unicode (sans backticks,
    l'alignement se fait au niveau du bloc de code englobant)."""
    pct = max(0.0, min(100.0, temp))
    filled = round(pct / 100 * length)
    return "█" * filled + "░" * (length - filled)


def build_proches_embed() -> discord.Embed:
    """Construit l'embed du top 10 des mots les plus proches proposés aujourd'hui.
    Tout le tableau est dans un seul bloc de code pour que les colonnes s'alignent."""
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
    nouveau_mot_du_jour()
    channel = client.get_channel(CHANNEL_ID)
    if channel is not None:
        await channel.send("🌅 Nouveau mot du jour disponible, à vous de jouer !")


@client.event
async def on_ready():
    check_reset()
    if not tirage_minuit.is_running():
        tirage_minuit.start()
    print(f"Connecté en tant que {client.user}")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot or message.channel.id != CHANNEL_ID:
        return

    check_reset()
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

    if guess not in nlp.vocab:
        await message.reply("❌ Mot inconnu du dictionnaire.")
        return


    target = current_target()
    if target not in nlp.vocab:
        # Mot cible invalide, on en tire un nouveau et on réessaye
        nouveau_mot_du_jour()
        target = current_target()
        if target is None:
            await message.reply("⚠️ Impossible de tirer un nouveau mot. Réessayez plus tard.")
            return
    
    # Calcul de la similarité avec spaCy
    doc1 = nlp(guess)
    doc2 = nlp(target)
    sim = doc1.similarity(doc2)
    temp = round(float(sim) * 100, 1)
    emoji = "🔥" if temp > 60 else ("🌤️" if temp > 30 else "❄️")
    record_guess(guess, temp)
    save_state()
    await message.reply(
        content=f"{emoji} `{make_bar(temp)}` {temp}%", embed=build_proches_embed()
    )


@client.tree.command(name="start", description="Affiche l'état de la partie du jour")
async def start(interaction: discord.Interaction):
    check_reset()
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
