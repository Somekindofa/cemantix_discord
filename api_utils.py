"""
Utilitaires pour l'API Infomaniak et la gestion des embeddings.

Ce module gère:
- Les appels à l'API Infomaniak pour obtenir des embeddings
- Le chargement et le stockage des embeddings de vocabulaire
- La gestion du vocabulaire local
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration de l'API Infomaniak
INFOMANIAK_PRODUCT_ID = os.environ.get("INFOMANIAK_PRODUCT_ID")
INFOMANIAK_API_KEY = os.environ.get("INFOMANIAK_API_KEY")

# Configuration des chemins
DATA_DIR = Path("data")
VOCAB_DIR = DATA_DIR / "vocab"
EMBEDDINGS_FILE = VOCAB_DIR / "embeddings.npy"
VOCAB_FILE = VOCAB_DIR / "vocab.json"
WORD_TO_INDEX_FILE = VOCAB_DIR / "word_to_index.json"
STATE_API_FILE = Path("state_api.json")

# Configuration du logger
logger = logging.getLogger(__name__)

# Modèle d'embedding Infomaniak (nom du modèle à utiliser)
# Selon la documentation: https://developer.infomaniak.com/docs/api/post/2/ai/%7Bproduct_id%7D/openai/v1/embeddings
# Le modèle par défaut est probablement "text-embedding-ada-002" ou similaire
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_infomaniak_endpoint(product_id: str) -> str:
    """Retourne l'URL de l'endpoint pour les embeddings."""
    return f"https://api.infomaniak.com/2/ai/{product_id}/openai/v1/embeddings"


async def call_infomaniak_embedding(text: str, product_id: str = None, api_key: str = None) -> Optional[List[float]]:
    """
    Appelle l'API Infomaniak pour obtenir l'embedding d'un texte.
    
    Args:
        text: Le texte à embedder
        product_id: L'ID du produit Infomaniak (si None, utilise la variable d'environnement)
        api_key: La clé API Infomaniak (si None, utilise la variable d'environnement)
    
    Returns:
        Liste des floats représentant l'embedding, ou None en cas d'erreur
    """
    if product_id is None:
        product_id = INFOMANIAK_PRODUCT_ID
    if api_key is None:
        api_key = INFOMANIAK_API_KEY
    
    if not product_id or not api_key:
        logger.error("INFOMANIAK_PRODUCT_ID ou INFOMANIAK_API_KEY non configurés")
        return None
    
    import aiohttp
    
    url = get_infomaniak_endpoint(product_id)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": text,
        "model": DEFAULT_EMBEDDING_MODEL
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extraire l'embedding du premier élément
                    if "data" in data and len(data["data"]) > 0:
                        embedding = data["data"][0]["embedding"]
                        logger.info(f"Embedding obtenu pour '{text[:50]}...' (dim: {len(embedding)})")
                        return embedding
                    else:
                        logger.error(f"Réponse API invalide: {data}")
                        return None
                elif response.status == 429:
                    logger.error("Rate limit dépassé pour l'API Infomaniak")
                    return None
                elif response.status == 401:
                    logger.error("Authentification échouée pour l'API Infomaniak")
                    return None
                elif response.status == 404:
                    logger.error(f"Endpoint non trouvé: {url}")
                    return None
                else:
                    error_text = await response.text()
                    logger.error(f"Erreur API Infomaniak (status {response.status}): {error_text}")
                    return None
    except aiohttp.ClientError as e:
        logger.error(f"Erreur de connexion à l'API Infomaniak: {e}")
        return None
    except Exception as e:
        logger.error(f"Erreur inattendue lors de l'appel API: {e}")
        return None


async def call_infomaniak_embedding_batch(texts: List[str], product_id: str = None, api_key: str = None, batch_size: int = 10) -> Optional[List[List[float]]]:
    """
    Appelle l'API Infomaniak pour obtenir des embeddings pour plusieurs textes.
    Gère les appels par batch pour respecter les limites de l'API.
    
    Args:
        texts: Liste des textes à embedder
        product_id: L'ID du produit Infomaniak
        api_key: La clé API Infomaniak
        batch_size: Taille des batches (max 10 pour éviter les rate limits)
    
    Returns:
        Liste des embeddings, ou None en cas d'erreur
    """
    if product_id is None:
        product_id = INFOMANIAK_PRODUCT_ID
    if api_key is None:
        api_key = INFOMANIAK_API_KEY
    
    if not product_id or not api_key:
        logger.error("INFOMANIAK_PRODUCT_ID ou INFOMANIAK_API_KEY non configurés")
        return None
    
    import aiohttp
    
    url = get_infomaniak_endpoint(product_id)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    all_embeddings = []
    
    # Traiter par batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        payload = {
            "input": batch,
            "model": DEFAULT_EMBEDDING_MODEL
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and len(data["data"]) == len(batch):
                            batch_embeddings = [item["embedding"] for item in data["data"]]
                            all_embeddings.extend(batch_embeddings)
                            logger.info(f"Batch {i//batch_size + 1}: {len(batch)} embeddings obtenus")
                        else:
                            logger.error(f"Réponse API invalide pour le batch: {data}")
                            return None
                    elif response.status == 429:
                        # Rate limit, attendre avant de continuer
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limit atteint, attente de {retry_after} secondes...")
                        time.sleep(retry_after)
                        # Réessayer ce batch
                        i -= batch_size
                        continue
                    else:
                        error_text = await response.text()
                        logger.error(f"Erreur API pour le batch (status {response.status}): {error_text}")
                        return None
        except Exception as e:
            logger.error(f"Erreur lors de l'appel batch: {e}")
            return None
    
    return all_embeddings


def load_vocabulary(vocab_path: Path = None) -> List[str]:
    """
    Charge le vocabulaire depuis un fichier texte.
    
    Args:
        vocab_path: Chemin vers le fichier de vocabulaire (si None, utilise VOCAB_DIR/vocab.txt)
    
    Returns:
        Liste des mots du vocabulaire
    """
    if vocab_path is None:
        vocab_path = VOCAB_DIR / "vocab.txt"
    
    if not vocab_path.exists():
        logger.error(f"Fichier de vocabulaire introuvable: {vocab_path}")
        return []
    
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        # Nettoyer les mots
        cleaned_words = []
        for word in words:
            # Supprimer les mots avec espaces, tirets ou apostrophes
            if ' ' in word or '-' in word or "'" in word:
                continue
            # Garder uniquement les mots avec des lettres (pas de chiffres ou caractères spéciaux)
            if word.isalpha():
                cleaned_words.append(word)
        
        logger.info(f"Vocabulaire chargé: {len(cleaned_words)} mots depuis {vocab_path}")
        return cleaned_words
    except Exception as e:
        logger.error(f"Erreur lors du chargement du vocabulaire: {e}")
        return []


def save_embeddings(embeddings: np.ndarray, words: List[str]):
    """
    Sauvegarde les embeddings et les métadonnées du vocabulaire.
    
    Args:
        embeddings: Tableau NumPy des embeddings
        words: Liste des mots correspondants
    """
    try:
        # Créer le répertoire si nécessaire
        VOCAB_DIR.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les embeddings en float16 pour économiser de l'espace
        embeddings_float16 = embeddings.astype(np.float16)
        np.save(EMBEDDINGS_FILE, embeddings_float16)
        
        # Sauvegarder la liste des mots
        vocab_data = {
            "words": words,
            "count": len(words),
            "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else None
        }
        with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        # Sauvegarder le mappage mot -> index
        word_to_index = {word: idx for idx, word in enumerate(words)}
        with open(WORD_TO_INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(word_to_index, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Embeddings sauvegardés: {len(words)} mots, {embeddings.shape[1]} dimensions")
        logger.info(f"Taille des embeddings: {embeddings_float16.nbytes / 1024 / 1024:.2f} Mo")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des embeddings: {e}")
        return False


def load_embeddings() -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[Dict[str, int]]]:
    """
    Charge les embeddings et le vocabulaire depuis les fichiers.
    
    Returns:
        Tuple de (embeddings, words, word_to_index) ou (None, None, None) en cas d'erreur
    """
    try:
        if not EMBEDDINGS_FILE.exists() or not VOCAB_FILE.exists() or not WORD_TO_INDEX_FILE.exists():
            logger.warning("Fichiers d'embeddings introuvables")
            return None, None, None
        
        # Charger les embeddings
        embeddings = np.load(EMBEDDINGS_FILE)
        
        # Charger le vocabulaire
        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        words = vocab_data.get("words", [])
        
        # Charger le mappage
        with open(WORD_TO_INDEX_FILE, 'r', encoding='utf-8') as f:
            word_to_index = json.load(f)
        
        logger.info(f"Embeddings chargés: {len(words)} mots, {embeddings.shape[1]} dimensions")
        return embeddings, words, word_to_index
    except Exception as e:
        logger.error(f"Erreur lors du chargement des embeddings: {e}")
        return None, None, None


async def ingest_vocabulary(vocab_path: Path = None, batch_size: int = 10) -> bool:
    """
    Ingestion complète du vocabulaire: charge les mots, appelle l'API pour les embeddings,
    et sauvegarde le tout.
    
    Args:
        vocab_path: Chemin vers le fichier de vocabulaire
        batch_size: Taille des batches pour l'API
    
    Returns:
        True si succès, False sinon
    """
    logger.info("Début de l'ingestion du vocabulaire...")
    
    # Charger le vocabulaire
    words = load_vocabulary(vocab_path)
    if not words:
        logger.error("Aucun mot valide trouvé dans le vocabulaire")
        return False
    
    logger.info(f"Ingestion de {len(words)} mots...")
    
    # Obtenir les embeddings par batch
    all_embeddings = []
    for i in range(0, len(words), batch_size):
        batch_words = words[i:i + batch_size]
        logger.info(f"Traitement du batch {i//batch_size + 1}/{len(words)//batch_size + 1}...")
        
        batch_embeddings = await call_infomaniak_embedding_batch(batch_words)
        if batch_embeddings is None:
            logger.error(f"Échec de l'obtention des embeddings pour le batch {i//batch_size + 1}")
            return False
        
        all_embeddings.extend(batch_embeddings)
        
        # Petite pause entre les batches pour éviter les rate limits
        time.sleep(1)
    
    # Convertir en tableau NumPy
    embeddings_array = np.array(all_embeddings)
    
    # Sauvegarder
    return save_embeddings(embeddings_array, words)


def get_similarity(word1: str, word2: str, embeddings: np.ndarray, word_to_index: Dict[str, int]) -> float:
    """
    Calcule la similarité cosinus entre deux mots.
    
    Args:
        word1: Premier mot
        word2: Deuxième mot
        embeddings: Tableau des embeddings
        word_to_index: Mappage mot -> index
    
    Returns:
        Similarité cosinus entre -1 et 1
    """
    if word1 not in word_to_index or word2 not in word_to_index:
        logger.warning(f"Mot(s) non trouvé(s) dans le vocabulaire: {word1}, {word2}")
        return 0.0
    
    idx1 = word_to_index[word1]
    idx2 = word_to_index[word2]
    
    emb1 = embeddings[idx1]
    emb2 = embeddings[idx2]
    
    # Similarité cosinus
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def get_nearest_neighbors(word: str, embeddings: np.ndarray, word_to_index: Dict[str, int], words: List[str], topn: int = 100) -> List[Tuple[str, float]]:
    """
    Obtient les N mots les plus proches d'un mot donné.
    
    Args:
        word: Mot de référence
        embeddings: Tableau des embeddings
        word_to_index: Mappage mot -> index
        words: Liste des mots
        topn: Nombre de voisins à retourner
    
    Returns:
        Liste de tuples (mot, similarité) triés par similarité décroissante
    """
    if word not in word_to_index:
        logger.warning(f"Mot non trouvé dans le vocabulaire: {word}")
        return []
    
    idx = word_to_index[word]
    emb = embeddings[idx]
    
    # Calculer les similarités avec tous les mots
    similarities = []
    for i, w in enumerate(words):
        if w == word:
            continue
        sim = get_similarity(word, w, embeddings, word_to_index)
        similarities.append((w, sim))
    
    # Trier par similarité décroissante
    similarities.sort(key=lambda x: -x[1])
    
    return similarities[:topn]


def get_rank(word: str, target: str, embeddings: np.ndarray, word_to_index: Dict[str, int], words: List[str]) -> Optional[int]:
    """
    Obtient le rang d'un mot par rapport à un mot cible (combien de mots sont plus similaires).
    
    Args:
        word: Mot à évaluer
        target: Mot cible
        embeddings: Tableau des embeddings
        word_to_index: Mappage mot -> index
        words: Liste des mots
    
    Returns:
        Rang du mot (1 = le plus similaire après la cible elle-même), ou None si erreur
    """
    if word not in word_to_index or target not in word_to_index:
        return None
    
    if word == target:
        return 0
    
    # Obtenir les voisins de la cible
    neighbors = get_nearest_neighbors(target, embeddings, word_to_index, words, topn=len(words))
    
    # Trouver le rang du mot
    for rank, (neighbor_word, _) in enumerate(neighbors, start=1):
        if neighbor_word == word:
            return rank
    
    return None


def compress_similarity(sim: float) -> float:
    """
    Applique une échelle logarithmique pour compresser les scores de similarité.
    
    Args:
        sim: Similarité cosinus entre 0 et 1
    
    Returns:
        Score compressé entre 0 et 100
    """
    sim = max(0.0, min(1.0, sim))
    compressed_sim = np.log1p(sim * 10) / np.log1p(10)
    return round(float(compressed_sim) * 100, 1)
