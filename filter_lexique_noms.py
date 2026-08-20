#!/usr/bin/env python3
"""
Script pour filtrer Lexique TSV et extraire uniquement les noms communs (NOM).

Usage:
    python filter_lexique_noms.py [/chemin/vers/Lexique4.tsv] [options]
    
    Options:
        --min-freq N    Filtrer les mots avec fréquence >= N (défaut: 0 = tous)
        --max-words N   Limiter à N mots (défaut: tous)
        --output FILE   Fichier de sortie (défaut: data/vocab/vocab.txt)
    
    Si aucun chemin n'est spécifié, le script cherche automatiquement dans:
    - data/vocab/Lexique4.tsv
    - Lexique4.tsv (répertoire courant)
    - /home/somekindofathing/cemantix-bot/data/vocab/Lexique4.tsv
    - ~/cemantix-bot/data/vocab/Lexique4.tsv
    - ../data/vocab/Lexique4.tsv
    
    Supporte plusieurs formats de colonnes:
    - Format standard: ortho, Lexique4__CgramOrtho, Lexique4__FreqOrtho
    - Format numéroté: 1_Mot, 6_CgramOrtho, 10_FreqMot/11_FreqOrtho
"""

import sys
import csv
import argparse
from pathlib import Path

# Chemins
VOCAB_DIR = Path("data/vocab")
DEFAULT_INPUT = VOCAB_DIR / "Lexique4.tsv"
DEFAULT_OUTPUT = VOCAB_DIR / "vocab.txt"

# Colonnes Lexique4 - plusieurs formats possibles
# Format standard : ortho, Lexique4__CgramOrtho, Lexique4__FreqOrtho
# Format numéroté : 1_Mot, 6_CgramOrtho, 10_FreqMot/11_FreqOrtho
ORTHO_COLUMN_NAMES = ["ortho", "1_Mot", "Mot"]
CGRAM_COLUMN_NAMES = ["Lexique4__CgramOrtho", "6_CgramOrtho", "CgramOrtho", "Cgram"]
FREQ_COLUMN_NAMES = ["Lexique4__FreqOrtho", "11_FreqOrtho", "10_FreqMot", "FreqOrtho", "FreqMot", "frequency"]


def find_lexique_file(input_path: Path = None) -> Path:
    """
    Trouve le fichier Lexique4.tsv dans les emplacements courants.
    
    Args:
        input_path: Chemin spécifié par l'utilisateur (si None, cherche automatiquement)
    
    Returns:
        Path vers le fichier trouvé, ou None
    """
    if input_path and input_path.exists():
        return input_path
    
    # Liste des chemins à essayer
    search_paths = [
        VOCAB_DIR / "Lexique4.tsv",
        Path("Lexique4.tsv"),
        Path("/home/somekindofathing/cemantix-bot/data/vocab/Lexique4.tsv"),
        Path("~/cemantix-bot/data/vocab/Lexique4.tsv").expanduser(),
        Path("../data/vocab/Lexique4.tsv"),
        Path("../../data/vocab/Lexique4.tsv"),
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None


def find_column(fieldnames, possible_names):
    """
    Trouve une colonne parmi plusieurs noms possibles.
    
    Args:
        fieldnames: Liste des noms de colonnes disponibles
        possible_names: Liste des noms possibles pour la colonne
    
    Returns:
        Le nom de la colonne trouvée, ou None
    """
    for name in possible_names:
        if name in fieldnames:
            return name
    return None


def filter_lexique_noms(input_path: Path, output_path: Path, min_freq: int = 0, max_words: int = None) -> int:
    """
    Filtre le fichier TSV Lexique pour ne garder que les noms (NOM).
    
    Args:
        input_path: Chemin vers le fichier TSV Lexique
        output_path: Chemin vers le fichier de sortie
        min_freq: Fréquence minimale (0 = pas de filtre)
        max_words: Nombre maximum de mots à garder (None = tous)
    
    Returns:
        Nombre de mots filtrés
    """
    if not input_path.exists():
        print(f"Erreur: Fichier introuvable: {input_path}")
        return 0
    
    # Lire le fichier TSV
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        # Trouver les colonnes nécessaires
        ortho_col = find_column(reader.fieldnames, ORTHO_COLUMN_NAMES)
        cgram_col = find_column(reader.fieldnames, CGRAM_COLUMN_NAMES)
        freq_col = find_column(reader.fieldnames, FREQ_COLUMN_NAMES)
        
        if ortho_col is None:
            print(f"Erreur: Aucune colonne d'orthographe trouvée.")
            print(f"Colonnes disponibles: {reader.fieldnames}")
            print(f"Colonnes recherchées: {ORTHO_COLUMN_NAMES}")
            return 0
        
        if cgram_col is None:
            print(f"Erreur: Aucune colonne CgramOrtho trouvée.")
            print(f"Colonnes disponibles: {reader.fieldnames}")
            print(f"Colonnes recherchées: {CGRAM_COLUMN_NAMES}")
            return 0
        
        print(f"Colonnes utilisées: ortho='{ortho_col}', cgram='{cgram_col}', freq='{freq_col or 'aucune'}'")
        
        # Filtrer les mots
        words_with_freq = []
        for row in reader:
            cgram = row.get(cgram_col, "").strip()
            # On veut uniquement les lignes où CgramOrtho est exactement "NOM"
            if cgram == "NOM":
                ortho = row.get(ortho_col, "").strip()
                freq = row.get(freq_col, "0").strip() if freq_col else "0"
                
                # Nettoyer le mot
                word = ortho.lower()
                
                # Garder uniquement les mots alphabétiques sans espaces/tirets/apostrophes
                if word.isalpha() and len(word) >= 3:
                    try:
                        freq_val = float(freq) if freq else 0
                    except ValueError:
                        freq_val = 0
                    words_with_freq.append((word, freq_val))
    
    # Appliquer les filtres
    if min_freq > 0:
        words_with_freq = [(w, f) for w, f in words_with_freq if f >= min_freq]
    
    # Trier par fréquence décroissante pour garder les mots les plus courants
    words_with_freq.sort(key=lambda x: -x[1])
    
    # Extraire les mots
    words = [w for w, f in words_with_freq]
    
    # Limiter le nombre de mots
    if max_words is not None:
        words = words[:max_words]
    
    # Écrire le fichier de sortie
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words))
    
    print(f"Filtrage terminé: {len(words)} noms extraits")
    print(f"  - Fréquence minimale: {min_freq}")
    print(f"  - Limite: {max_words if max_words else 'aucun'}")
    print(f"Fichier de sortie: {output_path}")
    return len(words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filtrer Lexique4 TSV pour extraire les noms communs (NOM)"
    )
    parser.add_argument(
        "input", 
        nargs='?', 
        default=None,
        help="Chemin vers le fichier TSV Lexique4 (optionnel)"
    )
    parser.add_argument(
        "--min-freq", 
        type=int, 
        default=0,
        help="Fréquence minimale pour filtrer les mots (défaut: 0 = tous)"
    )
    parser.add_argument(
        "--max-words", 
        type=int, 
        default=None,
        help="Nombre maximum de mots à garder (défaut: tous)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(DEFAULT_OUTPUT),
        help="Fichier de sortie (défaut: data/vocab/vocab.txt)"
    )
    
    args = parser.parse_args()
    
    # Trouver le fichier
    input_path = None
    if args.input:
        input_path = Path(args.input)
    
    input_path = find_lexique_file(input_path)
    output_path = Path(args.output)
    
    if not input_path:
        print(f"Erreur: Fichier Lexique4.tsv introuvable.")
        print(f"J'ai cherché dans:")
        print(f"  - data/vocab/Lexique4.tsv")
        print(f"  - Lexique4.tsv (répertoire courant)")
        print(f"  - /home/somekindofathing/cemantix-bot/data/vocab/Lexique4.tsv")
        print(f"  - ~/cemantix-bot/data/vocab/Lexique4.tsv")
        print(f"  - ../data/vocab/Lexique4.tsv")
        print(f"\nSpécifiez le chemin avec: python {sys.argv[0]} /chemin/vers/Lexique4.tsv")
        sys.exit(1)
    
    print(f"Fichier trouvé: {input_path}")
    count = filter_lexique_noms(input_path, output_path, args.min_freq, args.max_words)
    
    if count > 0:
        print(f"\n✅ Vocabulaire prêt dans: {output_path}")
        print(f"   Tu peux maintenant lancer l'ingestion:")
        print(f"   python cemantix_bot.py --mode api --ingest-vocab")
    else:
        print("❌ Aucun nom trouvé. Vérifie le format du fichier.")
        sys.exit(1)
