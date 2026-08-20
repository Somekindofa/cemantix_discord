#!/usr/bin/env python3
"""
Script pour filtrer Lexique TSV et extraire uniquement les noms communs (NOM).

Usage:
    python filter_lexique_noms.py /chemin/vers/Lexique4.tsv [options]
    
    Options:
        --min-freq N    Filtrer les mots avec fréquence >= N (défaut: 0 = tous)
        --max-words N   Limiter à N mots (défaut: tous)
        --output FILE   Fichier de sortie (défaut: data/vocab/vocab.txt)
    
    Ou place le fichier dans data/vocab/ et lance:
    python filter_lexique_noms.py
"""

import sys
import csv
import argparse
from pathlib import Path

# Chemins
VOCAB_DIR = Path("data/vocab")
DEFAULT_INPUT = VOCAB_DIR / "Lexique4.tsv"
DEFAULT_OUTPUT = VOCAB_DIR / "vocab.txt"

# Colonnes Lexique4
ORTHO_COLUMN = "ortho"
CGRAM_COLUMN = "Lexique4__CgramOrtho"
FREQ_COLUMN = "Lexique4__FreqOrtho"


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
        
        # Vérifier que les colonnes nécessaires existent
        if ORTHO_COLUMN not in reader.fieldnames:
            print(f"Erreur: Colonne '{ORTHO_COLUMN}' introuvable. Colonnes: {reader.fieldnames}")
            return 0
        
        if CGRAM_COLUMN not in reader.fieldnames:
            print(f"Erreur: Colonne '{CGRAM_COLUMN}' introuvable. Colonnes: {reader.fieldnames}")
            return 0
        
        # Filtrer les mots
        words_with_freq = []
        for row in reader:
            cgram = row.get(CGRAM_COLUMN, "").strip()
            # On veut uniquement les lignes où CgramOrtho est exactement "NOM"
            if cgram == "NOM":
                ortho = row.get(ORTHO_COLUMN, "").strip()
                freq = row.get(FREQ_COLUMN, "0").strip()
                
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
        default=str(DEFAULT_INPUT),
        help="Chemin vers le fichier TSV Lexique4 (défaut: data/vocab/Lexique4.tsv)"
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
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Erreur: Fichier introuvable: {input_path}")
        print(f"Usage: python {sys.argv[0]} /chemin/vers/Lexique4.tsv [options]")
        print(f"Par défaut, cherche: {DEFAULT_INPUT}")
        sys.exit(1)
    
    count = filter_lexique_noms(input_path, output_path, args.min_freq, args.max_words)
    
    if count > 0:
        print(f"\n✅ Vocabulaire prêt dans: {output_path}")
        print(f"   Tu peux maintenant lancer l'ingestion:")
        print(f"   python cemantix_bot.py --mode api --ingest-vocab")
    else:
        print("❌ Aucun nom trouvé. Vérifie le format du fichier.")
        sys.exit(1)
