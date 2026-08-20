# Vocabulaire Français

Ce dossier contient les fichiers de vocabulaire pour le mode API du bot.

## Sources Recommandées

### 1. Lexique 3.83 (Recommandé)
Le fichier **Lexique 3.83** contient ~140,000 mots français avec des métadonnées.
- **Téléchargement**: https://www.lexique.org/
- **Fichier**: `Lexique383.tsv` ou `Lexique383.txt`
- **Format**: Tab-separated avec colonnes: ortho, phon, lemme, cgram, genre, nombre, freqlemfilms2, freqlemlivres2, freqlemsous-titres2, freqtot, an, nbhomographes, nbcar, nbsyll, nbpoly, freqfilms2, freqlivres2, freqsous-titres2

### 2. Wiktionnaire (Alternative)
Liste de noms communs français extraits du Wiktionnaire.
- **Source**: https://fr.wiktionary.org
- **Méthode**: Utiliser l'API MediaWiki avec `categorymembers` pour la catégorie "Noms communs en français"

### 3. Fichiers existants du projet
Le projet utilise déjà:
- `dico_mm.txt` (~50k mots)
- `dico_ms.txt` (~10k mots)

Ces fichiers peuvent être copiés ici et utilisés comme base.

## Format Attendu

Le fichier de vocabulaire doit être un fichier texte simple avec:
- Un mot par ligne
- Encodage UTF-8
- Pas d'espaces, tirets ou apostrophes (mots composés exclus)
- Mots en minuscules

Exemple (`vocab.txt`):
```
abricot
absurde
accueil
achat
acide
... 
```

## Préparation du Vocabulaire

Pour préparer votre vocabulaire:

1. **Depuis Lexique 383**:
   ```bash
   # Extraire uniquement les mots (colonne 1) et filtrer
   cut -f1 Lexique383.tsv | grep -E '^[a-z]+$' | grep -v -E '[ -]' > vocab.txt
   ```

2. **Depuis dico_mm.txt**:
   ```bash
   cp dico_mm.txt data/vocab/vocab.txt
   ```

3. **Nettoyage supplémentaire**:
   ```bash
   # Supprimer les mots trop courts
   grep -E '^.{3,}$' vocab.txt > vocab_filtered.txt
   
   # Supprimer les mots avec caractères spéciaux
   grep -E '^[a-z]+$' vocab.txt > vocab_clean.txt
   ```

## Fichiers Générés

Après ingestion, les embeddings seront stockés dans:
- `data/vocab/embeddings.npy` - Tableau NumPy des embeddings (format float16)
- `data/vocab/vocab.json` - Liste des mots avec métadonnées
- `data/vocab/word_to_index.json` - Mappage mot -> index

## Usage

Pour ingérer un nouveau vocabulaire:
```bash
python cemantix_bot.py --ingest-vocab data/vocab/vocab.txt
```

Ou placer votre fichier dans `data/vocab/` et le bot le détectera automatiquement.
