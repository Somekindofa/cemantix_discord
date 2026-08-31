#!/bin/bash
set -euo pipefail

# Docker entrypoint script for Cemantix Discord Bot
# This script handles model and dictionary downloads before starting the bot

echo "========================================"
echo "Cemantix Discord Bot - Docker Entrypoint"
echo "========================================"

# Model configuration - 500-dim skip-gram, 298MB (better than 200-dim)
MODEL_NAME="frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin"
MODEL_URL="https://embeddings.net/embeddings/${MODEL_NAME}"
MODEL_MD5="af38908c244dce973e289c70a4ce7242"
MODEL_PATH="/app/${MODEL_NAME}"

# Dictionary files
DICT_MM_URL="https://raw.githubusercontent.com/TikSL/semanTikSl/main/dico_mm.txt"
DICT_MS_URL="https://raw.githubusercontent.com/TikSL/semanTikSl/main/dico_ms.txt"
DICT_MM_PATH="/app/dico_mm.txt"
DICT_MS_PATH="/app/dico_ms.txt"

# Function to check if a file exists and has content
file_exists_and_valid() {
    [[ -f "$1" ]] && [[ -s "$1" ]]
}

# Function to download a file with retry
download_file() {
    local url="$1"
    local dest="$2"
    local name="$3"
    
    echo "Downloading ${name}..."
    
    # Try with curl first, fallback to wget
    if command -v curl &> /dev/null; then
        curl -L -f --retry 3 --retry-delay 5 "$url" -o "$dest" || {
            echo "Failed to download ${name} with curl, trying wget..."
            if command -v wget &> /dev/null; then
                wget -O "$dest" "$url" || return 1
            else
                echo "Neither curl nor wget available, cannot download ${name}"
                return 1
            fi
        }
    elif command -v wget &> /dev/null; then
        wget -O "$dest" "$url" || return 1
    else
        echo "Neither curl nor wget available, cannot download ${name}"
        return 1
    fi
    
    echo "Successfully downloaded ${name}"
    return 0
}

# Function to verify MD5 checksum
verify_md5() {
    local file="$1"
    local expected_md5="$2"
    
    if command -v md5sum &> /dev/null; then
        local actual_md5=$(md5sum "$file" | awk '{print $1}')
    elif command -v md5 &> /dev/null; then
        local actual_md5=$(md5 -q "$file")
    else
        echo "Warning: Cannot verify MD5 checksum (no md5sum or md5 command)"
        return 0
    fi
    
    if [[ "$actual_md5" != "$expected_md5" ]]; then
        echo "MD5 mismatch for $file: expected $expected_md5, got $actual_md5"
        return 1
    fi
    
    return 0
}

# Check and download model if needed
if ! file_exists_and_valid "$MODEL_PATH"; then
    echo "Model file not found or empty: $MODEL_NAME"
    download_file "$MODEL_URL" "$MODEL_PATH" "model (298MB)"
    
    # Verify MD5
    if ! verify_md5 "$MODEL_PATH" "$MODEL_MD5"; then
        echo "Warning: Model MD5 checksum could not be verified or failed"
        echo "The file may be corrupted. Try deleting it and restarting."
    fi
else
    echo "Model file already exists: $MODEL_NAME"
    
    # Verify MD5 of existing file
    if ! verify_md5 "$MODEL_PATH" "$MODEL_MD5"; then
        echo "Existing model MD5 mismatch! Redownloading..."
        rm "$MODEL_PATH"
        download_file "$MODEL_URL" "$MODEL_PATH" "model (298MB)"
    fi
fi

# Check and download dictionary files
if ! file_exists_and_valid "$DICT_MM_PATH"; then
    download_file "$DICT_MM_URL" "$DICT_MM_PATH" "dico_mm.txt"
else
    echo "dico_mm.txt already exists"
fi

if ! file_exists_and_valid "$DICT_MS_PATH"; then
    download_file "$DICT_MS_URL" "$DICT_MS_PATH" "dico_ms.txt"
else
    echo "dico_ms.txt already exists"
fi

# Create data directory if it doesn't exist
mkdir -p /app/data/vocab

# Check for .env file
echo ""
echo "Checking for .env file..."
if [[ -f "/app/.env" ]]; then
    echo ".env file found"
else
    echo "WARNING: No .env file found!"
    echo "The bot will fail to start without DISCORD_TOKEN and CHANNEL_ID."
    echo "Create a .env file with:"
    echo "  DISCORD_TOKEN=your_bot_token"
    echo "  CHANNEL_ID=your_channel_id"
fi

echo ""
echo "========================================"
echo "Starting Cemantix Discord Bot..."
echo "========================================"
echo ""

# Execute the original command
# If no command was provided, default to running the bot
exec "$@"
