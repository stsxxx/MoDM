#!/bin/bash

# Detect diffusers installation path using Python
DIFFUSERS_BASE=$(python3 -c "import diffusers, os; print(os.path.dirname(diffusers.__file__))")

if [[ ! -d "$DIFFUSERS_BASE" ]]; then
    echo "❌ Could not locate diffusers package. Is it installed?"
    exit 1
fi

echo "✅ Found diffusers at: $DIFFUSERS_BASE"

# Current directory with your replacement files
SRC_DIR=$(pwd)

# List of files to replace
FILES=("pipeline_flux.py" "pipeline_sana.py" "pipeline_stable_diffusion_3.py")

for file in "${FILES[@]}"; do
    # Search for matching file inside diffusers base
    DEST=$(find "$DIFFUSERS_BASE" -type f -name "$file")

    if [[ -f "$SRC_DIR/$file" ]]; then
        if [[ -n "$DEST" ]]; then
            echo "🔄 Replacing $DEST with $SRC_DIR/$file"
            cp "$SRC_DIR/$file" "$DEST"
        else
            echo "⚠️ Could not find $file inside $DIFFUSERS_BASE"
        fi
    else
        echo "⚠️ Source file $SRC_DIR/$file does not exist"
    fi
done

echo "✅ Replacement complete."
