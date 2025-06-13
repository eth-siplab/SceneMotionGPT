#!/bin/bash

# --- CONFIGURATION ---
# Full path to the Snap version of ffmpeg to ensure we use the right one.
FFMPEG_CMD="/snap/bin/ffmpeg"

# Path to the font file you want to use.
FONT_PATH="/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

# --- SCRIPT LOGIC ---
# GIF_DIRECTORY="/home/dominik/Documents/repos/MotionGPT/results/mgpt/debug--Instruct_HumanML3D/samples_2025-06-13-15-36-26"
GIF_DIRECTORY="/home/dominik/Documents/repos/MotionGPT/results/mgpt/debug--Instruct_HumanML3D/samples_2025-06-13-13-32-44_m2t"

# --- VALIDATION ---
if [ -z "$GIF_DIRECTORY" ]; then
  echo "Error: You must provide the path to your GIF folder."
  echo "Usage: ./process_gifs.sh /path/to/your/folder"
  exit 1
fi

if [ ! -d "$GIF_DIRECTORY" ]; then
  echo "Error: Directory not found at '$GIF_DIRECTORY'"
  exit 1
fi

# --- PROCESSING ---
echo "Changing to directory: $GIF_DIRECTORY"
cd "$GIF_DIRECTORY"

echo "Starting GIF processing with high-quality two-pass method..."
WRAP_WIDTH=40 # <--- NEW CONFIGURATION VARIABLE

for gif_in in *_in.gif; do
  base="${gif_in%_in.gif}"
  txt_out="${base}_out.txt"
  gif_out="${base}_final.gif"
  palette="temp_palette.png"

  if [ -f "$txt_out" ]; then
    echo "Processing: ${gif_in} -> ${gif_out}"

    # --- NEW: Read text and wrap it ---
    # Read the entire text file into a variable.
    original_text=$(cat "$txt_out")
    # Use 'fold' to wrap the text at the specified width.
    wrapped_text=$(echo "$original_text" | fold -s -w "$WRAP_WIDTH")
    # --- END NEW ---

    # MODIFIED: Use `text=` instead of `textfile=`. Note the single quotes
    # around '$wrapped_text' to preserve the newlines.
    drawtext_filter="drawtext=fontfile=$FONT_PATH:text='$wrapped_text':x=(w-text_w)/2:y=h-text_h-20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5"

    # PASS 1: Generate the palette.
    "$FFMPEG_CMD" -i "$gif_in" -vf "$drawtext_filter,palettegen" -y "$palette"

    # PASS 2: Create the final GIF.
    "$FFMPEG_CMD" -i "$gif_in" -i "$palette" -filter_complex "$drawtext_filter[v];[v][1:v]paletteuse" -y "$gif_out"

    rm "$palette"
  else
    if [ "$gif_in" == "*_in.gif" ]; then
        echo "No '*_in.gif' files found in this directory."
        break
    fi
    echo "Skipping ${gif_in}: Matching text file '${txt_out}' not found."
  fi
done

echo "All done!"