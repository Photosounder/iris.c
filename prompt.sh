#!/bin/sh

seed=7
output="Iris $(date -u '+%Y-%m-%d %H.%M.%S') seed $seed.png"

cp iris.exe iris_copy.exe ; ./iris_copy.exe -d zimage-turbo -o "$output" -s 8 --transformer "E:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\zImageTurboQuantized_fp8ScaledE4m3fnKJ.safetensors" --gpu-friendly \
	-W 1120 -H 1440 --seed $seed -p \
"Photograph of feminine athletic fit muscular crossfitter woman Kristen Stewart, sitting on a thick branch up in a tree on a warm sunny day with one leg up. Her face is pretty and cute, her eyes are dark green, her hair is dark brown. She's wearing baggy silk pants that are loose at the ankles and an open cotton shirt. The Sun is shining on her skin, the background is the blue sky."
