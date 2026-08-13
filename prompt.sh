#!/bin/sh

seed_file="seed.txt"

cp iris.exe iris_copy.exe || exit 1

stop_requested=0
echo "Generating from seed $seed. Press Q to stop after the current image."

while :; do
	output="Iris $(date -u '+%Y-%m-%d %H.%M.%S') seed $seed.png"

	./iris_copy.exe -d zimage-turbo -o "$output" -s 8 --transformer "E:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\zImageTurboQuantized_fp8ScaledE4m3fnKJ.safetensors" --gpu-friendly \
		-W 1120 -H 1440 --seed "$seed" -p \
		"Photograph of feminine athletic fit muscular woman Kristen Stewart, standing on a thick branch up in a tree on a warm sunny day. Her face is pretty and cute, her eyes are dark green, her hair is dark brown. She's wearing iridescent satin pajama pants and shirt. The Sun is shining on her pajamas, the background is the blue sky." &
	iris_pid=$!

	while kill -0 "$iris_pid" 2>/dev/null; do
		if [ -t 0 ]; then
			key=''
			if read -r -s -n 1 -t 0.2 key && { [ "$key" = Q ] || [ "$key" = q ]; }; then
				stop_requested=1
				echo
				echo "Q pressed; stopping after seed $seed finishes."
			fi
		else
			sleep 1
		fi
	done

	if ! wait "$iris_pid"; then
		echo "Generation failed for seed $seed; seed file was not advanced" >&2
		exit 1
	fi

	seed=$((seed + 1))
	printf '%s\n' "$seed" > "$seed_file"

	if [ "$stop_requested" -eq 1 ]; then
		break
	fi
done
