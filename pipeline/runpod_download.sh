# Included in the pod's /bin/sh command. The pinned Hugging Face file supports
# byte ranges. Verify each response before assembling the file for SHA-256.
download_weights() (
    set -eu
    url=$1 size=$2 output=$3
    parts=$(mktemp -d)
    meter=''
    trap 'if [ -n "$meter" ]; then kill "$meter" 2>/dev/null || true; fi; rm -rf "$parts"' EXIT
    width=$(( (size + 7) / 8 ))
    pids=''
    for i in 0 1 2 3 4 5 6 7; do
        first=$((i * width))
        last=$((first + width - 1))
        [ "$first" -lt "$size" ] || break
        if [ "$last" -ge "$size" ]; then last=$((size - 1)); fi
        expected=$((last - first + 1))
        part="$parts/part-$i"
        : > "$part"
        (
            http=$(curl -fsSL --connect-timeout 15 --speed-limit 16384 --speed-time 45 \
                --max-time 300 --retry 2 --retry-max-time 480 --max-filesize "$expected" \
                --range "$first-$last" --output "$part" --write-out '%{http_code}' \
                "$url&part=$i") || { rc=$?; echo "CHUCKLES_RANGE_ERROR $i $http $rc"; exit "$rc"; }
            bytes=$(wc -c < "$part")
            if [ "$http" != 206 ] || [ "$bytes" -ne "$expected" ]; then
                echo "CHUCKLES_RANGE_ERROR $i $http 1"
                exit 1
            fi
        ) &
        pids="$pids $!"
    done
    (
        while true; do
            wc -c "$parts"/part-* | awk 'END {print "CHUCKLES_BYTES " $1}'
            sleep 15 >/dev/null 2>&1
        done
    ) &
    meter=$!
    result=0
    for pid in $pids; do wait "$pid" || result=$?; done
    [ "$result" -eq 0 ] || exit "$result"
    kill "$meter" 2>/dev/null || true
    meter=''
    # Append and release one range at a time so a second full copy never
    # occupies the container disk.
    mv "$parts/part-0" "$output"
    for part in "$parts"/part-*; do
        [ -f "$part" ] || break
        cat "$part" >> "$output"
        rm "$part"
    done
    echo "CHUCKLES_BYTES $size"
)
