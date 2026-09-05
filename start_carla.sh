#!/usr/bin/env bash

SCRIPT="train/train_custom.py"
INTERVAL=5


while true; do
    if pgrep -f "python3 $SCRIPT" > /dev/null; then
        echo "OK"
    else
        echo "caido, lanzando"
        python3 "$SCRIPT" 
    fi
    sleep "$INTERVAL"
done
