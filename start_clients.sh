#!/bin/bash

ALPHA=0.1

for i in {0..9}
do
    osascript -e "tell app \"Terminal\" to do script \"cd $(pwd) && python3 src/client.py --cid $i --alpha $ALPHA\""

    sleep 2
done