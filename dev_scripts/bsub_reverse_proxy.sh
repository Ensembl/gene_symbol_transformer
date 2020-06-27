#!/bin/bash


COMPUTE_NODE="hx-noah-13-10"

PORT=54321
#PORT=58080

bsub -Is -P rh74 -m $COMPUTE_NODE.ebi.ac.uk -q production-rh74 ssh -R $PORT:localhost:$PORT noah-login-01
