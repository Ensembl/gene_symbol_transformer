#!/bin/bash


COMPUTE_NODE=""

PORT=54321

bsub -Is -P rh74 -m ${COMPUTE_NODE}.ebi.ac.uk -q production-rh74 ssh -R $PORT:localhost:$PORT noah-login-01
