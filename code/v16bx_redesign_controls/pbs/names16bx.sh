#!/bin/bash
# Job names (<= 15 chars). Gate prefixes: smoke v16xs<M>, repro v16xr_BFR, grid v16xg<M>s<SEED>  (M = T for M4trim, Z for BFRZZ)
mletter() { [ "$1" = "M4trim" ] && echo T || echo Z; }
smoke_name() { echo "v16xs$(mletter $1)"; }
grid_name()  { echo "v16xg$(mletter $1)s$2"; }
queued_names() { qselect -u "$USER" 2>/dev/null | xargs -r qstat -f 2>/dev/null | awk '/Job_Name/{print $3}'; }
