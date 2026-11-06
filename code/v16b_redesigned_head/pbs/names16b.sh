#!/bin/bash
# Job names (<= 15 chars). The gate finds logs by these prefixes:
#   smoke  v16bs<K><T><N>          repro  v16br_M4
#   grid   v16bg<K><T><N>s<SEED>
tletter() { [ "$1" = "all" ] && echo X || echo "$1"; }
smoke_name() { echo "v16bs$1$(tletter $2)$3"; }          # K TAG N
grid_name()  { echo "v16bg$1$(tletter $2)$3s$4"; }       # K TAG N SEED
queued_names() { qselect -u "$USER" 2>/dev/null | xargs -r qstat -f 2>/dev/null | awk '/Job_Name/{print $3}'; }
