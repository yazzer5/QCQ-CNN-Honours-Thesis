#!/bin/bash
# Job names (<= 15 chars). The gate finds logs by these prefixes:
#   smoke  v16sc<K><T><N>          v16sz<K><T><N><M>         repro  v16r_M2
#   grid   v16gc<K><T><N>s<SEED>   v16gz<K><T><N><M>s<SEED>
tletter() { [ "$1" = "all" ] && echo X || echo "$1"; }
smoke_name() { if [ "$1" = c ]; then echo "v16sc$2$(tletter $3)$4"; else echo "v16sz$2$(tletter $3)$4$5"; fi; }   # ARM K TAG N MODEL
grid_name()  { if [ "$1" = c ]; then echo "v16gc$2$(tletter $3)$4s$6"; else echo "v16gz$2$(tletter $3)$4$5s$6"; fi; }  # ARM K TAG N MODEL SEED
queued_names() { qselect -u "$USER" 2>/dev/null | xargs -r qstat -f 2>/dev/null | awk '/Job_Name/{print $3}'; }
