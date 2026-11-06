#!/bin/bash
# Job names. Smoke: v15s_*, grid: v15g_*. The gate finds logs by these prefixes.
smoke_name() { echo "v15s_k${1}${2}_a${3}_${4}"; }            # K TAG ALPHA MODEL
grid_name()  { echo "v15g_k${1}${2}_a${3}_${4}_s${5}"; }      # K TAG ALPHA MODEL SEED
queued_names() { qselect -u "$USER" 2>/dev/null | xargs -r qstat -f 2>/dev/null | awk '/Job_Name/{print $3}'; }
