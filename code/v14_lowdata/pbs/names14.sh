#!/bin/bash
# Job names (<= 15 chars). Logs are found by these prefixes:
#   smoke  v14s_<K><T><N><M>        e.g. v14s_5A50M3, v14s_10X250M5
#   repro  v14r_<M>                 e.g. v14r_M3
#   grid   v14g<K><T><N><M>s<SEED>  e.g. v14g5B100M2s47, v14g10X250M5s51
tletter() { [ "$1" = "all" ] && echo X || echo "$1"; }
smoke_name() { echo "v14s_$1$(tletter $2)$3$4"; }          # K TAG N MODEL
grid_name()  { echo "v14g$1$(tletter $2)$3$4s$5"; }        # K TAG N MODEL SEED
repro_name() { echo "v14r_$1"; }                           # MODEL
