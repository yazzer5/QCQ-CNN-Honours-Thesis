#!/bin/bash
# Job names. Smoke: v158s_*, grid: v158g_*. The gate finds logs by these prefixes.
smoke_name() { echo "v158s_k${1}${2}_s${3}_${4}"; }            # K TAG SIGMA MODEL
grid_name()  { echo "v158g_k${1}${2}_s${3}_${4}_r${5}"; }      # K TAG SIGMA MODEL SEED
queued_names() { qselect -u "$USER" 2>/dev/null | xargs -r qstat -f 2>/dev/null | awk '/Job_Name/{print $3}'; }
# Free gdata inodes for project wa66 (iQuota - iUsage), from lquota. Empty if lquota is unavailable.
free_inodes() { lquota 2>/dev/null | awk '$1=="wa66" && $2=="gdata"{print $(NF-1) - $(NF-2)}'; }
inode_guard() {   # $1 = number of jobs about to be submitted
  local need=$(( $1 * FILES_PER_JOB + INODE_MARGIN )) free
  free=$(free_inodes)
  if [ -z "$free" ]; then
    if [ "${FORCE_INODES:-0}" = "1" ]; then echo "WARN: lquota unreadable; FORCE_INODES=1 set, continuing"; return 0; fi
    echo "REFUSED: could not read free gdata inodes from lquota (set FORCE_INODES=1 to override after checking by hand)"; return 1
  fi
  if [ "$free" -lt "$need" ]; then
    echo "REFUSED: gdata has $free free inodes; $1 jobs need about $need (incl. margin $INODE_MARGIN). Free some first."; return 1
  fi
  echo "inodes: $free free on gdata, about $need needed for $1 jobs -- OK"
}
