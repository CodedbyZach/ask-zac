#!/bin/bash
stdbuf -oL -eL python3 main.py \
  2> >(grep -vE 'ALSA|jack|JackShm|Cannot connect to server|request channel|XDG_SESSION_TYPE' >&2)
