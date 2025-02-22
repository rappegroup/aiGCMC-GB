#!/bin/bash

if [ $(cat OSZICAR | grep -c "RMM:  60") == 1 ]; 
then
  touch STOPCAR
  printf 'LSTOP = .TRUE.' > STOPCAR
fi
