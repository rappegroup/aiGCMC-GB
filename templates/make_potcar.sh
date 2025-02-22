#!/bin/sh
DIR=/global/common/software/nersc/pm-2022q2/sw/vasp/pseudopotentials
cat ${DIR}/PBE/potpaw_PBE/Al/POTCAR \
    ${DIR}/PBE/potpaw_PBE/O/POTCAR\
    ${DIR}/PBE/potpaw_PBE/Ti_sv/POTCAR\
    > POTCAR