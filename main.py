import sys
import copy
from io_lammps import xsf_info, el_info
from io_lammps import qe_out_info, make_qe_in
from io_lammps import init_log, upd_log
from io_lammps import init_axsf, upd_axsf
from io_lammps import make_lammps_in,get_energy_lammps,get_forces_lammps,get_coordinates_lammps
from mc import mc
from bv import bv
import os
import numpy as np
import subprocess
import ase
from ase import Atoms
#from deepmd.calculator import DP
from ase.constraints import FixAtoms
from ase.optimize import FIRE, LBFGS,BFGSLineSearch,BFGS
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
#import tensorflow as tf
#import torch
import time
import py4vasp
# import boo
####################
# READ INPUT FILES #
####################
# xsf_filename   = sys.argv[1] # read xsf filename
# el_filename    = sys.argv[2] # read element list filename

#############################
# SET SIMULATION PARAMETERS #
#############################
niter    =100
max_disp = 0.1                                # angstroms
T_move   = 1800                                # kelvin
ry_ev    = 13.605693009
bohr_ang = 0.52917721067
buf_len  = 0.0                                # length above surface within which atoms can be added

mu_list  = [-4.763901898,-9.292889113,-8.949121225]

act_p    = np.array([1, 1, 0, 1, 1, 0, 1]) # probablity of taking different actions
# [0]: move, [1]: swap, [2]: jump, [3]: add, [4]: remove [5]: grain boundary motion [6]: replace
fail_en  = 999.
nproc    = 144
nkdiv    = 1
ndiag    = 144
seed = 81

###########################
# GET ELEMENT INFORMATION #
###########################
el_filename="el_list.txt"
el = el_info() 
el.pop_attr(el_filename,T_move)

##############################
# GET STRUCTURAL INFORMATION #
##############################
xsf_filename="grain_boundary_modified.xsf"
xsf = xsf_info()
xsf.pop_attr(xsf_filename, el, buf_len)
xsf.c_max=xsf.lat_vec[2][2]/2.0e0+1.25e0
xsf.c_min=xsf.lat_vec[2][2]/2.0e0-1.25e0
xsf.max_grain_width=3.0e0
xsf.max_grain_width=10.0e0
####################################
# GET NEAREST NEIGHBOR INFORMATION #
####################################
bvo = bv()
bvo.init(xsf, el)

###################################
# INSTANTIATE MONTE CARLO ROUTINE #
###################################
mc_run = mc()
mc_run.init(T_move, max_disp, xsf)

###################################
# RUN GRAND CANONICAL MONTE CARLO #
###################################
# name="Ag_"+str(mu_Ag)+"_temp_"+str(seed)
name="temp_"+str(seed)
os.system('mkdir -p '+name)                                              # make temp directory for qe calculations
os.chdir(name)   
                                                     # enter temp
f = open("condition.txt", "w")
f.write("initialxsf," +str(xsf_filename)+'\n')
f.write("T_move," +str(T_move)+'\n')
f.write("move,"   +str(act_p[0])+'\n')
f.write("swap,"   +str(act_p[1])+'\n')
f.write("jump,"   +str(act_p[2])+'\n')
f.write("add, "   +str(act_p[3])+'\n')
f.write("remove," +str(act_p[4])+'\n')
f.write("gmove,"  +str(act_p[5])+'\n')
f.write("replace,"+str(act_p[6])+'\n')
for i in np.arange(0,len(el.sym)):
    f.write("mu_"+str(el.sym[i])+","+str(mu_list[i])+'\n')
f.close()

log_file              = init_log('log.dat',el)                             # initialize log file
#axsf_opt_file         = init_axsf('coord_opt.axsf', niter, xsf)         # "        " axsf file recording optimized structure
#axsf_new_file         = init_axsf('coord_new.axsf', niter, xsf)         # "                            " structure created in current iteration
axsf_accept_file      = init_axsf('coord_accept.axsf', niter, xsf)      # "                                      " accepted in current iteration
#axsf_failed_file      = init_axsf('coord_failed.axsf', niter, xsf)      # initialize axsf file recording structure failed in qe
#axsf_failed_iter_file = init_axsf('coord_failed_iter.axsf', niter, xsf) # initialize axsf file recording structure failed in qe
failed_cnt = 0

os.system("cp ../templates/INCAR   ./INCAR")
os.system("cp ../templates/POTCAR  ./POTCAR")
os.system("cp ../templates/KPOINTS ./KPOINTS")
os.system("cp ../templates/OSZICAR  ./OSZICAR")
os.system("cp ../templates/monitor.sh ./monitor.sh")
os.system("chmod +x monitor.sh")
os.system("watch -n 30 ./monitor.sh &>/dev/null &")

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def make_poscar_in(xsf, el):
 with open("./POSCAR", 'w') as f_new :
  f_new.write("Al2O3"+ '\n')
  f_new.write(" 1.0000000000000000"+ '\n')
  f_new.write(str(xsf.lat_vec[0][0])+' 0.0000000000000000  0.0000000000000000 \n')
  f_new.write("0.0000000000000000  "+str(xsf.lat_vec[1][1])+' 0.0000000000000000 \n')
  f_new.write("0.0000000000000000 0.0000000000000000  "+str(xsf.lat_vec[2][2])+' \n')
  if(xsf.el_each_num[2]>0):
    f_new.write(str(el.sym[0])+" "+str(el.sym[1])+" "+str(el.sym[2])+' \n')
    f_new.write(str(xsf.el_each_num[0])+" "+str(xsf.el_each_num[1])+" "+str(xsf.el_each_num[2])+' \n')
  elif(xsf.el_each_num[2]==0):
    f_new.write(str(el.sym[0])+" "+str(el.sym[1])+' \n')
    f_new.write(str(xsf.el_each_num[0])+" "+str(xsf.el_each_num[1])+' \n')
  f_new.write('Selective dynamics\n')
  f_new.write('Cartesian\n')
  row_poscar=0
  xsf.poscar= np.zeros(xsf.at_num).astype(np.int64) 
  for row in range(0,xsf.at_num):
    if el.sym[xsf.at_type[row]]=="Al":
      if row in xsf.at_grain1 or row in xsf.at_grain2:
        fix_str = 'F F F'
      else:
        fix_str = 'T T T'
      f_new.write(str(xsf.at_coord[row, 0]) + ' ' +
				  str(xsf.at_coord[row, 1]) + ' ' +
				  str(xsf.at_coord[row, 2]) + ' ' + fix_str+ '\n')
      xsf.poscar[row]=row_poscar
      row_poscar=row_poscar+1
  for row in range(0,xsf.at_num):
    if el.sym[xsf.at_type[row]]=="O":
      if row in xsf.at_grain1 or row in xsf.at_grain2:
        fix_str = 'F F F'
      else:
        fix_str = 'T T T'
      f_new.write(str(xsf.at_coord[row, 0]) + ' ' +
				  str(xsf.at_coord[row, 1]) + ' ' +
				  str(xsf.at_coord[row, 2]) + ' ' + fix_str+ '\n')
      xsf.poscar[row]=row_poscar
      row_poscar=row_poscar+1
  for row in range(0,xsf.at_num):
    if el.sym[xsf.at_type[row]]=="Ti":
      if row in xsf.at_grain1 or row in xsf.at_grain2:
        fix_str = 'F F F'
      else:
        fix_str = 'T T T'
      f_new.write(str(xsf.at_coord[row, 0]) + ' ' +
				  str(xsf.at_coord[row, 1]) + ' ' +
				  str(xsf.at_coord[row, 2]) + ' ' + fix_str+ '\n')
      xsf.poscar[row]=row_poscar
      row_poscar=row_poscar+1

def read_convergence(filename,lines=None):
    with open(filename) as fd:
        lines = fd.readlines()
    converged = True
    # First check electronic convergence
    for line in lines:
       if "RMM:  60" in line:
         converged=False
    return converged

whole_start = time.time()
for i in range(niter) :
    xsf.get_r_min_max(buf_len)
    xsf.get_vol()
    # attempt uvt action and store xsf attributes in xsf_new
    if i == 0 : 
        # alway start with move
        xsf = mc_run.uvt_new_structure(xsf, el, np.array([1,0,0,0,0,0,0]), bvo) 
    else :
        # xsf = mc_run.uvt_new_structure_np(xsf, el, act_p, bvo) 
        xsf = mc_run.uvt_new_structure(xsf, el, act_p, bvo) 

    print(i,mc_run.uvt_act)
    # make input file
    make_poscar_in(xsf, el)
    start = time.time()
    os.system("rm STOPCAR")
    os.system("srun -n 256 vasp_std")
    end = time.time()
    time_diff = end - start

    converged=read_convergence("OSZICAR")
    if converged==True:
      new_en= py4vasp.calculation.energy.read()['energy(sigma->0)']
    else:
      new_en=fail_en

    # if not move step and qe does not fail at first step, update atomic coordinates
    # if (mc_run.uvt_act != 0 ) :
    mc_run.new_xsf.at_coord = py4vasp.calculation.structure.cartesian_positions()[xsf.poscar]
    # update T
    mc_run.update_T_const(i - failed_cnt, 3000)

    # decide whether or not to accept uvt action 
    # note that old_xsf is changed to new_xsf if accepted
    accept = mc_run.uvt_mc(new_en, el, mu_list)

    # calculate free energy 
    free_en, _ = mc_run.get_free_g_p(new_en, el, mu_list)

    # if step not accepted, copy attributes from old (previous) xsf to xsf
    if accept == 0 or new_en == fail_en :
        xsf = mc_run.old_xsf.copy()
    # othewise copy new xsf to xsf
    else :
        xsf = mc_run.new_xsf.copy()

    # update logs if no qe error
    # if(new_en != fail_en):
        # write energies, number of accepted steps, and acceptance rate to log file
    upd_log(log_file, i, new_en,free_en, mc_run,xsf,el,time_diff)
        # write atomic coordinates to axsf file
#        upd_axsf(axsf_opt_file, i - failed_cnt, mc_run.opt_xsf, el)
#        upd_axsf(axsf_new_file, i - failed_cnt, mc_run.new_xsf, el)
    upd_axsf(axsf_accept_file, i, mc_run.old_xsf, el)
    # else :
        # failed_cnt += 1
#        upd_axsf(axsf_failed_file, failed_cnt, mc_run.new_xsf, el)
#        upd_axsf(axsf_failed_iter_file, i, mc_run.new_xsf, el)

log_file.close()
#axsf_opt_file.close()
#axsf_new_file.close()
axsf_accept_file.close()
#axsf_failed_file.close()
#axsf_failed_iter_file.close()
whole_end = time.time()
whole_time_diff = whole_end - whole_start
time_file = open("Total_time.txt", 'w')
time_file.write(str(whole_time_diff))
time_file.close()
os.chdir('../')

