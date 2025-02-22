import numpy as np
from ase.lattice.triclinic import TriclinicFactory
import ase
import ase.io
import os
###################################
### Step 1: Make a grain by ase ###
###################################
class Al2O3Factory(TriclinicFactory):
    bravais_basis = [[0.3333333333333333,    0.6666666666666666,    0.8145706666666667],
                     [0.6666666666666666,    0.3333333333333333,    0.6854293333333333],
                     [0.0000000000000000,    0.0000000000000000,    0.6479040000000000],
                     [0.3333333333333333,    0.6666666666666667,    0.5187626666666667],
                     [0.0000000000000000,    0.0000000000000000,    0.1479040000000000],
                     [0.3333333333333333,    0.6666666666666666,    0.0187626666666667],
                     [0.6666666666666666,    0.3333333333333333,    0.9812373333333333],
                     [0.0000000000000000,    0.0000000000000000,    0.8520960000000000],
                     [0.6666666666666666,    0.3333333333333333,    0.4812373333333333],
                     [0.0000000000000000,    0.0000000000000000,    0.3520960000000000],
                     [0.3333333333333333,    0.6666666666666666,    0.3145706666666666],
                     [0.6666666666666666,    0.3333333333333335,    0.1854293333333334],
                     [0.3605211666666668,    0.3333333333333333,    0.5833333333333334],
                     [0.6938545000000000,    0.6938545000000000,    0.7500000000000000],
                     [0.9728121666666665,    0.6394788333333332,    0.5833333333333334],
                     [0.6666666666666667,    0.0271878333333335,    0.5833333333333334],
                     [0.0000000000000000,    0.3061454999999998,    0.7500000000000000],
                     [0.3061454999999998,    0.0000000000000000,    0.7500000000000000],
                     [0.0271878333333335,    0.6666666666666666,    0.9166666666666667],
                     [0.3605211666666666,    0.0271878333333333,    0.0833333333333333],
                     [0.6394788333333330,    0.9728121666666665,    0.9166666666666667],
                     [0.3333333333333335,    0.3605211666666668,    0.9166666666666667],
                     [0.6666666666666666,    0.6394788333333332,    0.0833333333333333],
                     [0.9728121666666665,    0.3333333333333333,    0.0833333333333333],
                     [0.6938545000000000,    0.0000000000000000,    0.2500000000000000],
                     [0.0271878333333333,    0.3605211666666666,    0.4166666666666665],
                     [0.3061454999999997,    0.3061455000000000,    0.2500000000000000],
                     [0.0000000000000000,    0.6938545000000000,    0.2500000000000000],
                     [0.3333333333333333,    0.9728121666666665,    0.4166666666666665],
                     [0.6394788333333332,    0.6666666666666666,    0.4166666666666665]]
    element_basis = (0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)

Al2O3 =  Al2O3Factory()
lat_a=4.81e0
lat_b=4.81e0
lat_c=13.12e0

# ortho_a=lat_a
# ortho_b=lat_b/(np.tan(np.radians(120-90)))
# ortho_c=lat_c

ortho_a=lat_a/(np.tan(np.radians(120-90)))
ortho_b=lat_b
ortho_c=lat_c/(np.tan(np.radians(120-90)))

atoms=Al2O3(symbol=("Al","O"), latticeconstant={'a':lat_a ,'b':lat_b ,'c':lat_c,'alpha':90,'beta':90,'gamma':120},size=(25,25,10))
# atoms.rotate(a=90,v=(1,0,0))
atoms.rotate(a=90,v=(0,0,1))
ase.io.write("Triclinic_grain.xyz",images=atoms,format="xyz")

############################################################
### Step 2: Cut grain with ortho cell from triclinic xyz ###
############################################################
### read data ###
with open("Triclinic_grain.xyz", 'r') as f :
    at_num=int(f.readline())
    f.readline()
    at_eleme = np.empty((at_num),dtype="<U10")
    at_coord = np.zeros((at_num, 3))
    lines=f.readlines()
    for at in range(at_num) :
        line=lines[at].split()
        at_eleme[at]=line[0]	
        at_coord[at]=line[1:4]	

### Rotation matrix around y ###
x_ori=-50.1e0 
y_ori=50.1
z_ori=50.1e0

# x_ori=0.0e0 
# y_ori=0.0e0
# z_ori=0.0e0

at_elemeR=at_eleme
at_coordR=at_coord

### Cutting ###
cut_lx=ortho_a*15.0e0
cut_ly=ortho_b*15.0e0
cut_lz=ortho_c*15.0e0

at_numR=0
at_eleme = np.empty((at_num),dtype="<U10")
at_coord = np.zeros((at_num, 3))
for at in range(at_num):
    if(at_coordR[at][0] >= x_ori and at_coordR[at][0]<x_ori+cut_lx and 
       at_coordR[at][1] >= y_ori and at_coordR[at][1]<y_ori+cut_ly and 
       at_coordR[at][2] >= z_ori and at_coordR[at][2]<z_ori+cut_lz):
        at_eleme[at_numR]=at_elemeR[at]
        at_coord[at_numR]=at_coordR[at]
        at_numR=at_numR+1

at_eleme=at_eleme[0:at_numR]
at_coord=at_coord[0:at_numR]

### write a data ###
with open("./Ortho_grain.xyz", 'w') as f_new :
    f_new.write(str(at_numR)+ '\n')
    f_new.write('{0:.5f} {1:.5f} {2:.5f} \n'.format(cut_lx,cut_ly,cut_lz))
    for at in range(0,at_numR):
        f_new.write(str(at_eleme[at]) +' ' +
                    str(at_coord[at, 0]-x_ori) + ' ' +
                    str(at_coord[at, 1]-y_ori) + ' ' +
                    str(at_coord[at, 2]-z_ori) +  '\n')

# cuts=ase.io.read("./Ortho_grain.xyz")
# cuts.set_cell([cut_lx,cut_ly,cut_lz])
# cuts.pbc=[1, 1, 1]
# ase.io.write("lammps_structure.in",images=cuts,format="lammps-data")
# ase.io.write("lammps_structure.xsf",images=cuts,format="xsf")


###################################
### Step 3: Cut a rotated grain ###
###################################
### read data ###
with open("Ortho_grain.xyz", 'r') as f :
    at_num=int(f.readline())
    f.readline()
    at_eleme = np.empty((at_num),dtype="<U10")
    at_coord = np.zeros((at_num, 3))
    lines=f.readlines()
    for at in range(at_num) :
        line=lines[at].split()
        at_eleme[at]=line[0]	
        at_coord[at]=line[1:4]	

### Rotation matrix around y ###
x_ori=-4.9e0 
y_ori=0.0
z_ori=15.55e0
cell_num_x=1.0
cell_num_y=1.0
cell_num_z=1.0e0*0.65e0
# theta = np.radians(103.6/2.0)
# theta = np.radians(67.38/2.0)
theta = np.radians(76.4/2.0)
c, s = np.cos(theta), np.sin(theta)		

R = np.array(((c, 0, s), 
              (0, 1, 0),
              (-s,0, c))) 

at_elemeR=at_eleme
at_coordR=np.dot(at_coord,R)

### Cutting ###
cut_lx=ortho_a/abs(c)*cell_num_x*2/3
cut_ly=ortho_b*cell_num_y
cut_lz=ortho_c*abs(s)*cell_num_z

at_numR=0
at_eleme = np.empty((at_num),dtype="<U10")
at_coord = np.zeros((at_num, 3))
for at in range(at_num):
    if(at_coordR[at][0] >= x_ori and at_coordR[at][0]<x_ori+cut_lx and 
       at_coordR[at][1] >= y_ori and at_coordR[at][1]<y_ori+cut_ly and 
       at_coordR[at][2] >= z_ori and at_coordR[at][2]<z_ori+cut_lz):
        at_eleme[at_numR]=at_elemeR[at]
        at_coord[at_numR]=at_coordR[at]
        at_numR=at_numR+1

at_eleme=at_eleme[0:at_numR]
at_coord=at_coord[0:at_numR]

### write a data ###
with open("./grain_cut.xyz", 'w') as f_new :
    f_new.write(str(at_numR)+ '\n')
    f_new.write('{0:.5f} {1:.5f} {2:.5f} \n'.format(cut_lx,cut_ly,cut_lz))
    for at in range(0,at_numR):
        f_new.write(str(at_eleme[at]) +' ' +
                    str(at_coord[at, 0]-x_ori) + ' ' +
                    str(at_coord[at, 1]-y_ori) + ' ' +
                    str(at_coord[at, 2]-z_ori) +  '\n')

cuts=ase.io.read("./grain_cut.xyz")
cuts.set_cell([cut_lx,cut_ly,cut_lz])
cuts.pbc=[1, 1, 1]
#ase.io.write("lammps_structure_cut.in",images=cuts,format="lammps-data")
# cuts=ase.io.read("./Ortho_bulk.xyz")
# cuts.set_cell([cut_lx,cut_ly,cut_lz])
# cuts.pbc=[1, 1, 1]
# ase.io.write("lammps_structure.in",images=cuts,format="lammps-data")


#####################################
### Step 3: Make a grain boundary ###
#####################################
### input ###
thickness=2.5e0
outer_thick=10.0e0
### read data ###
with open("grain_cut.xyz", 'r') as f :
    at_num=int(f.readline())
    cut_lx,cut_ly,cut_lz=np.array(f.readline().split(),dtype=np.float32)
    at_eleme = np.empty((at_num),dtype="<U10")
    at_coord = np.zeros((at_num, 3))
    lines=f.readlines()
    for at in range(at_num) :
        line=lines[at].split()
        at_eleme[at]=line[0]	
        at_coord[at]=line[1:4]	

### make mirror image ###
MirroMat = np.array(((1,  0, 0), 
                     (0,   1, 0),
                     (0,   0, -1))) 

# at_eleme_mirro=at_eleme
# at_coord_mirro=np.dot(at_coord,MirroMat)+np.array([0.0,0.0,2*cut_lz+thickness])
at_eleme_mirro=at_eleme
at_coord_mirro=np.dot(at_coord,MirroMat)+np.array([cut_lx*2.0/4.0,cut_ly*2.0/4.0,2*cut_lz+thickness])
for at in range(0,at_num):
   if at_coord_mirro[at,0]<0:
     at_coord_mirro[at,0]=at_coord_mirro[at,0]+cut_lx
   if at_coord_mirro[at,0]>cut_lx:
     at_coord_mirro[at,0]=at_coord_mirro[at,0]-cut_lx
   if at_coord_mirro[at,1]<0:
     at_coord_mirro[at,1]=at_coord_mirro[at,1]+cut_ly
   if at_coord_mirro[at,1]>cut_ly:
     at_coord_mirro[at,1]=at_coord_mirro[at,1]-cut_ly

### write a xyz data ###
# with open("./grain_boundary.xyz", 'w') as f_new :
#     f_new.write(str(at_num)+ '\n')
#     f_new.write('{0:.5f} {1:.5f} {2:.5f} \n'.format(cut_lx,cut_ly,2*cut_lz+thickness))
#     for at in range(0,at_num):
#       if(at_coord[at, 0]>=0 and at_coord[at, 0]<cut_lx):
#         f_new.write(str(at_eleme[at]) +' ' +
#                     str(at_coord[at, 0]) + ' ' +
#                     str(at_coord[at, 1]) + ' ' +
#                     str(at_coord[at, 2]) +  '\n')
#     for at in range(0,at_num):
#       if(at_coord_mirro[at, 0]>=0 and at_coord_mirro[at, 0]<cut_lx):
#         f_new.write(str(at_eleme_mirro[at]) +' ' +
#                     str(at_coord_mirro[at, 0]) + ' ' +
#                     str(at_coord_mirro[at, 1]) + ' ' +
#                     str(at_coord_mirro[at, 2]) +  '\n')

### write a xsf data ###
with open("./grain_boundary.xsf", 'w') as f_new :
    f_new.write('CRYSTAL\n')
    f_new.write('PRIMVEC\n')
    f_new.write('{0:.5f} 0.0000000000 0.0000000000 \n'.format(cut_lx))
    f_new.write('0.0000000000 {0:.5f} 0.0000000000 \n'.format(cut_ly))
    f_new.write('0.0000000000  0.0000000000 {0:.5f}\n'.format(2*cut_lz+thickness+2*outer_thick))
    f_new.write('PRIMCOORD\n')
    f_new.write(str(2*at_num)+' 1 \n')
    for at in range(0,at_num):
        if (at_coord[at, 2]+outer_thick>=(2*cut_lz+thickness+2*outer_thick)/2.0e0-3.0e0):
            mol_id=3
        elif (at_coord[at, 2]+outer_thick>=(2*cut_lz+thickness+2*outer_thick)/2.0e0-6.0e0):
            mol_id=3
        else:
            mol_id=1
        f_new.write(str(at_eleme[at]) +' ' +
                    str(at_coord[at, 0]) + ' ' +
                    str(at_coord[at, 1]) + ' ' +
                    str(at_coord[at, 2]+outer_thick) + ' #'+str(mol_id)+'\n')
    for at in range(0,at_num):
        if (at_coord_mirro[at, 2]+outer_thick<=(2*cut_lz+thickness+2*outer_thick)/2.0e0+3.0e0):
            mol_id=3
        elif (at_coord_mirro[at, 2]+outer_thick<=(2*cut_lz+thickness+2*outer_thick)/2.0e0+6.0e0):
            mol_id=3
        else:
            mol_id=2
        f_new.write(str(at_eleme_mirro[at]) +' ' +
                    str(at_coord_mirro[at, 0]) + ' ' +
                    str(at_coord_mirro[at, 1]) + ' ' +
                    str(at_coord_mirro[at, 2]+outer_thick) + ' #'+str(mol_id)+'\n')


f=open("./grain_boundary.xsf")
lines=f.readlines()
f.close()

zeros=[]
f=open("./grain_boundary_modified.xsf","w")
f.writelines(lines[0:7])
for i in range(7,len(lines)):
   num=lines[i].split()[4]
   if(num=="#0"):
       zeros.append(lines[i])
   else:
       f.writelines(lines[i])

for i in range(0,len(zeros)):
  f.writelines(zeros[i])

f.close()




os.system("rm Triclinic_grain.xyz") 
os.system("rm Ortho_grain.xyz") 
os.system("rm grain_cut.xyz")

#cuts=ase.io.read("./grain_boundary.xsf")
#ase.io.write("lammps_structure_GB.in",images=cuts,format="lammps-data")