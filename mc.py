"""this module defines monte carlo objects and operations """
##### : Need to implement/correct immediately
####  : Need to implement/correct when switching to another function
###   : A very rare case may cause error
##    : Better implementation
import random
import copy
import numpy as np
from io_lammps import xsf_info
from io_lammps import el_info
from bv import bv

# set random seeds
#rand_seed = 52
#random.seed(a = rand_seed)
# np.random.seed(7)

# global variable definitions
ry_ev = 13.605693009
kb = 8.6173303e-5 # ev / k

class mc :
    """class for representing monte carlo operations"""
    def __init__(self) :
        self.T              = 0.0                        # System temperature 
        self.T_max          = 0.0                        # Max system temperature
        self.pace           = 0.0                        # Max displacement 
        self.pace_max       = 0.0                        # Max displacement limit 
        self.nvt_run_cnt    = 0                          # Canonical ensemble, running count
        self.uvt_run_cnt    = 0                          # Grand canonical ensemble, ensemble running count
        self.check_acc      = 0                          # Max displacement update rate
        self.nvt_acc        = 0                          # Canonical ensemble, number of acceptance
        self.at_mv_num      = 0                          # Number of atoms to move
        self.at_mv_ind      = np.array([]).astype('int') # Indices of atoms to move
        self.at_mv_vec      = np.zeros((0, 3))           # Displacement of atoms
        self.old_xsf        = xsf_info()                 # xsf in previous iteration
        self.new_xsf        = xsf_info()                 # xsf in this iteration
        self.opt_xsf        = xsf_info()                 # xsf associated with the lowest energy
        self.curr_en        = 0.0                        # Energy in current iteration
        self.curr_g         = 0.0                        # Free energy in current iteration
        self.opt_en         = 0.0                        # Lowest energy
        self.opt_g          = 0.0                        # Lowest free energy
        self.uvt_at_rm      = 0                          # Grand canonical ensemble, index of the atom to be removed
        self.uvt_at_ad      = 0                          # Grand canonical ensemble, index of the atom to be added
        self.uvt_act        = 0                          # Grand canonical ensemble, 0: move, 1: swap, 2: jump, 3: add, 4: remove
        self.uvt_at_exc_num = 0                          # Grand canonical ensemble, number of exchanged atoms, can only be 0, 1, -1
        self.uvt_el_exc     = 0                          # Grand canonical ensemble, index of the element to be exchanged
        self.uvt_el_exc2    = 0                          # Grand canonical ensemble, index of the element to be exchanged for replacement

    # enable explicit copy
    def copy(self) :
        cp_self = mc()
        cp_self.T              = self.T            
        cp_self.T_max          = self.T_max
        cp_self.pace           = self.pace          
        cp_self.pace_max       = self.pace_max      
        cp_self.nvt_run_cnt    = self.nvt_run_cnt   
        cp_self.uvt_run_cnt    = self.uvt_run_cnt   
        cp_self.check_acc      = self.check_acc     
        cp_self.nvt_acc        = self.nvt_acc       
        cp_self.at_mv_num      = self.at_mv_num
        cp_self.at_mv_ind      = np.array(self.at_mv_ind)     
        cp_self.at_mv_vec      = np.array(self.at_mv_vec)     
        cp_self.old_xsf        = self.old_xsf.copy()       
        cp_self.new_xsf        = self.new_xsf.copy()       
        cp_self.opt_xsf        = self.opt_xsf.copy()       
        cp_self.curr_en        = self.curr_en       
        cp_self.curr_g         = self.curr_g        
        cp_self.opt_en         = self.opt_en        
        cp_self.opt_g          = self.opt_g         
        cp_self.uvt_at_rm      = self.uvt_at_rm 
        cp_self.uvt_at_ad      = self.uvt_at_ad 
        cp_self.uvt_act        = self.uvt_act       
        cp_self.uvt_at_exc_num = self.uvt_at_exc_num   
        cp_self.uvt_el_exc     = self.uvt_el_exc
        cp_self.uvt_el_exc2     = self.uvt_el_exc2
        return cp_self
    
    def init(self, T, pace, xsf) :
        self.T           = T
        self.T_max       = T
        self.pace        = pace
        self.pace_max    = 0.3
        self.nvt_run_cnt = 1
        self.uvt_run_cnt = 1
        self.check_acc   = 25
        self.old_xsf     = xsf.copy()
        self.new_xsf     = xsf.copy()
        self.opt_xsf     = xsf.copy()
        self.curr_en     = 10000.0
        self.curr_g      = 10000.0
        self.opt_en      = 10000.0
        self.opt_g       = 10000.0

    # determine atoms to mv and step 
    # def rand_mv(self, xsf) :
    #     # number of atoms to move
    #     self.at_mv_num = np.random.randint(1, xsf.at_num + 1)
    #     if self.at_mv_num == xsf.at_num :
    #         self.at_mv_ind = np.array(range(self.at_mv_num))
    #     else :
    #          self.at_mv_ind = np.array(random.sample(range(xsf.at_num), self.at_mv_num))
    #     self.at_mv_vec = (np.random.rand(self.at_mv_num, 3) * 2 - 1) * self.pace

    def rand_mv(self, xsf) :
        # number of atoms to move
        if len(np.concatenate([xsf.at_rmb,xsf.at_buffer])) > 1 :
         self.at_mv_num = np.random.randint(1, len(np.concatenate([xsf.at_rmb,xsf.at_buffer])) + 1)
         if self.at_mv_num == len(np.concatenate([xsf.at_rmb,xsf.at_buffer])) :
            self.at_mv_ind = np.concatenate([xsf.at_rmb,xsf.at_buffer])
         else :
             self.at_mv_ind = np.array(random.sample(list(np.concatenate([xsf.at_rmb,xsf.at_buffer])), self.at_mv_num))
         self.at_mv_vec = (np.random.rand(self.at_mv_num, 3) * 2 - 1) * self.pace

    # update new coords
    def new_coords(self, xsf) :
        self.rand_mv(xsf)
        self.old_xsf = xsf.copy()
        self.new_xsf = xsf.copy()
        for i, ind in enumerate(self.at_mv_ind) :
            self.new_xsf.at_coord[ind, :] += self.at_mv_vec[i, :]
    
    # update pace
    def update_pace(self) :
        if float(self.nvt_acc) / self.check_acc < 0.2 :
            self.pace /= 2.0
        elif float(self.nvt_acc) / self.check_acc > 0.4 :
            self.pace *= 1.618

        if self.pace > self.pace_max :
            self.pace = self.pace_max
        self.nvt_acc = 0
    
    # uvt update structure
    def uvt_new_structure(self,xsf,el,act_p,bvo) : # el is of el_info class # act_p defines probability for different actions, [0]: move, [1]: swap, [2]: jump, [3]: add, [4]: remove
        self.old_xsf = xsf.copy()
        self.new_xsf = xsf.copy()
        act_pp = np.array(act_p)
        
        #------------------------------------------adjust act_p-----------------------------------------------------------------
        # avoid swapping action if only one element is removable(swappable)
        if act_pp[0] > 0 :
            if len(np.concatenate([xsf.at_rmb,xsf.at_buffer])) < 1 :
                act_pp[1] = 0

        # avoid swapping action if only one element is removable(swappable)
        if act_pp[1] > 0 :
            el_swap_num = 0
            for i in range(len(xsf.at_swap)) :
                if len(xsf.at_swap[i]) > 0 :
                    el_swap_num += 1
            if el_swap_num <= 1 :
                act_pp[1] = 0
        # avoid jumpping action if no appropriate site or no appropriate atom
        if act_pp[2] > 0 :
            # choose atom to jump
            at_neighbor_list = bvo.at_all_nn(xsf)
            at_neighbor_pref = np.zeros(xsf.at_num).astype('int')
            for i in range(xsf.at_num) :
                el_ind = xsf.at_type[i]
                at_neighbor_pref[i] = el.pref_nn[el_ind]
            weight = np.power((at_neighbor_list - at_neighbor_pref),4).astype('float')
            if np.sum(weight) != 0 :
                weight /= np.sum(weight)
                jump_at_ind = np.random.choice(range(xsf.at_num), 1, p=weight)[0]
                jump_el_ind = xsf.at_type[jump_at_ind]
                """choose site to jump to"""
                for i in np.arange(xsf.vol * 1000) : 
                    jump_vec = np.zeros(3)
                    jump_vec += np.random.rand() * xsf.lat_vec[0]
                    jump_vec += np.random.rand() * xsf.lat_vec[1]
                    jump_vec += (np.random.rand() * (xsf.c_max - xsf.c_min) + xsf.c_min) / np.linalg.norm(xsf.lat_vec[2]) * xsf.lat_vec[2]
                    jump_neighbor = bvo.at_single_nn(xsf, jump_at_ind, jump_vec)
                    if (jump_neighbor == el.pref_nn[jump_el_ind]) : 
                        break
                if i >= xsf.vol * 1000 - 1 :
                    act_pp[2] = 0
            else :
                act_pp[2] = 0
        # avoid adding if no elements are able to add to the system
        if act_pp[3] > 0 :
            if np.sum(el.p_add) == 0 :
                act_pp[3] = 0
        # also avoid adding if could not find a site based on coordination rule
        if act_pp[3] > 0 :
            at_ad = xsf.at_num # index of atom to be added
            self.uvt_el_exc  = np.random.choice(range(el.num), 1, p=el.p_add)[0]   # find the element index
            dis = 0
            trial = 0
            while dis < el.r_min[self.uvt_el_exc] or dis > el.r_max[self.uvt_el_exc] : # control atom distance
                at_ad_coord  = np.zeros(3)
                at_ad_coord += np.random.rand() * xsf.lat_vec[0] 
                at_ad_coord += np.random.rand() * xsf.lat_vec[1]
                at_ad_coord += (np.random.rand() * (xsf.c_max - xsf.c_min) + xsf.c_min) / np.linalg.norm(xsf.lat_vec[2]) * xsf.lat_vec[2]
                dis = min([np.linalg.norm(at_ad_coord - xsf.at_coord[ind]) for ind in range(xsf.at_num)])
                trial += 1
                if trial >= 100000 :
                    break
            if trial >= 100000 and (dis < el.r_min[self.uvt_el_exc] or dis > el.r_max[self.uvt_el_exc]) :
                act_pp[3] = 0

        # avoid removing action if no removable atoms
        if act_pp[4] > 0 :
            if len(xsf.at_rmb) == 0 : 
                act_pp[4] = 0
        
        # if act_pp[5] > 0 :
        #     width= np.min(xsf.at_coord[xsf.at_grain2][:,2])-np.max(xsf.at_coord[xsf.at_grain1][:,2])
        #     if width > xsf.max_grain_width or width < xsf.min_grain_width: 
        #         act_pp[5] = 0

        # avoid swapping action if only one element is removable(swappable)
        if act_pp[6] > 0 :
            if len(xsf.at_rmb) == 0 : 
                act_pp[6] = 0

        # normalize act_p, and make it accumalate probability
        act_pp = act_pp / float(np.sum(act_pp))
        for i in range(len(act_pp) - 1) :
            act_pp[i + 1] += act_pp[i]

        #----------------------------------------end adjust act_p--------------------------------------------------------------
        # generate a random number between 0 and 1
        cndt = np.random.rand()

        #-------------move atoms--------------
        if cndt < act_pp[0] :    
            self.uvt_act        = 0
            self.uvt_at_exc_num = 0
            self.uvt_el_exc     = 0
            self.new_coords(xsf)
            self.rand_mv(xsf)
            self.new_xsf.at_coord[xsf.at_grain1] =xsf.at_coord[xsf.at_grain1]+ np.array([np.random.rand()*0.1e0-0.05e0, np.random.rand()*0.1e0-0.05e0, np.random.rand()*0.1e0-0.05e0])
            self.new_xsf.at_coord[xsf.at_grain2] =xsf.at_coord[xsf.at_grain2]+ np.array([np.random.rand()*0.1e0-0.05e0, np.random.rand()*0.1e0-0.05e0, np.random.rand()*0.1e0-0.05e0])
            return self.new_xsf.copy()

        #-------------swap atoms-------------
        elif cndt < act_pp[1] : 
            self.uvt_act        = 1
            self.uvt_at_exc_num = 0
            # element to be swapped
            swap_el_1 = np.random.randint(el.num)
            while len(xsf.at_swap[swap_el_1]) == 0 :
                swap_el_1 = np.random.randint(el.num)
            swap_el_2 = np.random.randint(el.num)
            while swap_el_2 == swap_el_1 or len(xsf.at_swap[swap_el_2]) == 0 :
                swap_el_2 = np.random.randint(el.num)
            # atom to be swapped
            swap_at_1 = random.choice(xsf.at_swap[swap_el_1])
            swap_at_2 = random.choice(xsf.at_swap[swap_el_2])
            self.new_xsf.at_coord[swap_at_1, :] = np.array(self.old_xsf.at_coord[swap_at_2, :])
            self.new_xsf.at_coord[swap_at_2, :] = np.array(self.old_xsf.at_coord[swap_at_1, :])
            return self.new_xsf.copy()

        #-------------make atom jump, apply coord rule------------
        elif cndt < act_pp[2]:
            self.uvt_act        = 2
            self.uvt_at_exc_num = 0
            self.new_xsf.at_coord[jump_at_ind, :] = np.array(jump_vec)
            return self.new_xsf.copy()

        #--------------------add one atom--------------------------------
        ### Need to move site searching part into adjusting p part
        elif cndt < act_pp[3] : 
            self.uvt_act        = 3
            self.uvt_at_exc_num = 1
            self.new_xsf.at_coord = np.vstack((xsf.at_coord, at_ad_coord))   # add coordinates to xsf
            self.new_xsf.at_type  = np.append(xsf.at_type, self.uvt_el_exc)  # add atom to the atom list
            self.new_xsf.at_rmb   = np.append(xsf.at_rmb, at_ad)             # add atom to removable atom array
            self.new_xsf.at_swap[self.uvt_el_exc].append(at_ad)              # add atom to swappable atom list
            self.new_xsf.el_each_num[self.uvt_el_exc] += 1                   # increase the number of that element
            self.new_xsf.at_num += 1  
            return self.new_xsf.copy()

        #-----------------------remove one atom----------------------------
        elif cndt < act_pp[4] :        
            self.uvt_act        = 4
            self.uvt_at_exc_num = -1
            at_rm = random.choice(xsf.at_rmb)                           # index of atom to be removed
            self.uvt_el_exc = xsf.at_type[at_rm]                        # element index
            self.new_xsf.at_coord = np.delete(xsf.at_coord, at_rm, 0)   # remove the coordinates
            self.new_xsf.at_type = np.delete(xsf.at_type, at_rm, 0)     # remove the atom from atoms list
            self.new_xsf.el_each_num[self.uvt_el_exc] -= 1              # decrease the number of that element
            self.new_xsf.at_num -= 1                                    # decrease the total number of atoms
            self.new_xsf.at_rmb = np.append(xsf.at_rmb[xsf.at_rmb < at_rm], xsf.at_rmb[xsf.at_rmb > at_rm] - 1) # remove the atom from removable atoms
            self.new_xsf.at_swap[self.uvt_el_exc].remove(at_rm)         # remove the atom from swappable atoms
            for i in range(len(self.new_xsf.at_swap)) :                        # remove the atom from swappable atoms
                for j in range(len(self.new_xsf.at_swap[i])) :
                    if self.new_xsf.at_swap[i][j] > at_rm :
                        self.new_xsf.at_swap[i][j] -= 1
            return self.new_xsf.copy()

        elif cndt < act_pp[5]:
            self.uvt_act        = 5
            self.uvt_at_exc_num = 0
            self.new_xsf.at_coord = xsf.at_coord
            # self.new_xsf.at_coord[xsf.at_grain1] =xsf.at_coord[xsf.at_grain1]+ np.array([np.random.rand() * 0.05e0-0.025e0,np.random.rand() * 0.05e0-0.025e0,np.random.rand() * 0.05e0-0.025e0])
            # self.new_xsf.at_coord[xsf.at_grain2] =xsf.at_coord[xsf.at_grain2]+ np.array([np.random.rand() * 0.05e0-0.025e0,np.random.rand() * 0.05e0-0.025e0,np.random.rand() * 0.05e0-0.025e0])
            # self.new_xsf.at_coord[xsf.at_grain1] =xsf.at_coord[xsf.at_grain1]+ np.array([0.0e0,0.0e0,np.random.rand() * 0.01e0-0.005e0])
            # self.new_xsf.at_coord[xsf.at_grain2] =xsf.at_coord[xsf.at_grain2]+ np.array([0.0e0,0.0e0,np.random.rand() * 0.01e0-0.005e0])
            # self.new_xsf.at_coord[xsf.at_grain1] =xsf.at_coord[xsf.at_grain1]+ np.array([np.random.rand() * 0.01e0-0.005e0,0.0e0,np.random.rand() * 0.01e0-0.005e0])
            # self.new_xsf.at_coord[xsf.at_grain2] =xsf.at_coord[xsf.at_grain2]+ np.array([np.random.rand() * 0.01e0-0.005e0,0.0e0,np.random.rand() * 0.01e0-0.005e0])
            # self.new_xsf.at_coord[xsf.at_grain1] =xsf.at_coord[xsf.at_grain1]+ np.array([np.random.rand() * 0.2e0-0.1e0,0.0e0,np.random.rand() * 0.2e0-0.1e0])
            # self.new_xsf.at_coord[xsf.at_grain2] =xsf.at_coord[xsf.at_grain2]+ np.array([np.random.rand() * 0.2e0-0.1e0,0.0e0,np.random.rand() * 0.2e0-0.1e0])
            # self.new_xsf.at_coord[xsf.at_grain1] =xsf.at_coord[xsf.at_grain1]+ np.array([np.random.rand()*0.01e0-0.005e0, np.random.rand()*0.01e0-0.005e0, np.random.rand()*0.01e0-0.005e0])
            # self.new_xsf.at_coord[xsf.at_grain2] =xsf.at_coord[xsf.at_grain2]+ np.array([np.random.rand()*0.01e0-0.005e0, np.random.rand()*0.01e0-0.005e0, np.random.rand()*0.01e0-0.005e0])
            self.new_xsf.at_coord[xsf.at_grain1] =xsf.at_coord[xsf.at_grain1]+ np.array([np.random.rand()*0.1e0-0.05e0, np.random.rand()*0.1e0-0.05e0, np.random.rand()*0.1e0-0.05e0])
            self.new_xsf.at_coord[xsf.at_grain2] =xsf.at_coord[xsf.at_grain2]+ np.array([np.random.rand()*0.1e0-0.05e0, np.random.rand()*0.1e0-0.05e0, np.random.rand()*0.1e0-0.05e0])
            # self.new_xsf.at_coord[xsf.at_grain1] =xsf.at_coord[xsf.at_grain1]+ np.array([np.random.rand()*0.5e0-0.25e0, np.random.rand()*0.5e0-0.25e0, np.random.rand()*0.5e0-0.25e0])
            # self.new_xsf.at_coord[xsf.at_grain2] =xsf.at_coord[xsf.at_grain2]+ np.array([np.random.rand()*0.5e0-0.25e0, np.random.rand()*0.5e0-0.25e0, np.random.rand()*0.5e0-0.25e0])
            # self.new_xsf.at_coord[xsf.at_grain1] =xsf.at_coord[xsf.at_grain1]+ np.array([np.random.rand() * 0.1e0-0.05e0,0.0e0,np.random.rand() * 0.1e0-0.05e0])
            # self.new_xsf.at_coord[xsf.at_grain2] =xsf.at_coord[xsf.at_grain2]+ np.array([np.random.rand() * 0.1e0-0.05e0,0.0e0,np.random.rand() * 0.1e0-0.05e0])
            return self.new_xsf.copy()

        #-----------------------replace one atom----------------------------
        elif cndt < act_pp[6] :      
            self.uvt_act        = 6
            self.uvt_at_exc_num = 0
            at_rm = random.choice(xsf.at_rmb)                           # index of atom to be removed
            at_ad_coord=xsf.at_coord[at_rm]
            removed_ele=xsf.at_type[at_rm]  
            self.uvt_el_exc = xsf.at_type[at_rm]                        # element index
            self.new_xsf.at_coord = np.delete(xsf.at_coord, at_rm, 0)   # remove the coordinates
            self.new_xsf.at_type = np.delete(xsf.at_type, at_rm, 0)     # remove the atom from atoms list
            self.new_xsf.el_each_num[self.uvt_el_exc] -= 1              # decrease the number of that element
            self.new_xsf.at_num -= 1                                    # decrease the total number of atoms
            self.new_xsf.at_rmb = np.append(xsf.at_rmb[xsf.at_rmb < at_rm], xsf.at_rmb[xsf.at_rmb > at_rm] - 1) # remove the atom from removable atoms
            self.new_xsf.at_swap[self.uvt_el_exc].remove(at_rm)         # remove the atom from swappable atoms
            for i in range(len(self.new_xsf.at_swap)) :                 # remove the atom from swappable atoms
                for j in range(len(self.new_xsf.at_swap[i])) :
                    if self.new_xsf.at_swap[i][j] > at_rm :
                        self.new_xsf.at_swap[i][j] -= 1
            at_ad = self.new_xsf.at_num # index of atom to be added
            add_ele=list(range(el.num))
            add_ele.remove(removed_ele)
            self.uvt_el_exc2 = np.random.choice(add_ele, 1)[0] #, p=el.p_add  
            self.new_xsf.at_coord = np.vstack((self.new_xsf.at_coord, at_ad_coord))   # add coordinates to xsf
            self.new_xsf.at_type  = np.append(self.new_xsf.at_type, self.uvt_el_exc2)  # add atom to the atom list
            self.new_xsf.at_rmb   = np.append(self.new_xsf.at_rmb, at_ad)             # add atom to removable atom array
            self.new_xsf.at_swap[self.uvt_el_exc2].append(at_ad)              # add atom to swappable atom list
            self.new_xsf.el_each_num[self.uvt_el_exc2] += 1                   # increase the number of that element
            self.new_xsf.at_num += 1 
            return self.new_xsf.copy()

    # canonical acceptance condition, return 1 if accepted, 0 otherwise
    # en in unit of eV
    def nvt_mc(self, en) :
        self.nvt_run_cnt += 1
        if en < self.curr_en :
            self.curr_en = en
            self.nvt_acc += 1
            if en < self.opt_en :
                self.opt_en = en
                self.opt_xsf = self.new_xsf.copy()
            if self.nvt_run_cnt % self.check_acc == 0 :
                self.update_pace()
            return 1
        elif np.random.uniform() < np.exp(-(en - self.curr_en) / (self.T * kb)) :
            self.curr_en = en
            self.nvt_acc += 1
            if self.nvt_run_cnt % self.check_acc == 0 :
                self.update_pace()
            return 1
        else :
            if self.nvt_run_cnt % self.check_acc == 0 :
                self.update_pace()
            return 0

    # Calculate free energy and probability threshhold of accepting steps
    # en in unit of eV
    # def get_free_g_p(self, en, el, mu_list) : 
    #     free_g_new = en
    #     # calculate free energy
    #     for i in range(el.num) :
    #         free_g_new -= mu_list[i] * self.new_xsf.el_each_num[i]
    #     # update thermal de broglie wavelengths
    #     el.update_therm_db(self.T)
    #     # get thermal de broglie wavelengths of exchange element
    #     exc_therm_db = el.therm_db[self.uvt_el_exc]
    #     exc_el_num = self.new_xsf.el_each_num[self.uvt_el_exc] - (self.uvt_at_exc_num - 1) / 2
    #     if self.uvt_act <= 6 :
    #         exp_coef = np.exp(-(free_g_new - self.curr_g) / (self.T * kb))
    #     else : 
    #         print('Wrong action number for get_free_g_p! Undefined action!')
    #         exit()
    #     prob_acc = np.minimum(1, exp_coef * (self.new_xsf.vol / exc_therm_db**3 / exc_el_num )**self.uvt_at_exc_num)
    #     return free_g_new, prob_acc

    def get_free_g_p(self, en, el, mu_list) : 
        free_g_new = en
        # calculate free energy
        for i in range(el.num) :
            free_g_new -= mu_list[i] * self.new_xsf.el_each_num[i]
        # update thermal de broglie wavelengths
        el.update_therm_db(self.T)
        if self.uvt_act <= 6 :
            exp_coef = np.exp(-(free_g_new - self.curr_g) / (self.T * kb))
        else : 
            print('Wrong action number for get_free_g_p! Undefined action!')
            exit()
        if self.uvt_act <= 5: 
            # get thermal de broglie wavelengths of exchange element
            exc_therm_db = el.therm_db[self.uvt_el_exc]
            exc_el_num = self.new_xsf.el_each_num[self.uvt_el_exc] - (self.uvt_at_exc_num - 1) / 2    
            prob_acc = np.minimum(1.0, exp_coef * (self.new_xsf.vol / exc_therm_db**3 / exc_el_num )**self.uvt_at_exc_num)
        elif self.uvt_act == 6: ### replacement
            # get thermal de broglie wavelengths of exchange element
            exc_therm_db = el.therm_db[self.uvt_el_exc]
            exc_therm_db2= el.therm_db[self.uvt_el_exc2]
            exc_el_num = self.new_xsf.el_each_num[self.uvt_el_exc]+1  ### el_each_num is N-1. So we should add 1.
            exc_el_num2= self.new_xsf.el_each_num[self.uvt_el_exc2]   ### el_each_num is already N+1. So, we should not add anything.
            prob_acc = np.minimum(1.0, exp_coef * (exc_el_num*exc_therm_db**3) /(exc_el_num2*exc_therm_db2**3) )
        return free_g_new, prob_acc


    # Adjust T
    def update_T_const(self, iter, period) :
        self.T = self.T_max

    def update_T_linear(self, iter, period) : 
        self.T = self.T_max + float(1 - self.T_max) / period * (iter % period)

    def update_T_exp(self, iter, period) : 
        self.T = self.T-max * (1 / float(self.T_max))**( float(iter % period) / (period-1))

    def update_T_quadratic(self, iter, period) : 
        self.T = float(self.T_max - 1) / (period-1)**2 * (period - iter%period - 1)**2 + 1

    # Grand canonical acceptance condition, return 1 if accepted, 0 otherwise
    def uvt_mc(self, en, el, mu_list) :
        self.uvt_run_cnt += 1
        rand = np.random.rand()
        free_g, prob_acc = self.get_free_g_p(en, el, mu_list)
        if self.uvt_act == 0 : # move
            self.nvt_run_cnt += 1
            if rand <= prob_acc : # accept
                self.nvt_acc += 1
                self.curr_g = free_g
                self.old_xsf = self.new_xsf.copy()
                if free_g < self.opt_g : 
                    self.opt_g = free_g
                    self.opt_xsf = self.new_xsf.copy()

                if self.nvt_run_cnt % self.check_acc == 0 :
                    self.update_pace()
                return 1
            else :
                if self.nvt_run_cnt % self.check_acc == 0 :
                    self.update_pace()
                return 0
        elif self.uvt_act == 1 or self.uvt_act == 2 : # swap & jump
            if rand <= prob_acc : # accept
                self.curr_g = free_g
                self.old_xsf = self.new_xsf.copy()
                if free_g < self.opt_g : 
                    self.opt_g = free_g
                    self.opt_xsf = self.new_xsf.copy()
                return 1
            else :
                return 0
        elif self.uvt_act == 3 or self.uvt_act == 4 : # exchange
            if rand <= prob_acc : # accept
                self.curr_g = free_g
                self.old_xsf = self.new_xsf.copy()
                if free_g < self.opt_g : 
                    self.opt_g = free_g
                    self.opt_xsf = self.new_xsf.copy()
                return 1
            else :
                return 0
        elif self.uvt_act == 5 : # grain boundary motion
            if rand <= prob_acc : # accept
                self.curr_g = free_g
                self.old_xsf = self.new_xsf.copy()
                if free_g < self.opt_g : 
                    self.opt_g = free_g
                    self.opt_xsf = self.new_xsf.copy()
                return 1
            else :
                return 0
        elif self.uvt_act == 6 : # grain boundary motion
            if rand <= prob_acc : # accept
                self.curr_g = free_g
                self.old_xsf = self.new_xsf.copy()
                if free_g < self.opt_g : 
                    self.opt_g = free_g
                    self.opt_xsf = self.new_xsf.copy()
                return 1
            else :
                return 0
        else : 
            print('Wrong action number! Undefined action!')
            exit()

    # Some thing might need to do: instead of fixing the action probability, separate acc into acc_a, acc_r, acc_m, and adjust action probability based on these numbers, (basicly acc_a and acc_r), i.e. if acc_a is large, meaning that adding atom is favored, we should increase the action probability of add atoms
