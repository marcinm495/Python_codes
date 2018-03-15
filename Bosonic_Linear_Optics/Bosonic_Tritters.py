# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:25:20 2018

@author: Lenovo
"""
#sympy library for symbolic operations is used:
from sympy import MatrixSymbol, Matrix, Identity, Sum, Symbol
from cmath import exp, sqrt, pi, e
from sympy.physics.secondquant import KroneckerDelta

class Bosonic_Tritters:
    'This class allows to calculate distribution of bosons in no-touching scheme for n 3-level subsystems'
    
    #class constructor takes:
    # n_systems - the number of 3-level systems, type: arbitrary integer,
    # inp_boson_list - the list which gives an initial configuration of bosons, type: list of 3n integers,
    # inp_op_list - the list of operations on input subsystems, which can be either symmetric tritter "T" or Identity "Id", type: n-tuple of strings "T" or "Id"
    # out_op_list - the list of operations on output subsystems, which can be either symmetric tritter "T" or Identity "Id", type: n-tuple of strings "T" or "Id"
    # perm_list - list describing permutation of modes inbetween input and output operations, type: list of 3n integers being permutation of numbers 1...3n
   
    def __init__(self, n_systems, inp_boson_list, inp_op_list, out_op_list, perm_list):
        self.n_systems = n_systems
        self.inp_boson_list = inp_boson_list
        self.inp_op_list = inp_op_list
        self.out_op_list = out_op_list
        self.perm_list = perm_list
    
    
    #this method defined the action of a single-mode symmetric tritter
    #list_modes is a list of 3 numbers defining the position of a tritter in the system of modes
    #inp_op_mode is a number in list_modes which defines the location of input creation operator
    #op_name is a string which defines the name of the output creation operators
    
    def singModeTritter(self, list_modes, inp_op_mode, op_name):
        #matrix of creation operators:
        op_name = MatrixSymbol(op_name, 3*self.n_systems, 1)
        #complex 3-rd root of unity, as a symbol:
        #w = exp(2*pi*1j/3)
        w = Symbol('w')
        try:
            state1 = 0;
            for k in list_modes:
                state1 = state1 + (1/sqrt(3))*(w**KroneckerDelta(inp_op_mode, k))*op_name[k, 0]
            return state1
        except (RuntimeError, TypeError, NameError, IndexError):
            print("Improper constructing parameters - check compatibility of object's dimensions and types")  
    
    #this method applies a series of tritters or identities specified by the list op_list
    #state is the initial state 
    #op_list is the list of the form("T", "Id", "T",...) which specifies on which subsequent modes we apply tritter and on which identity
    #op_name_state is the name of the boson operators in state
    #op_name_new is the name of the output boson operators
    def applyTritters(self, state, op_list, op_name_state, op_name_new):
        #matrices of creation operators:
        keep_op_name_new = op_name_new
        op_name_state = MatrixSymbol(op_name_state, 3*self.n_systems, 1)
        op_name_new = MatrixSymbol(op_name_new, 3*self.n_systems, 1)
        #loop number:
        ln = 0;
        newState = state
        for k in op_list:
            if k == "T": 
                for m in range(ln, ln+3):
                    newState = newState.subs(op_name_state[m, 0], Bosonic_Tritters.singModeTritter(self, [ln, ln+1, ln+2], m, keep_op_name_new))
            else: #in the case of identity operator the modes still must be renamed:
                for m in range(ln, ln+3):
                    newState = newState.subs(op_name_state[m, 0], op_name_new[m, 0])
            ln = ln + 3
        return newState
    
    #applies permutation of modes specified by the permutation list self.perm_list specified in class constructor
    #state is the initial state
    #op_name_state is the name of the boson operators in state
    #op_name_new is the name of the output boson operators
    def applyPerms(self, state, op_name_state, op_name_new):
        #matrices of creation operators:
        keep_op_name_new = op_name_new
        op_name_state = MatrixSymbol(op_name_state, 3*self.n_systems, 1)
        op_name_new = MatrixSymbol(op_name_new, 3*self.n_systems, 1)
        newState = state
        for k in range(0, 3*self.n_systems):
            newState = newState.subs(op_name_state[k, 0], op_name_new[self.perm_list[k], 0])
        return newState
    
    
    #this method calculates the output state of the entire circuit in terms of creation bosonic operators:
    def calcOutOp(self):
        #initial matrix of creation operators:
        A = MatrixSymbol('A', 3*self.n_systems, 1)
        try:
            #state1 is a product of initial creation operators according to input list:
            state1 = 1;
            for k in range(0, 3*self.n_systems):
                state1=state1*A[k,0]**(self.inp_boson_list[k])    
            #calculating the state after the first round of local tritters or identities:
            state2 = Bosonic_Tritters.applyTritters(self, state1, self.inp_op_list, "A", "B")
            #calculating the state after permutations:
            state3 = Bosonic_Tritters.applyPerms(self, state2, "B", "C")
            #calculating the state after the second round of local tritters or identities:
            stateFin = Bosonic_Tritters.applyTritters(self, state3, self.out_op_list, "C", "A")
            return stateFin.expand()
            
        except (RuntimeError, TypeError, NameError, IndexError):
            print("Improper constructing parameters - check compatibility of object's dimensions and types")
    
    #this method calculates the output state and deletes modes specified in the list_del_modes list:
    def delSingModes(self, list_del_modes):
        A = MatrixSymbol('A', 3*self.n_systems, 1)
        state_out = Bosonic_Tritters.calcOutOp(self)
        for k in list_del_modes:
            state_out = state_out.subs(A[k, 0], 0)
        return state_out
    
    #this method calculates the output state and postselects on a single-particle events within output subsystems:
    #WARNING: this works only for n = 3 and 3 particles in the system!!
    def postselectSingleSystems(self):
        A = MatrixSymbol('A', 3*self.n_systems, 1)
        state_out = Bosonic_Tritters.calcOutOp(self)
        for k in range(0, self.n_systems):
            i1 = 3*k
            i2 = 3*k+1
            i3 = 3*k+2
            state_out = state_out.subs(A[i1,0]*A[i2,0]*A[i3,0],0).subs(A[i1,0]*A[i2,0],0).subs(A[i1,0]*A[i3,0],0).subs(A[i2,0]*A[i3,0],0)
            state_out = state_out.subs(A[i1,0]**3,0).subs(A[i2,0]**3,0).subs(A[i3,0]**3,0)
            state_out = state_out.subs(A[i1,0]**2,0).subs(A[i2,0]**2,0).subs(A[i3,0]**2,0)
        return state_out
    
    #this method calculates the output state, postselects on a single-particle events within output subsystems, and deletes modes specified in the list:
    def singSystDelModes(self, list_del_modes):
        A = MatrixSymbol('A', 3*self.n_systems, 1)
        state_out = Bosonic_Tritters.postselectSingleSystems(self)
        for k in list_del_modes:
            state_out = state_out.subs(A[k, 0], 0)
        return state_out
