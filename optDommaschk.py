#!/usr/bin/env python3
###################################################################
######## DOMMASCHK Optimization with Matt's Residue Script ########
################ Rogerio Jorge, April 29, 2021 ####################
###################################################################
from simsopt import LeastSquaresProblem,least_squares_serial_solve
from simsopt.geo.magneticfieldclasses import Dommaschk
from pyoculus.solvers  import FixedPoint, PoincarePlot
from simsopt._core.optimizable import Optimizable
from periodic_field_line import periodic_field_line
from tangent_map import tangent_map
from pyoculus.problems import CartesianBfield
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
# ###########################
# BioSavart class to pyOculus
class SimsgeoBiotSavart(CartesianBfield):
    def __init__(self, bs, R0, Z0, Nfp=1):
        super().__init__(R0, Z0, Nfp)
        self._bs = bs
    def B(self, xyz, args=None):
        point = np.array([xyz])
        self._bs.set_points(point)
        Bfield=self._bs.B()
        return Bfield[0]
    def dBdX(self, xyz, args=None):
        point = np.array([xyz])
        self._bs.set_points(point)
        dB=self._bs.dB_by_dX()
        return dB[0]
###########################################
# Field class for the Matt Landreman's residue script
class Field():
    def __init__(self, Bfield, NFP):
        self.nfp = NFP
        self.Bfield=Bfield
    def BR_Bphi_BZ(self,R,phi,Z):
        self.Bfield.set_points([[R*np.cos(phi),R*np.sin(phi),Z]])
        Bxyz  = self.Bfield.B()
        B_phi = cos(phi)*Bxyz[0,1] - sin(phi)*Bxyz[0,0]
        B_r   = cos(phi)*Bxyz[0,0] + sin(phi)*Bxyz[0,1]
        B_z   = Bxyz[0, 2]
        return B_r, B_phi, B_z
    def grad_B(self,R,phi,Z):
        self.Bfield.set_points([[R*np.cos(phi),R*np.sin(phi),Z]])
        dB       = self.Bfield.dB_by_dX()[0]
        drBr     = cos(phi)*( cos(phi)*dB[0,0]+sin(phi)*dB[0,1])+sin(phi)*( cos(phi)*dB[1,0]+sin(phi)*dB[1,1])
        dphiBr   = cos(phi)*(-sin(phi)*dB[0,0]+cos(phi)*dB[0,1])+sin(phi)*(-sin(phi)*dB[1,0]+cos(phi)*dB[1,1])
        dzBr     = cos(phi)*dB[0,2]+sin(phi)*dB[1,2]
        drBphi   = cos(phi)*( cos(phi)*dB[1,0]+sin(phi)*dB[1,1])-sin(phi)*( cos(phi)*dB[0,0]+sin(phi)*dB[0,1])
        dphiBphi = cos(phi)*(-sin(phi)*dB[1,0]+cos(phi)*dB[1,1])-sin(phi)*(-sin(phi)*dB[0,0]+cos(phi)*dB[0,1])
        dzBphi   = cos(phi)*dB[1,2]-sin(phi)*dB[0,2]
        drBz     = cos(phi)*dB[2,0]+sin(phi)*dB[2,1]
        dphiBz   =-sin(phi)*dB[2,0]+cos(phi)*dB[2,1]
        dzBz     = dB[2,2]
        return np.array([[drBr,dphiBr,dzBr],[drBphi,dphiBphi,dzBphi],[drBz,dphiBz,dzBz]])
    def d_RZ_d_phi(self, phi, RZ):
        R = RZ[0]
        Z = RZ[1]
        BR, Bphi, BZ = self.BR_Bphi_BZ(R, phi, Z)
        return [R * BR / Bphi, R * BZ / Bphi]
######################################################
# Optimizable class specific to the CaryHanson problem
class objBfieldResidue(Optimizable):
    def __init__(self):
        self.mn      = [[5,2],[5,4],[5,10]]
        self.coeffs  = [[1.4,1.4],[19.25,0],[0,0]]
        self.NFP     = 5
        self.magnetic_axis_radius = 1.0
        self.Bfield  = Dommaschk(self.mn,self.coeffs)
        self.sbsp    = SimsgeoBiotSavart(self.Bfield, self.magnetic_axis_radius, Z0=0, Nfp=self.NFP)
        self._set_names()
    def _set_names(self):
        self.names = ['m(1)', 'n(1)', 'm(2)', 'n(2)', 'm(3)', 'n(3)', 'b(5,2)', 'c(5,2)', 'b(5,4)', 'c(5,4)', 'bc(5,10)']
    def get_dofs(self):
        return np.ndarray.flatten(np.concatenate((self.mn,self.coeffs)))[:-1]
    def set_dofs(self, dofs):
        self.mn     = [[dofs[0],dofs[1]],[dofs[2],dofs[3]],[dofs[4],dofs[5]]]
        self.coeffs = [[dofs[6],dofs[7]],[dofs[8],dofs[9]],[dofs[10],dofs[10]]]
        self.Bfield = Dommaschk(self.mn,self.coeffs)
        self.sbsp   = SimsgeoBiotSavart(self.Bfield, self.magnetic_axis_radius, Z0=0, Nfp=self.NFP)
    def residue(self, R0=1.16, periods=9, points=100):
        R0 = R0 - np.log(abs(self.get_dofs()[10])+1e0)**2/15400
        try:
            field = Field(self.Bfield,self.NFP)
            pfl   = periodic_field_line(field, points, R0=R0, periods=periods)
            R_k   = pfl.R_k
            Z_k   = pfl.Z_k
            if max(pfl.R_k)<1.01:
                return [2e12, [0], [0], periods, R0]
            else:
                tm    = tangent_map(field, pfl)
                return [tm.residue, R_k, Z_k, periods, R0]
        except:
            return [1e12, [0], [0], periods, R0]
    # def residue0(self): return self.residue(1.142,8 )[0]
    def residue1(self): return self.residue(1.189 ,8 )[0]
    def residue2(self): return self.residue(1.124 ,9 )[0]
    def residue3(self): return self.residue(1.075 ,11)[0]
    # def residue4(self): return self.residue(1.1186,10)[0]
    def poincare(self, Rbegin=1.0, Rend=1.165, nPpts=150, nPtrj=35):
        params = dict()
        params["Rbegin"]     = Rbegin
        params["Rend"]       = Rend
        params["nPpts"]      = nPpts
        params["nPtrj"]      = nPtrj
        self.p               = PoincarePlot(self.sbsp, params)
        self.poincare_output = self.p.compute()
        self.iota            = self.p.compute_iota()
        return self.p
# ############################################
if __name__ == "__main__":
    ## Start optimizable class
    obj             = objBfieldResidue()
    initialDofs     = obj.get_dofs()
    residueInputs   = [[1.1884,8],[1.124,9],[1.075,11],[1.136,10]]
    initialPoincare = 0
    # residueInputs = [[1.1884,8],[1.124,9],[1.075,11],[1.165,9],[1.1186,10],[1.136,10]]
    # obj.set_dofs([5,2,5,4,5,10,1.4,1.4,19.25,0,-2.2e-5])
    # a1=obj.residue(R0=1.1884,periods=8 ,points=100)
    # a2=obj.residue(R0=1.124,periods=9 ,points=100)
    # a2=obj.residue(R0=1.165,periods=9 ,points=100)
    # a3=obj.residue(R0=1.136,periods=10,points=100)
    # a4=obj.residue(R0=1.1186,periods=10,points=100)
    # a5=obj.residue(R0=1.075,periods=11,points=100)
    # print(a1)
    # print(a2)
    # print(a3)
    # print(a4)
    # print(a5)
    # exit()
    ## Create initial Poincare Plot
    solInitial = []
    [solInitial.append(obj.residue(resIn[0],resIn[1])) for resIn in residueInputs]
    if initialPoincare == 1:
        p = obj.poincare(nPpts=500,nPtrj=50,Rbegin=1.0,Rend=1.18); p.plot(s=1.5)
        [plt.scatter(fSol[1], fSol[2], s=35, marker="x", label=f"Periods = {fSol[3]:.0f}, Residue = {fSol[0]:.4f}") for fSol in solInitial]
        plt.xlim([0.8 , 1.2]); plt.ylim([-0.08, 0.08]); plt.tight_layout()
        plt.legend()
        plt.savefig('Results/DommaschkInitialPoincare.png', dpi=500)
        plt.savefig('Results/DommaschkInitialPoincare.pdf')
        p.plot_iota(); plt.ylim([0.4,0.6]); plt.tight_layout()
        plt.savefig('Results/DommaschkInitialIota.png')

    ## Optimization
    prob = LeastSquaresProblem([(obj.residue1,0,1),
                                (obj.residue2,0,1),
                                (obj.residue3,0,1)])
    # Set degrees of freedom for the optimization
    obj.all_fixed()
    obj.set_fixed('bc(5,10)', False)
    obj.set_dofs([5,2,5,4,5,10,1.4,1.4,19.25,0,-1e4])
    # Run optimization problem
    nIterations = 400
    print('Starting optimization...')
    least_squares_serial_solve(prob, xtol=1e-9, ftol=1e-9, gtol=1e-9, method='lm', max_nfev=nIterations)
    print('Optimization finished...')

    ## Create final Poincare Plot
    solFinal = []
    # residueInputs = [[1.1884,8],[1.124,9],[1.075,11],[1.150,8]]
    residueInputs = [[1.1884,8],[1.124,9],[1.075,11]]
    [solFinal.append(obj.residue(resIn[0],resIn[1])) for resIn in residueInputs]
    p = obj.poincare(nPpts=500,nPtrj=50,Rbegin=0.98,Rend=1.18); p.plot(s=1.5)
    [plt.scatter(fSol[1], fSol[2], s=35, marker="x", label=f"Periods = {fSol[3]:.0f}, Residue = {fSol[0]:.4f}") for fSol in solFinal]
    plt.xlim([0.8 , 1.2]); plt.ylim([-0.08, 0.08]); plt.tight_layout()
    plt.legend()
    plt.savefig('Results/DommaschkFinalPoincare.png', dpi=500)
    plt.savefig('Results/DommaschkFinalPoincare.pdf')
    p.plot_iota(); plt.ylim([0.4,0.6]); plt.tight_layout()
    plt.savefig('Results/DommaschkFinalIota.png')

    ## Print final results
    print('Initial degrees of freedom =',initialDofs)
    print('Final   degrees of freedom =',obj.get_dofs())
    print('Initial Residue1 = ',solInitial[0][0])
    print('Final   Residue1 = ',solFinal[0][0])
    print('Initial Residue2 = ',solInitial[1][0])
    print('Final   Residue2 = ',solFinal[1][0])
    print('Initial Residue3 = ',solInitial[2][0])
    print('Final   Residue3 = ',solFinal[2][0])

    # Show plots
    # plt.show()