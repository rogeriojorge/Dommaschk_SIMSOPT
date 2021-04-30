#!/usr/bin/env python3
######################################################
######## DOMMASCHK Optimization with pyOculus ########
########### Rogerio Jorge, April 29, 2021 ############
######################################################
from simsopt import LeastSquaresProblem,least_squares_serial_solve
from simsopt.geo.magneticfieldclasses import Dommaschk
from pyoculus.solvers  import FixedPoint, PoincarePlot
from simsopt._core.optimizable import Optimizable
from pyoculus.problems import CartesianBfield
import matplotlib.pyplot as plt
import numpy as np
# ############################################
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
# ############################################
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
    def residue(self, guess=1.165, qq=9, sbegin=1.1, send=1.17, pp=5):
        guess = guess - np.log(abs(self.get_dofs()[10])+1e0)**2/20000
        try:
            fp       = FixedPoint(self.sbsp, {"Z":0.0})
            output   = fp.compute(guess=guess, pp=pp, qq=qq, sbegin=sbegin, send=send)
            residueN = output.GreenesResidue
            R_k      = output.x
            Z_k      = output.z
            if max(R_k)<1.01:
                return [2e12, [0], [0], qq, guess]
            else:
                return [residueN, R_k, Z_k, qq, guess]
        except:
            return [1e12, [0], [0], qq, guess]
    def residue1(self): return self.residue(guess=1.180,qq=8 ,sbegin=1.165,send=1.190)[0]
    def residue2(self): return self.residue(guess=1.148,qq=9 ,sbegin=1.125,send=1.168)[0]
    def residue3(self): return self.residue(guess=1.130,qq=10,sbegin=1.080,send=1.135)[0]
    def residue4(self): return self.residue(guess=1.075,qq=11,sbegin=1.015,send=1.088)[0]
    def poincare(self, Rbegin=1.0, Rend=1.16, nPpts=150, nPtrj=35):
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
    residueInputs   = [[1.18,8,1.165,1.19],[1.148,9,1.125,1.168],[1.075,11,1.015,1.088],[1.130,10,1.08,1.135]]
    initialPoincare = 0
    # obj.set_dofs([5,2,5,4,5,10,1.4,1.4,19.25,0,-1.9e5])
    # a1=obj.residue(guess=1.18,qq=8,sbegin=1.165,send=1.19)
    # a2=obj.residue(guess=1.148,qq=9,sbegin=1.125,send=1.168)
    # a3=obj.residue(guess=1.130,qq=10,sbegin=1.08,send=1.135)
    # a4=obj.residue(guess=1.075,qq=11,sbegin=1.015,send=1.088)
    # print(a1)
    # print(a2)
    # print(a3)
    # print(a4)
    # exit()
    ## Create initial Poincare Plot
    solInitial = []
    [solInitial.append(obj.residue(guess=resIn[0],qq=resIn[1],sbegin=resIn[2],send=resIn[3])) for resIn in residueInputs]
    if initialPoincare == 1:
        p = obj.poincare(nPpts=500,nPtrj=50,Rbegin=1.0,Rend=1.18); p.plot(s=1.5)
        [plt.scatter(fSol[1], fSol[2], s=35, marker="x", label=f"Periods = {fSol[3]:.0f}, Residue = {fSol[0]:.4f}") for fSol in solInitial]
        plt.xlim([0.8 , 1.2]); plt.ylim([-0.08, 0.08]); plt.tight_layout()
        plt.legend()
        plt.savefig('Results/DommaschkInitialPoincare_pyOculus.png', dpi=500)
        plt.savefig('Results/DommaschkInitialPoincare_pyOculus.pdf')
        p.plot_iota(); plt.ylim([0.4,0.6]); plt.tight_layout()
        plt.savefig('Results/DommaschkInitialIota_pyOculus.png')

    ## Optimization
    prob = LeastSquaresProblem([(obj.residue1,0,1),
                                (obj.residue2,0,1),
                                (obj.residue4,0,1)])
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
    residueInputs = [[1.18,8,1.165,1.19],[1.148,9,1.125,1.168],[1.075,11,1.015,1.088]]
    [solFinal.append(obj.residue(guess=resIn[0],qq=resIn[1],sbegin=resIn[2],send=resIn[3])) for resIn in residueInputs]
    p = obj.poincare(nPpts=500,nPtrj=50,Rbegin=0.98,Rend=1.18); p.plot(s=1.5)
    [plt.scatter(fSol[1], fSol[2], s=35, marker="x", label=f"Periods = {fSol[3]:.0f}, Residue = {fSol[0]:.4f}") for fSol in solFinal]
    plt.xlim([0.8 , 1.2]); plt.ylim([-0.08, 0.08]); plt.tight_layout()
    plt.legend()
    plt.savefig('Results/DommaschkFinalPoincare_pyOculus.png', dpi=500)
    plt.savefig('Results/DommaschkFinalPoincare_pyOculus.pdf')
    p.plot_iota(); plt.ylim([0.4,0.6]); plt.tight_layout()
    plt.savefig('Results/DommaschkFinalIota_pyOculus.png')

    ## Print final results
    print('Initial degrees of freedom =',initialDofs)
    print('Final   degrees of freedom =',obj.get_dofs())
    print('Initial Residue1 = ',solInitial[0][0])
    print('Initial Residue1 = ',solFinal[0][0])
    print('Initial Residue2 = ',solInitial[1][0])
    print('Initial Residue2 = ',solFinal[1][0])
    print('Initial Residue3 = ',solInitial[2][0])
    print('Initial Residue3 = ',solFinal[2][0])

    # Show plots
    # plt.show()