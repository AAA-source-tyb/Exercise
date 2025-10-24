import ctypes
from build import build_lib
import numpy as np

_cint = build_lib()

_cint.cint1e_ovlp_sph.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.double, ndim=2),
    (ctypes.c_int * 2),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1),
]
_cint.cint1e_ovlp_sph.restype = ctypes.c_int

_cint.cint1e_ovlp_cart.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.double, ndim=2),
    (ctypes.c_int * 2),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1),
]
_cint.cint1e_ovlp_cart.restype = ctypes.c_int

def get_shell_dim_sph(bas, i):
    ang = bas[i, 1]  
    ncont = bas[i, 3]  
    if ncont > 1:
        return ncont * (2 * ang + 1)
    else:
        return 2 * ang + 1 

def get_shell_dim_cart(bas, i):
    ang = bas[i, 1]  
    return (ang + 1) * (ang + 2) // 2  

def get_shell_loc_sph(bas, i):
    loc = 0
    for j in range(i):
        loc += get_shell_dim_sph(bas, j)
    return loc

def get_shell_loc_cart(bas, i):
    loc = 0
    for j in range(i):
        loc += get_shell_dim_cart(bas, j)
    return loc

def ovlp_sph(mol):
    atm = mol._atm
    bas = mol._bas
    env = mol._env
    natm = atm.shape[0]
    nbas = bas.shape[0]
    
    total_dim = 0
    for i in range(nbas):
        total_dim += get_shell_dim_sph(bas, i)

    S = np.zeros((total_dim, total_dim))
    
    for i in range(nbas):
        for j in range(nbas):
            di = get_shell_dim_sph(bas, i)
            dj = get_shell_dim_sph(bas, j)
            buf = np.empty((di, dj), order="F")
            
            if (_cint.cint1e_ovlp_sph(
                buf,
                (ctypes.c_int * 2)(i, j),
                atm,
                natm,
                bas,
                nbas,
                env,
            ) == 0):
                raise RuntimeError("cint1e_ovlp_sph failed")
            
            loc_i = get_shell_loc_sph(bas, i)
            loc_j = get_shell_loc_sph(bas, j)
            S[loc_i:loc_i+di, loc_j:loc_j+dj] = buf
    
    return S

def ovlp_cart(mol):
    atm = mol._atm
    bas = mol._bas
    env = mol._env
    natm = atm.shape[0]
    nbas = bas.shape[0]
    
    total_dim = 0
    for i in range(nbas):
        total_dim += get_shell_dim_cart(bas, i)
    
    S = np.zeros((total_dim, total_dim))
    
    for i in range(nbas):
        for j in range(nbas):
            di = get_shell_dim_cart(bas, i)
            dj = get_shell_dim_cart(bas, j)
            buf = np.empty((di, dj), order="F")
            
            if (_cint.cint1e_ovlp_cart(
                buf,
                (ctypes.c_int * 2)(i, j),
                atm,
                natm,
                bas,
                nbas,
                env,
            ) == 0):
                raise RuntimeError("cint1e_ovlp_cart failed")
            
            loc_i = get_shell_loc_cart(bas, i)
            loc_j = get_shell_loc_cart(bas, j)
            S[loc_i:loc_i+di, loc_j:loc_j+dj] = buf
    
    return S

if __name__ == "__main__":
    from pyscf import gto

    mol = gto.M(
        atom="O 0.0 0.0 0.0",
        basis="anorcc",
        spin=0, 
        charge=0,
    )
    
    print(" 球谐坐标 ")
    S_sph = ovlp_sph(mol)
    print(S_sph)

    ovlp = mol.intor("int1e_ovlp")
    print(ovlp)

    assert np.allclose(S_sph, ovlp)

    print(mol._bas)