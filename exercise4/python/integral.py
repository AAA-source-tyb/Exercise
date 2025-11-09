import numpy as np
from pyscf import gto, scf

def molecular_orbital_transformation():
    mol_sph = gto.M(atom="H 0 0 0; F 0 0 1", basis="6-31g", verbose=4)
    mf_sph = scf.RHF(mol_sph)
    mf_sph.kernel()
    C_sph = mf_sph.mo_coeff
    

    ovlp_sph = mol_sph.intor("int1e_ovlp_sph")
    kin_sph = mol_sph.intor("int1e_kin_sph")
    nuc_sph = mol_sph.intor("int1e_nuc_sph")
    eri_sph = mol_sph.intor("int2e_sph", aosym="s1")
    h1e_sph = kin_sph + nuc_sph
    
    # 单电子积分转换
    h1e_mo_sph = np.einsum('pi,pq,qj->ij', C_sph, h1e_sph, C_sph)
    # 双电子积分转换
    eri_mo_sph = np.einsum('pi,qj,rk,sl,pqrs->ijkl', C_sph, C_sph, C_sph, C_sph, eri_sph)
    
    mol_cart = gto.M(atom="H 0 0 0; F 0 0 1", basis="6-31g", cart=True, verbose=4)
    
    mf_cart = scf.RHF(mol_cart)  
    mf_cart.kernel()
    C_cart = mf_cart.mo_coeff
    
   
    ovlp_cart = mol_cart.intor("int1e_ovlp_cart")
    kin_cart = mol_cart.intor("int1e_kin_cart")
    nuc_cart = mol_cart.intor("int1e_nuc_cart")
    eri_cart = mol_cart.intor("int2e_cart", aosym="s1")

    h1e_cart = kin_cart + nuc_cart
    
    # 单电子积分转换
    h1e_mo_cart = np.einsum('pi,pq,qj->ij', C_cart, h1e_cart, C_cart)
    # 双电子积分转换
    eri_mo_cart = np.einsum('pi,qj,rk,sl,pqrs->ijkl', C_cart, C_cart, C_cart, C_cart, eri_cart)
    
    return {
        'sph': {
            'mol': mol_sph, 'mf': mf_sph, 'C': C_sph,
            'h1e_mo': h1e_mo_sph, 'eri_mo': eri_mo_sph,
            'h1e_ao': h1e_sph, 'eri_ao': eri_sph, 'ovlp': ovlp_sph
        },
        'cart': {
            'mol': mol_cart, 'mf': mf_cart, 'C': C_cart,
            'h1e_mo': h1e_mo_cart, 'eri_mo': eri_mo_cart,
            'h1e_ao': h1e_cart, 'eri_ao': eri_cart, 'ovlp': ovlp_cart
        }
    }

if __name__ == "__main__":
    results = molecular_orbital_transformation()
    
    print(" 球谐坐标结果 ：")
    print(f"HF能量: {results['sph']['mf'].e_tot:.8f}")
    print(f"轨道系数形状: {results['sph']['C'].shape}")
    print(f"分子轨道单电子积分形状: {results['sph']['h1e_mo'].shape}")
    print(f"分子轨道双电子积分形状: {results['sph']['eri_mo'].shape}")
    
    nocc_sph = results['sph']['mol'].nelectron // 2
    mo_energy_sph = np.diag(results['sph']['h1e_mo'])
    print(f"\n前几个分子轨道能量:")
    for i in range(min(4, len(mo_energy_sph))):
        occ = "占据" if i < nocc_sph else "空"
        print(f"轨道 {i}: {mo_energy_sph[i]:.6f} ({occ})")
    
    print(f"\n重要双电子积分:")
    print(f"(00|00) = {results['sph']['eri_mo'][0,0,0,0]:.6f}")
    
    # 显示结果 - 笛卡尔坐标
    print("\n笛卡尔坐标结果：")
    print(f"HF能量: {results['cart']['mf'].e_tot:.8f}")
    print(f"轨道系数形状: {results['cart']['C'].shape}")
    print(f"分子轨道单电子积分形状: {results['cart']['h1e_mo'].shape}")
    print(f"分子轨道双电子积分形状: {results['cart']['eri_mo'].shape}")
    
    nocc_cart = results['cart']['mol'].nelectron // 2
    mo_energy_cart = np.diag(results['cart']['h1e_mo'])
    print(f"\n前几个分子轨道能量:")
    for i in range(min(4, len(mo_energy_cart))):
        occ = "占据" if i < nocc_cart else "空"
        print(f"轨道 {i}: {mo_energy_cart[i]:.6f} ({occ})")
    
    print(f"\n重要双电子积分:")
    print(f"(00|00) = {results['cart']['eri_mo'][0,0,0,0]:.6f}")
    