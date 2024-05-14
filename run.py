from pathlib import Path
import copy

import torch
import pyscf
import numpy as np

from cadft.utils import FCNet, gen_dm1, gen_f_mat
from cadft.utils import NAO
from cadft import CC_DFT_DATA, Mol


key_l = []
model_dict = {}

ATOM_LIST = [
    "H",
    "C",
]

molecular = copy.deepcopy(Mol["Methane"])
distance = 0.1
print(f"Distance: {distance}")
molecular[0][1] += distance

dft2cc = CC_DFT_DATA(
    molecular,
    name="test",
    basis="cc-pvdz",
    if_basis_str=True,
)

mf = pyscf.scf.RHF(dft2cc.mol)
mf.kernel()
mycc = pyscf.cc.CCSD(mf)
mycc.kernel()

dm1_cc = mycc.make_rdm1(ao_repr=True)
e_cc = mycc.e_tot

mdft = pyscf.scf.RKS(dft2cc.mol)
mdft.xc = "b3lyp"
mdft.kernel()
dm1_dft = mdft.make_rdm1(ao_repr=True)

dir_checkpoint = Path("checkpoints/checkpoint-2024-05-08-21-00-34-400/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i_atom in ATOM_LIST:
    for j_atom in ATOM_LIST:
        atom_name = i_atom + j_atom
        key_l.append(atom_name)

        model_dict[atom_name + "2"] = FCNet(NAO[i_atom] * NAO[j_atom], 400, 1).to(
            device
        )
        model_dict[atom_name + "2"].double()
        list_of_path = dir_checkpoint.glob(f"{atom_name}-2-*.pth")
        load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
        state_dict = torch.load(load_path, map_location=device)
        model_dict[atom_name + "2"].load_state_dict(state_dict)
        print(f"Model loaded from {load_path}")

        model_dict[atom_name + "1"] = FCNet(
            NAO[i_atom] * NAO[j_atom], 400, NAO[i_atom] * NAO[j_atom]
        ).to(device)
        model_dict[atom_name + "1"].double()
        list_of_path = dir_checkpoint.glob(f"{atom_name}-1-*.pth")
        load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
        state_dict = torch.load(load_path, map_location=device)
        model_dict[atom_name + "1"].load_state_dict(state_dict)
        print(f"Model loaded from {load_path}")

dm1_predict = gen_dm1(dft2cc, dm1_dft, model_dict, device)
print(np.mean(np.abs(dm1_predict - dm1_cc)))
f_mat, ene_xc = gen_f_mat(dft2cc, dm1_predict, model_dict, device)
h1e = dft2cc.mol.intor("int1e_nuc") + dft2cc.mol.intor("int1e_kin")
eri = dft2cc.mol.intor("int2e")
print(
    1000
    * (
        ene_xc
        + np.einsum("pqrs,pq,rs", eri, dm1_predict, dm1_predict) / 2
        + np.sum(h1e * dm1_predict)
        + dft2cc.mol.energy_nuc()
        - e_cc
    )
)
print(
    1000
    * (
        ene_xc
        + np.einsum("pqrs,pq,rs", eri, dm1_predict, dm1_predict) / 2
        + np.sum(h1e * dm1_predict)
        + dft2cc.mol.energy_nuc()
        - e_cc
    )
)
