import os

import torch
from Bio import PDB

def infer_O(coords):
    return coords
    N, CA, C = coords[..., 0:1, :], coords[..., 1:2, :], coords[..., 2:3, :]

    CA_N = (N - CA) / torch.linalg.vector_norm(N - CA, dim=-1).unsqueeze(-1)
    CA_C = (C - CA) / torch.linalg.vector_norm(C - CA, dim=-1).unsqueeze(-1)

    bisector = (CA_N + CA_C) / torch.linalg.vector_norm(CA_N + CA_C, dim=-1).unsqueeze(-1)
    CA_O = 1.24 * (- bisector)  # this is largely arbitrary

    O = CA + CA_O
    return torch.cat([N, CA, C, O], dim=-2)

# adapted from https://github.com/ProteinDesignLab/IgVAE/blob/main/model/utils.py
def save_pdb(xyz, pdb_out="out.pdb"):
    ATOMS = ["N","CA","C"]#,"O"]
    out = open(pdb_out,"w")
    k = 0
    a = 0
    for x,y,z in xyz:
        out.write(
            "ATOM  %5d  %2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
            #% (k+1,ATOMS[k%4],"GLY","A",a+1,x,y,z,1,0)
            % (k+1,ATOMS[k%3],"GLY","A",a+1,x,y,z,1,0)
        )
        k += 1
        if k % 3 == 0: a += 1
        # if k % 4 == 0: a += 1
    out.close()

def save_fused_pdbs(xyzs, pdb_out="out.pdb"):
    pdb_parser = PDB.PDBParser()
    pdb_io = PDB.PDBIO()

    ms = PDB.Structure.Structure("master")

    structures = []
    for i, xyz in enumerate(xyzs):
        save_pdb(xyz, "tmp.pdb")
        struct = pdb_parser.get_structure(f"step{i}", "tmp.pdb")
        structures.append(struct)
    os.remove("tmp.pdb")

    i=0
    for structure in structures:
        for model in list(structure):
            new_model=model.copy()
            new_model.id=i
            new_model.serial_num=i+1
            i=i+1
            ms.add(new_model)

    pdb_io.set_structure(ms)
    pdb_io.save(pdb_out)
