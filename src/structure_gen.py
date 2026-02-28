from ase import Atoms
from ase.build import graphene_nanoribbon
from ase.io import write
import random
import os

os.makedirs("structures/pristine", exist_ok=True)
os.makedirs("structures/doped", exist_ok=True)
os.makedirs("structures/platinum", exist_ok=True)

def generate_pristine_graphene():
    g = graphene_nanoribbon(4, 4, type='armchair', vacuum=6.0)
    write("structures/pristine/graphene.xyz", g)
    return g

def dope_graphene(base_atoms, dopant="N"):
    atoms = base_atoms.copy()
    carbon_indices = [i for i, a in enumerate(atoms) if a.symbol == "C"]
    idx = random.choice(carbon_indices)
    atoms[idx].symbol = dopant
    write(f"structures/doped/graphene_{dopant}_doped.xyz", atoms)
    return atoms

def generate_platinum_cluster():
    pt = Atoms("Pt4",
               positions=[
                   (0,0,0),
                   (2.7,0,0),
                   (0,2.7,0),
                   (2.7,2.7,0)
               ])
    write("structures/platinum/pt_cluster.xyz", pt)
    return pt

if __name__ == "__main__":
    base = generate_pristine_graphene()
    dope_graphene(base, "N")
    dope_graphene(base, "S")
    generate_platinum_cluster()
    print("Structures generated successfully.")