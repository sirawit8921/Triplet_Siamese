from chemprop.features import get_atom_fdim, get_bond_fdim

atom_dim = get_atom_fdim()
bond_dim = get_bond_fdim()

print("atom_dim =", atom_dim)
print("bond_dim =", bond_dim)
print("total =", atom_dim + bond_dim)