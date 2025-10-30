from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths
import torch, pdb

layer = LevelSetLayer2D(size=(6,6), maxdim=1, sublevel=False)
x = torch.tensor([[2, 1, 1,1,1,1],[1,1,1,1, 0.5, 1],[1,5,2,1, 1, 1], [1,6,5, 1,3,2]], dtype=torch.float)
dgms, issublevelset = layer(x)
pdb.set_trace()
print(dgms)