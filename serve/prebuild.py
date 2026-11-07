"""Build every CUDA extension the lane loads, once per node, before the first start: compiling while four ranks load their
weights costs memory the lane needs. Run inside the serving container with this tree at /w (see the README)."""
import sys, time
sys.path.insert(0, '/w/src/tensorfold/src')
t = time.time()
from tensorfold.families.mistral_large4.cuda import cp4_kernels as C, tp_kernels as P
C._mla(); print('mla', round(time.time() - t), flush=True)
C._score8(); print('score8', round(time.time() - t), flush=True)
P._select(); print('select', round(time.time() - t), flush=True)
C._score4(); print('score4', round(time.time() - t), flush=True)
from tensorfold.cuda.exl3 import experts as X
X._ext(); print('experts', round(time.time() - t), flush=True)
from tensorfold.cuda.exl3 import linear as L
L._ext(); print('linear', round(time.time() - t), flush=True)
from tensorfold.cuda.exl3 import prefetch as F
F._ext(); print('prefetch', round(time.time() - t), flush=True)
from tensorfold.cuda import rdma as RD
RD._ext(); print('rdma', round(time.time() - t), flush=True)
