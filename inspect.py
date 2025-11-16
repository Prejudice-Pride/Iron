import numpy as np, os
# 修改路径如需
npz = np.load("output_videos/cotracker_only/sample_cotracker_output.npz", allow_pickle=True)
# 如果你保存的是 pred_tracks/pred_visibility
print("npz keys:", list(npz.keys()))
pred_tracks = npz["pred_tracks"]  # adjust if different key
print("pred_tracks.shape:", pred_tracks.shape, "min,max:", np.nanmin(pred_tracks), np.nanmax(pred_tracks))

# load init points CSV from LK (or from test.py variable)
import csv
init_pts = []
with open("output_videos/lk_only/sample_lk.csv","r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        idx,frame,x,y = row[0],int(row[1]),row[2],row[3]
        if frame==0 and x!="" and y!="":
            init_pts.append((int(idx), float(x), float(y)))
# build init array sorted by idx
init_pts_sorted = sorted(init_pts, key=lambda r: int(r[0]))
init = np.array([[r[1], r[2]] for r in init_pts_sorted], dtype=float)
print("num init pts:", init.shape)

T = pred_tracks.shape[0] if pred_tracks.ndim==3 and pred_tracks.shape[0] else (pred_tracks.shape[1] if pred_tracks.ndim==3 else None)
# Try to normalise pred_tracks to shape (N_cot, T, 2)
arr = pred_tracks
if arr.ndim==4 and arr.shape[0]==1:
    arr = arr[0]  # (T,N,2)
if arr.ndim==3 and arr.shape[0]==T:
    cot_tracks = np.transpose(arr, (1,0,2))
elif arr.ndim==3 and arr.shape[1]==T:
    cot_tracks = arr
else:
    print("pred_tracks shape not as expected:", arr.shape)
    cot_tracks = arr.reshape(-1, T, 2) if arr.size>=T*2 else None

print("cot_tracks.shape:", None if cot_tracks is None else cot_tracks.shape)
cot_first = cot_tracks[:,0,:] if cot_tracks is not None else None
print("cot_first sample:", cot_first[:5] if cot_first is not None else None)

# check normalized?
print("cot_first max:", np.nanmax(cot_first), "min:", np.nanmin(cot_first))

# find which cot index maps to init index 83 by distance
init_index = 83
# find position in our init array corresponding to id 83 (if we kept ordering)
# If your init CSV uses id index positional ordering, adjust accordingly.
# Here we try to find the init point with id 83:
ids = [int(r[0]) for r in init_pts_sorted]
if init_index in ids:
    pos = ids.index(init_index)
    p = init[pos]
    print("init[83] (x,y)=", p)
    dists = np.linalg.norm(cot_first - p[None,:], axis=1)
    j = int(np.argmin(dists))
    print("closest cot index:", j, "dist:", dists[j], "cot_first[j]:", cot_first[j])
else:
    print("id 83 not found in init list; please print some init entries to locate.")