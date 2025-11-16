import numpy as np
d = np.load("output_videos/lk_only/sample_lk.npz", allow_pickle=True)
print(d.files)
# adjust key names as saved
tracks = d["tracks"] if "tracks" in d.files else d["arr_0"]
print("tracks shape:", tracks.shape)
print("tracks[83,0]:", tracks[83,0])