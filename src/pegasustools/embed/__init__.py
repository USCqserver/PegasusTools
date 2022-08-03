import numpy as np

def minor_embed_positions(tgt_positions, minor_embedding: dict):
    src_positions = {}
    for node, chain in minor_embedding.items():
        chain_positions = np.stack([np.asarray(tgt_positions[c]) for c in chain], axis=0)
        mean_pos = np.mean(chain_positions, axis=0)
        src_positions[node] = mean_pos
    return src_positions

