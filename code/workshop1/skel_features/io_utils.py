import sys
import json
from pathlib import Path
import pandas as pd
from meshparty import meshwork
from . import skel_filtering as filtering

path = Path(__file__)
sys.path.append(str(path.absolute().parent))


def load_root_id(oid, nrn_dir, peel_threshold=0.1):
    "Load and apply dendrite labels to a neuron based on pre-classified synapse apical labels and more"
    nrn = meshwork.load_meshwork(f"{nrn_dir}/{oid}.h5")
    filtering.additional_component_masks(nrn, peel_threshold=peel_threshold)
    return nrn

def load_features(root_ids, feature_dir):
    "Load a feature dataframe from a directory"
    dats = []
    for root_id in root_ids:
        try:
            with open(f"{feature_dir}/{root_id}.json") as f:
                dats.append(json.load(f))
        except:
            continue
    return pd.DataFrame.from_records(dats)
