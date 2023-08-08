import os
import orjson
import json
import numpy as np
from .io_utils import load_root_id
from .skel_filtering import anno_mask_dict, pre_transform_neuron
from scipy import sparse
from pathos.pools import ProcessPool

radius_bins = np.arange(20, 300, 30)
radius_bin_width = 10


def make_depth_bins(height_bounds, spacing=50):
    return np.linspace(height_bounds[0], height_bounds[1], spacing)


def make_egocentric_bins(max_d, nbins=14):
    return np.linspace(-max_d, max_d, nbins)


def tip_len_dist(nrn, compartment=None):
    eps = nrn.end_points
    if compartment is not None:
        eps = eps[nrn.anno[anno_mask_dict[compartment]].mesh_mask[eps]]
    if len(eps) == 0:
        return None
    return nrn.distance_to_root(eps)


def tip_tort(nrn, compartment=None):
    eps = nrn.end_points
    if compartment is not None:
        eps = eps[nrn.anno[anno_mask_dict[compartment]].mesh_mask[eps]].to_skel_index
    if len(eps) == 0:
        return None
    dtr = nrn.skeleton.distance_to_root[eps]
    euc_dist = np.linalg.norm(
        (
            nrn.skeleton.vertices[eps]
            - np.atleast_2d(nrn.skeleton.vertices[nrn.skeleton.root])
        ),
        axis=1,
    )
    tort = dtr / euc_dist
    return tort


def num_syn(nrn, compartment=None):
    if compartment is None:
        df = nrn.anno.post_syn.df
    else:
        df = nrn.anno.post_syn.filter_query(
            nrn.anno[anno_mask_dict[compartment]].mesh_mask
        ).df
    return len(df)


def syn_size_distribution(nrn, compartment=None):
    if compartment is None:
        df = nrn.anno.post_syn.df
    else:
        df = nrn.anno.post_syn.filter_query(
            nrn.anno[anno_mask_dict[compartment]].mesh_mask
        ).df
    return df["size"].values


def syn_dist_distribution(nrn, compartment=None):
    if compartment is None:
        minds = nrn.anno.post_syn.mesh_index
    else:
        minds = nrn.anno.post_syn.filter_query(
            nrn.anno[anno_mask_dict[compartment]].mesh_mask
        ).mesh_index
    return nrn.distance_to_root(minds)


def syn_depth_dist(nrn, compartment=None):
    if compartment is None:
        df = nrn.anno.post_syn.df
    else:
        df = nrn.anno.post_syn.filter_query(
            nrn.anno[anno_mask_dict[compartment]].mesh_mask
        ).df
    if len(df) > 0:
        return np.vstack(df["ctr_pt_position"])[:, 1].copy(order='C')
    else:
        return np.ascontiguousarray([])


def _is_between(xs, a, b):
    return np.logical_and(xs > a, xs <= b)


def _branches_between(nrn, d_a, d_b, min_thresh):
    gap = _is_between(nrn.skeleton.distance_to_root, d_a, d_b)
    G = nrn.skeleton.csgraph_binary_undirected
    _, ccs = sparse.csgraph.connected_components(G[:, gap][gap])
    _, nvs = np.unique(ccs, return_counts=True)
    return sum(nvs > min_thresh)


def branches_between(nrn, d_a, d_b, min_thresh=1, compartment=None):
    if compartment is None:
        return _branches_between(nrn, d_a, d_b, min_thresh)
    else:
        try:
            with nrn.mask_context(
                nrn.anno[anno_mask_dict[compartment]].mesh_mask
            ) as nmc:
                return _branches_between(nmc, d_a, d_b, min_thresh)
        except:
            return None


def branches_with_distance(nrn, radius_bins, radius_bin_width, compartment=None):
    return [
        branches_between(nrn, d_a, (d_a + radius_bin_width), compartment=compartment)
        for d_a in radius_bins
    ]


def path_length(nrn, compartment=None):
    try:
        with nrn.mask_context(nrn.anno[anno_mask_dict[compartment]].mesh_mask) as nmc:
            return nmc.path_length()
    except:
        return None


def horizontal_extent(nrn, rdist_func, compartment=None):
    try:
        with nrn.mask_context(nrn.anno[anno_mask_dict[compartment]].mesh_mask) as nmc:
            d = rdist_func(
                nrn.skeleton.root_position,
                nmc.skeleton.vertices,
                transform_points=False,
            )
            if len(d) == 0:
                return None
        return np.percentile(d, 97).copy(order='C')
    except:
        return None


def soma_depth(nrn):
    return nrn.skeleton.root_position[1]


def make_depth_bins(height_bounds, nbins=50):
    return np.linspace(height_bounds[0], height_bounds[1], nbins)


def _node_weight(nrn):
    return np.squeeze(np.array(np.sum(nrn.skeleton.csgraph_undirected, axis=0)) / 2)


def _path_length_binned(nrn, depth_bins):
    ws = _node_weight(nrn)
    sk_vert_y = nrn.skeleton.vertices[:, 1]
    lens = []
    for d_a, d_b in zip(depth_bins[:-1], depth_bins[1:]):
        lens.append(np.sum(ws[_is_between(sk_vert_y, d_a, d_b)]))
    return np.array(lens)


def path_length_binned(nrn, depth_bins, compartment=None):
    if compartment is None:
        return _path_length_binned(nrn, depth_bins)
    else:
        try:
            with nrn.mask_context(
                nrn.anno[anno_mask_dict[compartment]].mesh_mask
            ) as nmc:
                return _path_length_binned(nmc, depth_bins)
        except:
            return None
    pass


def _syn_count_binned(nrn, depth_bins):
    syn_y = np.vstack(nrn.anno.post_syn.df["ctr_pt_position"])[:, 1]
    n_syn = []
    for ymin, ymax in zip(depth_bins[:-1], depth_bins[1:]):
        n_syn.append(sum(_is_between(syn_y, ymin, ymax)))
    return np.array(n_syn)


def syn_count_binned(nrn, depth_bins, compartment=None):
    if compartment is None:
        return _syn_count_binned(nrn, depth_bins)
    else:
        try:
            with nrn.mask_context(
                nrn.anno[anno_mask_dict[compartment]].mesh_mask
            ) as nmc:
                return _syn_count_binned(nmc, depth_bins)
        except:
            return None


def syn_count_egocentric(nrn, soma_depth, rel_bins):
    syn_y = np.vstack(nrn.anno.post_syn.df["ctr_pt_position"])[:, 1] - soma_depth
    n_syn = []
    for ymin, ymax in zip(rel_bins[:-1], rel_bins[1:]):
        n_syn.append(sum(_is_between(syn_y, ymin, ymax)))
    return np.array(n_syn)


def median_radius_close(nrn, rad_anno="r_eff", anno_name="segment_properties", dist=65):
    with nrn.mask_context(nrn.anno.is_dendrite.mesh_mask) as nrnf:
        seg_df = nrnf.anno[anno_name].df
        seg_df["dtr"] = nrnf.distance_to_root(nrnf.anno[anno_name].mesh_index)
    return seg_df.query("dtr < @dist and dtr > 20")[rad_anno].median()


def median_radius_distal(
    nrn, rad_anno="r_eff", anno_name="segment_properties", dist=65
):
    with nrn.mask_context(nrn.anno.is_dendrite.mesh_mask) as nrnf:
        seg_df = nrnf.anno[anno_name].df
        seg_df["dtr"] = nrnf.distance_to_root(nrnf.anno[anno_name].mesh_index)
    if len(seg_df.query("dtr > @dist")) > 0:
        return seg_df.query("dtr > @dist")[rad_anno].median()
    else:
        return 0


def area_factor(
    nrn,
    area_factor="area_factor",
    anno_name="segment_properties",
    dist=20,
):
    with nrn.mask_context(nrn.anno.is_dendrite.mesh_mask) as nrnf:
        seg_df = nrnf.anno[anno_name].df
        seg_df["dtr"] = nrnf.distance_to_root(nrnf.anno[anno_name].mesh_index)
    return seg_df.query("dtr > @dist")[area_factor].median()


def extract_features_dict(
    nrn,
    radius_bins,
    radius_bin_width,
    depth_bins,
    egocentric_bins,
    sl_dataset,
):
    nrn = pre_transform_neuron(nrn, sl_dataset)
    return {
        "root_id": nrn.seg_id,
        "soma_depth": soma_depth(nrn),
        "tip_len_dist_dendrite": tip_len_dist(nrn, "dendrite"),
        "tip_tort_dendrite": tip_tort(nrn, "dendrite"),
        "num_syn_dendrite": num_syn(nrn, "dendrite"),
        "num_syn_soma": num_syn(nrn, "soma"),
        "syn_size_distribution_soma": syn_size_distribution(nrn, "soma"),
        "syn_size_distribution_dendrite": syn_size_distribution(nrn, "dendrite"),
        "syn_dist_distribution_dendrite": syn_dist_distribution(nrn, "dendrite"),
        "syn_depth_dist_all": syn_depth_dist(nrn, "dendrite"),
        "radial_extent_dendrite": horizontal_extent(
            nrn, sl_dataset.streamline_nm.radial_distance, "dendrite"
        ),
        "path_length_dendrite": path_length(nrn, "dendrite"),
        "branches_dist": branches_with_distance(
            nrn, radius_bins, radius_bin_width, compartment="dendrite"
        ),
        "path_length_depth_dendrite": path_length_binned(
            nrn, depth_bins, compartment="dendrite"
        ),
        "syn_count_depth_dendrite": syn_count_binned(
            nrn, depth_bins, compartment="dendrite"
        ),
        "radius_dist": median_radius_distal(nrn, dist=30),
        "area_factor": area_factor(nrn),
        "egocentric_bins": syn_count_egocentric(nrn, soma_depth(nrn), egocentric_bins),
        "success": True,
    }


def extract_features(
    nrn,
    height_bounds,
    sl_dataset,
    feature_dir=None,
    filename=None,
    n_egocentric_bins=14,
):
    try:
        depth_bins = make_depth_bins(height_bounds)
        egocentric_bins = make_egocentric_bins(100, n_egocentric_bins)
        features = extract_features_dict(
            nrn, radius_bins, radius_bin_width, depth_bins, egocentric_bins, sl_dataset
        )
    except:
        features = {
            "root_id": nrn.seg_id,
            "success": False,
        }
    if feature_dir:
        if filename is None:
            filename = f"{nrn.seg_id}"
        with open(f"{feature_dir}/{filename}.json", "wb") as f:
            f.write(orjson.dumps(features, option=orjson.OPT_SERIALIZE_NUMPY))
    return features


def extract_features_root_id(
    root_id,
    skel_dir,
    height_bounds,
    sl_dataset,
    feature_dir,
    n_egocentric_bins=14,
    rerun=False,
):
    if os.path.exists(f"{feature_dir}/{root_id}.json") and not rerun:
        with open(f"{feature_dir}/{root_id}.json") as f:
            dat = json.load(f)
            if dat["success"] is True:
                return True
    try:
        nrn = load_root_id(root_id, skel_dir)
        features = extract_features(
            nrn,
            height_bounds,
            sl_dataset,
            feature_dir=feature_dir,
            n_egocentric_bins=n_egocentric_bins,
        )
        return features["success"]
    except Exception as e:
        print(e)
        return False


def extract_features_mp(
    root_ids,
    skel_dir,
    height_bounds,
    sl_dataset,
    feature_dir,
    n_egocentric_bins=14,
    rerun=False,
    nodes=8,
):
    pool = ProcessPool(nodes=nodes)
    return np.array(
        pool.map(
            extract_features_root_id,
            root_ids,
            [skel_dir] * len(root_ids),
            [height_bounds] * len(root_ids),
            [sl_dataset] * len(root_ids),
            [feature_dir] * len(root_ids),
            [n_egocentric_bins] * len(root_ids),
            [rerun] * len(root_ids),
        )
    )
