import numpy as np
import pandas as pd
from meshparty import meshwork
import copy

anno_mask_dict = {
    "dendrite": "is_dendrite",
    "soma": "is_soma",
}


def peel_sparse_segments(nrn, threshold, synapse_table="post_syn"):
    """
    Take all segments, and iteratively remove segments that are both tips and have a low synapse density.
    """
    segs = copy.deepcopy(nrn.skeleton.segments)
    segs_base = [s.to_skel_index_base for s in segs]
    path_inds = [
        np.append([x[0]], nrn.skeleton.parent_nodes(x)) for x in nrn.skeleton.segments
    ]

    pl = np.array([(nrn.path_length(path) + 1) / 1000 for path in path_inds])
    num_syn = np.array(
        [nrn.anno[synapse_table].filter_query(s.to_mesh_mask).count for s in segs]
    )

    has_root = np.array([nrn.skeleton.root in s for s in segs])
    syn_dens = num_syn / pl

    removed_segs = 1
    total_removed = 0
    while removed_segs > 0:
        has_tip = np.array(
            [
                np.any(np.isin(sb, nrn.skeleton.end_points.to_skel_index_base))
                for sb in segs_base
            ]
        )

        valid_segs = np.logical_and(has_tip, ~has_root)
        remove_segments = np.logical_and(syn_dens <= threshold, valid_segs)
        if remove_segments.sum() > 0:
            mask_array = np.vstack(
                [
                    nrn.SkeletonIndex(
                        nrn.skeleton.filter_unmasked_indices(segs_base[x])
                    ).to_mesh_mask
                    for x in np.flatnonzero(remove_segments)
                ]
            )

            mask = np.sum(mask_array, axis=0) == 0
            nrn.apply_mask(mask)
        removed_segs = sum(remove_segments)
        total_removed += removed_segs

    return total_removed


def low_proximate_dendrite_mask(nrn, prox=30, path_dens_th=0.1, path_len_th=20):
    """
    Remove large branches near the soma that lack inputs
    """
    dtr = nrn.skeleton.distance_to_root / 1000
    with nrn.mask_context(nrn.skeleton_property_to_mesh(dtr) < prox):
        prox_inds = nrn.end_points.to_mesh_index_base

    path_dens = []
    path_len = []
    for pind in prox_inds:
        ds_inds = nrn.downstream_of(pind)
        num_in_ds = nrn.anno.post_syn.filter_query(ds_inds.to_mesh_mask).count
        net_len = nrn.path_length(ds_inds) / 1000
        if net_len > 0:
            path_dens.append(num_in_ds / net_len)
        else:
            path_dens.append(0)
        path_len.append(net_len)

    branch_df = pd.DataFrame(
        {"mesh_ind": prox_inds, "path_dens": path_dens, "path_len": path_len}
    )

    bad_branch = branch_df.query(
        "path_dens < @path_dens_th and path_len > @path_len_th"
    )

    if len(bad_branch) > 0:
        bad_ds = []
        for mind in bad_branch["mesh_ind"]:
            bad_ds.append(nrn.downstream_of(mind).to_mesh_mask)
        dendrite_mask = np.vstack(bad_ds).sum(axis=0) == 0
    else:
        dendrite_mask = np.full(nrn.mesh.n_vertices, True)
    return dendrite_mask

def apply_dendrite_mask(nrn, sq_th=0.6):
    """
    Use either axon/dendrite split or the low_proximate_dendrite_mask function above to get a dendrite mask.
    """
    sq = meshwork.algorithms.axon_split_quality(
        nrn.anno.is_axon.mesh_index.to_mesh_mask,
        nrn.anno.pre_syn.mesh_index,
        nrn.anno.post_syn.mesh_index,
    )

    if sq > sq_th:
        dendrite_mask = ~nrn.anno.is_axon.mesh_index.to_mesh_mask
    else:
        dendrite_mask = low_proximate_dendrite_mask(nrn)
    nrn.apply_mask(dendrite_mask)
    return nrn

def add_axon_annotation(nrn):
    if "is_axon" not in nrn.anno.table_names:
        if len(nrn.anno.pre_syn) > 0 and len(nrn.anno.post_syn) > 0:
            is_axon, split_quality = meshwork.algorithms.split_axon_by_annotation(
                nrn, 'pre_syn', 'post_syn',
            )
        else:
            split_quality = -1
            is_axon = np.full(nrn.n_vertices, False)

        nrn.anno.add_annotations("is_axon", is_axon, mask=True)
    pass

def annotate_apical_from_syn_df(nrn, syn_df):
    "Use pre-classified synapse labels to infer skeleton labels"
    apical_syn_df = syn_df.query("is_apical == True")
    if len(apical_syn_df) == 0:
        nrn.anno.add_annotations(anno_mask_dict["apical"], [], mask=True)
        return

    child_verts = nrn.child_index(nrn.root)
    child_is_apical = []
    for vert in child_verts:
        child_is_apical.append(
            np.any(np.isin(nrn.downstream_of(vert), apical_syn_df["post_pt_mesh_ind"]))
        )

    mask_stack = []
    for vert in child_verts[child_is_apical]:
        mask_stack.append(nrn.downstream_of(vert).to_mesh_mask)

    apical_mask = np.any(np.vstack(mask_stack), axis=0)
    nrn.anno.add_annotations(
        anno_mask_dict["apical"], np.flatnonzero(apical_mask), mask=True
    )


def additional_component_masks(nrn, peel_threshold=0.1):
    "Apply soma, basal dendrite, and generic dendrite masks to a neuron"

    apply_dendrite_mask(nrn)
    if peel_threshold is not None:
        peel_sparse_segments(nrn, 0.1)
    dend_mask = nrn.mesh_mask.copy()
    nrn.reset_mask()

    nrn.anno.add_annotations(
        anno_mask_dict["dendrite"], np.flatnonzero(dend_mask), mask=True
    )
    nrn.anno.add_annotations(anno_mask_dict["soma"], nrn.root_region, mask=True)

def pre_transform_neuron(nrn, st_dataset):
    nrn.skeleton.vertices = st_dataset.transform_nm.apply(nrn.skeleton.vertices)
    
    for tbl in ['pre_syn', 'post_syn']:
        for col in ['pre_pt_position', 'ctr_pt_position', 'post_pt_position']:
            nrn.anno[tbl]._data[col] = st_dataset.transform_vx.apply(
                nrn.anno[tbl]._data[col]
            )
    return nrn