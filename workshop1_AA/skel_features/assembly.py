import numpy as np
from sklearn import decomposition, preprocessing
from scipy import stats

def percentile_above_zero(X, percentile=50):
    out = []
    for x in X:
        try:
            out.append(np.percentile(x[x > 0], percentile))
        except:
            out.append(0)
    return np.array(out)

def assemble_features_from_data(df, n_syn_comp=5, n_branch_comp=3, n_syn_ego=5, model_dict=None):
    df = df.copy()
    df["tip_len_dist_dendrite_p50"] = df["tip_len_dist_dendrite"].apply(
        lambda x: np.percentile(x, 75)
    )
        
    df["tip_tort_dendrite_p50"] = df["tip_tort_dendrite"].apply(
        lambda x: np.percentile(x, 75)
    )

    df["syn_size_distribution_soma_p50"] = df["syn_size_distribution_soma"].apply(
        np.median
    )
    df["syn_dist_distribution_dendrite_p50"] = df[
        "syn_dist_distribution_dendrite"
    ].apply(np.median)
    df["syn_size_distribution_dendrite_p50"] = df[
        "syn_size_distribution_dendrite"
    ].apply(np.median)
    df["syn_size_distribution_dendrite_p15"] = df[
        "syn_size_distribution_dendrite"
    ].apply(lambda x: np.percentile(x, 15))
    df["syn_size_distribution_dendrite_p85"] = df[
        "syn_size_distribution_dendrite"
    ].apply(lambda x: np.percentile(x, 85))
    # df["syn_size_distribution_dendrite_dyn_range"] = (
    #     df["syn_size_distribution_dendrite_p90"]
    #     - df["syn_size_distribution_dendrite_p10"]
    # )
    df["syn_size_dendrite_cv"] = df["syn_size_distribution_dendrite"].apply(
        np.std
    ) / df["syn_size_distribution_dendrite"].apply(np.mean)

    df["syn_size_distribution_soma_p50"] = df["syn_size_distribution_soma"].apply(
        np.median
    )

    df["syn_depth_dist_p5"] = df["syn_depth_dist_all"].apply(
        lambda x: np.percentile(x, 2.5)
    )
    df["syn_depth_dist_p95"] = df["syn_depth_dist_all"].apply(
        lambda x: np.percentile(x, 97.5)
    )
    df["syn_depth_extent"] = df["syn_depth_dist_p95"] - df["syn_depth_dist_p5"]
    df["l1_synapse_density"] = df["syn_depth_dist_all"].apply(
        lambda x: np.sum(np.array(x)<50) / len(x)
    )

    dbr = np.vstack(df["branches_dist"].values)
    if model_dict is None:
        svd_br = decomposition.TruncatedSVD(n_branch_comp)
        Xbr = svd_br.fit_transform(dbr)
    else:
        svd_br = model_dict.get("svd_br")
        Xbr = svd_br.transform(dbr)
    for ii in range(Xbr.shape[1]):
        df[f"branch_svd{ii}"] = Xbr[:, ii]

    pl_dat = np.vstack(df["syn_count_depth_dendrite"].values)
    if model_dict is None:
        syn_pca_pproc = preprocessing.StandardScaler()
        keep_dat_cols = np.sum(pl_dat, axis=0) > 0
        pl_dat_z = syn_pca_pproc.fit_transform(pl_dat[:, keep_dat_cols])
        syn_pca = decomposition.SparsePCA(n_components=n_syn_comp)
        X = syn_pca.fit_transform(pl_dat_z)
    else:
        keep_dat_cols = model_dict.get("keep_dat_cols")
        syn_pca_pproc = model_dict.get("syn_pca_pproc")
        pl_dat_z = syn_pca_pproc.transform(pl_dat[:, keep_dat_cols])
        syn_pca = model_dict.get("syn_pca")
        X = syn_pca.transform(pl_dat_z)
    for ii in range(X.shape[1]):
        df[f"syn_count_pca{ii}"] = X[:, ii]

    pl_dat = np.vstack(df["egocentric_bins"].values)
    pl_dat_norm = pl_dat / np.atleast_2d(pl_dat.sum(axis=1)).T
    if model_dict is None:
        ego_syn_pca = decomposition.SparsePCA(n_components=n_syn_ego)
        ego_pproc = preprocessing.StandardScaler()
        pl_dat_norm_z = ego_pproc.fit_transform(pl_dat_norm)
        Xego = ego_syn_pca.fit_transform(pl_dat_norm_z)
    else:
        ego_syn_pca = model_dict.get("ego_syn_pca")
        ego_pproc = model_dict.get("ego_pproc")
        pl_dat_norm_z = ego_pproc.transform(pl_dat_norm)
        Xego = ego_syn_pca.transform(pl_dat_norm_z)
    for ii in range(Xego.shape[1]):
        df[f"ego_count_pca{ii}"] = Xego[:, ii]

    if model_dict is None:
        model_dict = {}
        model_dict["svd_br"] = svd_br
        model_dict["keep_dat_cols"] = keep_dat_cols
        model_dict["syn_pca_pproc"] = syn_pca_pproc
        model_dict["syn_pca"] = syn_pca
        model_dict["ego_syn_pca"] = ego_syn_pca
        model_dict["ego_pproc"] = ego_pproc
        

    pl_depth = np.vstack(df["path_length_depth_dendrite"].values)
    sc_depth = np.vstack(df["syn_count_depth_dendrite"].values)
    keep_cols = pl_depth.sum(axis=0) > 0
    density_nan = sc_depth[:, keep_cols] / pl_depth[:, keep_cols]
    density_nan[np.isnan(density_nan)] = 0
    density_nan[np.isinf(density_nan)] = 0
    df["max_density"] = percentile_above_zero(density_nan, 50)

    dat_cols = [
        "tip_len_dist_dendrite_p50",
        "tip_tort_dendrite_p50",
        "num_syn_dendrite",
        "num_syn_soma",
        "path_length_dendrite",
        "radial_extent_dendrite",
        "syn_dist_distribution_dendrite_p50",
        "syn_size_distribution_soma_p50",
        "syn_size_distribution_dendrite_p50",
        "syn_size_dendrite_cv",
        "syn_depth_dist_p5",
        "syn_depth_dist_p95",
        "syn_depth_extent",
        "max_density",
        "radius_dist",
        "area_factor",
        "l1_synapse_density",
    ]

    for ii in range(X.shape[1]):
        dat_cols.append(f"syn_count_pca{ii}")
    for ii in range(Xbr.shape[1]):
        dat_cols.append(f"branch_svd{ii}")
    for ii in range(Xego.shape[1]):
        dat_cols.append(f"ego_count_pca{ii}")

    return_cols = ["root_id", "soma_depth"] + dat_cols
    return df[return_cols], dat_cols, syn_pca, svd_br, keep_dat_cols, ego_syn_pca, model_dict
