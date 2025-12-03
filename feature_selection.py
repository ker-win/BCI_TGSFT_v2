# feature_selection.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from tqdm import tqdm

from config import cfg
from data_loader import Dataset
from segmentation import SFTSSpec, ChannelGroup, TimeWindow, FreqBand
from features import precompute_freq_bands, get_sfts_data, extract_divcsp_features_for_sfts
from divcsp import DivCSP, DivCSPParams

@dataclass
class SFTSScore:
    sfts_id: int
    acc: float

@dataclass
class EnsembleMember:
    sfts_ids: List[int]                    # SFTS indices used
    svm: LinearSVC                         # Trained SVM model
    csp_params: Dict[int, DivCSPParams]    # sfts_id -> DivCSPParams

def lobo_blocks(dataset: Dataset) -> List[int]:
    """
    Returns unique block IDs for LOBO CV.
    """
    return sorted(np.unique(dataset.blocks).tolist())

def evaluate_single_sfts_lobo(
    dataset: Dataset,
    sfts_spec: SFTSSpec,
    X_fband: Dict[int, np.ndarray],
    channel_groups: List[ChannelGroup],
    time_windows: List[TimeWindow],
) -> float:
    """
    Evaluates a single SFTS using LOBO CV.
    """
    blocks = lobo_blocks(dataset)
    y = dataset.y
    acc_list = []

    # Optimization: 
    # Instead of re-fitting CSP for every fold (which is very slow for thousands of SFTS),
    # we can fit CSP once on the whole dataset (or training set of outer loop),
    # and then just do LOBO on SVM.
    # However, strictly speaking, CSP should be fitted on training folds only to avoid leakage.
    # The paper says "The projection matrix W is calculated... using training samples".
    # So we MUST fit CSP inside the loop.
    # To speed up, we can use a smaller number of iterations or parallelize at the SFTS level.
    
    # We will implement strict LOBO here.
    
    # Pre-extract data for this SFTS to avoid repeated slicing
    X_sfts_all = get_sfts_data(X_fband, sfts_spec, channel_groups, time_windows)

    for test_block in blocks:
        mask_test = dataset.blocks == test_block
        mask_train = ~mask_test

        trial_train = np.where(mask_train)[0]
        trial_test = np.where(mask_test)[0]
        
        if len(trial_train) == 0 or len(trial_test) == 0:
            continue

        # Fit CSP on training data
        divcsp = DivCSP()
        divcsp.fit(X_sfts_all[trial_train], y[trial_train])
        
        # Transform
        f_train = divcsp.transform(X_sfts_all[trial_train])
        f_test = divcsp.transform(X_sfts_all[trial_test])

        # Train SVM
        clf = LinearSVC(
            random_state=cfg.train_cfg.random_state,
            dual=cfg.svm_cfg.dual,
            max_iter=cfg.svm_cfg.max_iter
        )
        clf.fit(f_train, y[trial_train])
        
        # Predict
        y_pred = clf.predict(f_test)
        acc_list.append(accuracy_score(y[trial_test], y_pred))

    return float(np.mean(acc_list)) if acc_list else 0.0

def rank_all_sfts(
    dataset: Dataset,
    sfts_specs: List[SFTSSpec],
    channel_groups: List[ChannelGroup],
    freq_bands: List[FreqBand],
    time_windows: List[TimeWindow],
) -> List[SFTSScore]:
    """
    Ranks all SFTS by LOBO accuracy.
    Uses parallel processing.
    """
    print(f"Precomputing frequency bands...")
    X_fband = precompute_freq_bands(dataset, freq_bands)
    
    print(f"Evaluating {len(sfts_specs)} SFTS candidates...")
    
    # Parallel execution
    n_jobs = cfg.train_cfg.n_jobs
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_single_sfts_lobo)(
            dataset, spec, X_fband, channel_groups, time_windows
        ) for spec in tqdm(sfts_specs, desc="Ranking SFTS")
    )
    
    scores = [SFTSScore(sfts_id=spec.id, acc=acc) for spec, acc in zip(sfts_specs, results)]
    
    # Sort descending
    scores_sorted = sorted(scores, key=lambda s: s.acc, reverse=True)
    return scores_sorted

def _precalc_single_sfts(
    s_id: int,
    sfts_specs: List[SFTSSpec],
    X_fband: Dict[int, np.ndarray],
    channel_groups: List[ChannelGroup],
    time_windows: List[TimeWindow],
    y: np.ndarray,
) -> Tuple[int, DivCSPParams, np.ndarray]:
    """
    Helper for parallel pre-calculation of CSP features.
    """
    spec = next(s for s in sfts_specs if s.id == s_id)
    X_sfts = get_sfts_data(X_fband, spec, channel_groups, time_windows)
    
    divcsp = DivCSP()
    divcsp.fit(X_sfts, y)
    
    params = divcsp.get_params()
    feats = divcsp.transform(X_sfts)
    
    return s_id, params, feats

def _evaluate_ensemble_step(
    j: int,
    top_j_ids: List[int],
    feats_all: Dict[int, np.ndarray],
    csp_params_all: Dict[int, DivCSPParams],
    dataset: Dataset,
    y: np.ndarray,
    blocks: List[int],
) -> Tuple[int, float, EnsembleMember]:
    """
    Helper for parallel evaluation of an ensemble step.
    """
    # LOBO CV on concatenated features
    acc_list = []
    
    # Pre-concatenate features for efficiency
    # (n_trials, j * d)
    F_all_j = np.concatenate([feats_all[sid] for sid in top_j_ids], axis=1)
    
    for test_block in blocks:
        mask_test = dataset.blocks == test_block
        mask_train = ~mask_test
        idx_train = np.where(mask_train)[0]
        idx_test = np.where(mask_test)[0]
        
        if len(idx_train) == 0 or len(idx_test) == 0:
            continue

        clf = LinearSVC(
            random_state=cfg.train_cfg.random_state,
            dual=cfg.svm_cfg.dual,
            max_iter=cfg.svm_cfg.max_iter
        )
        clf.fit(F_all_j[idx_train], y[idx_train])
        y_pred = clf.predict(F_all_j[idx_test])
        acc_list.append(accuracy_score(y[idx_test], y_pred))

    mean_acc = float(np.mean(acc_list)) if acc_list else 0.0
    
    # Train final model on ALL data
    clf_final = LinearSVC(
        random_state=cfg.train_cfg.random_state,
        dual=cfg.svm_cfg.dual,
        max_iter=cfg.svm_cfg.max_iter
    )
    clf_final.fit(F_all_j, y)
    
    # Subset csp_params for this member
    member_csp_params = {sid: csp_params_all[sid] for sid in top_j_ids}
    
    member = EnsembleMember(
        sfts_ids=top_j_ids,
        svm=clf_final,
        csp_params=member_csp_params,
    )
    
    return j, mean_acc, member

def build_ensemble(
    dataset: Dataset,
    sfts_specs: List[SFTSSpec],
    channel_groups: List[ChannelGroup],
    freq_bands: List[FreqBand],
    time_windows: List[TimeWindow],
    scores_sorted: List[SFTSScore],
) -> List[EnsembleMember]:
    """
    Builds the ensemble by iteratively adding SFTS (Algorithm 2).
    Optimized with parallel pre-calculation and parallel search.
    """
    X_fband = precompute_freq_bands(dataset, freq_bands)
    blocks = lobo_blocks(dataset)
    y = dataset.y

    D = cfg.fs_cfg.D
    K = cfg.fs_cfg.K
    max_j = len(scores_sorted)
    step = D
    
    # Identify all unique SFTS IDs needed
    # We iterate j from step to max_j.
    # The largest set is top_max_j, which includes all sfts in scores_sorted.
    all_sfts_ids = [s.sfts_id for s in scores_sorted]
    
    print(f"Pre-calculating features for {len(all_sfts_ids)} candidates in parallel...")
    n_jobs = cfg.train_cfg.n_jobs
    
    # 1. Parallel Pre-calculation
    precalc_results = Parallel(n_jobs=n_jobs)(
        delayed(_precalc_single_sfts)(
            sid, sfts_specs, X_fband, channel_groups, time_windows, y
        ) for sid in tqdm(all_sfts_ids, desc="Pre-calc Features")
    )
    
    # Store in dictionaries for fast access
    csp_params_all: Dict[int, DivCSPParams] = {}
    feats_all: Dict[int, np.ndarray] = {}
    
    for sid, params, feats in precalc_results:
        csp_params_all[sid] = params
        feats_all[sid] = feats
        
    print(f"Building ensemble (Total SFTS: {max_j}, Step: {D}) in parallel...")
    
    # 2. Parallel Ensemble Search
    # Prepare arguments for each step
    steps = range(step, max_j + 1, step)
    
    # We need to pass the specific list of IDs for each step
    step_args = []
    for j in steps:
        top_j_ids = [s.sfts_id for s in scores_sorted[:j]]
        step_args.append((j, top_j_ids))
        
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_ensemble_step)(
            j, top_j_ids, feats_all, csp_params_all, dataset, y, blocks
        ) for j, top_j_ids in tqdm(step_args, desc="Building Ensemble")
    )
    
    # Sort results by j to maintain order if needed, or just collect
    # Actually we need to sort by accuracy to pick top K
    
    # results is a list of (j, mean_acc, member)
    # We can just collect them
    
    members_with_acc = []
    for j, acc, member in results:
        members_with_acc.append((acc, member))
        # print(f"  j={j}: Acc={acc:.4f}") # Can't print easily in parallel
        
    # Sort by accuracy descending
    members_with_acc.sort(key=lambda x: x[0], reverse=True)
    
    top_members = [m for acc, m in members_with_acc[:K]]
    
    print(f"Top {K} ensemble members selected. Best Acc: {members_with_acc[0][0]:.4f}")
    
    return top_members
