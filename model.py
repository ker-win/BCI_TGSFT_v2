# model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pickle
import os

from data_loader import Dataset
from config import cfg
from segmentation import (
    generate_time_windows,
    generate_freq_bands,
    generate_channel_groups,
    generate_sfts_specs,
    SFTSSpec, ChannelGroup, TimeWindow, FreqBand,
)
from preprocessing import Preprocessor
from feature_selection import rank_all_sfts, build_ensemble, EnsembleMember
from features import precompute_freq_bands, get_sfts_data
from divcsp import DivCSP
from divcsp_torch import DivCSPTorch


@dataclass
class FGSFTMIModelState:
    channel_groups: List[ChannelGroup]
    freq_bands: List[FreqBand]
    time_windows: List[TimeWindow]
    sfts_specs: List[SFTSSpec]
    ensemble: List[EnsembleMember]
    preprocessor: Preprocessor # Save preprocessor state


class FGSFTMIModel:
    def __init__(self):
        self.state: Optional[FGSFTMIModelState] = None

    def fit(self, dataset: Dataset) -> "FGSFTMIModel":
        """
        Main training pipeline.
        """
        print("Starting training pipeline...")
        
        # 1. Preprocess
        print("Preprocessing data...")
        # ds_proc = preprocess_pipeline(dataset)
        self.preprocessor = Preprocessor(fs_target=cfg.fs)
        self.preprocessor.fit(dataset.X, dataset.fs)
        ds_proc = self.preprocessor.transform(dataset)

        
        # 2. Generate Segments
        print("Generating segments...")
        time_windows = generate_time_windows()
        freq_bands = generate_freq_bands()
        channel_groups = generate_channel_groups(ds_proc.ch_names)
        sfts_specs = generate_sfts_specs(channel_groups, freq_bands, time_windows)
        
        print(f"Generated {len(sfts_specs)} SFTS specs.")
        
        # 3. Rank SFTS
        print("Ranking SFTS...")
        scores_sorted = rank_all_sfts(
            ds_proc, sfts_specs, channel_groups, freq_bands, time_windows
        )
        
        # 4. Build Ensemble
        print("Building ensemble...")
        top_members = build_ensemble(
            ds_proc, sfts_specs, channel_groups, freq_bands, time_windows, scores_sorted
        )
        
        self.state = FGSFTMIModelState(
            channel_groups=channel_groups,
            freq_bands=freq_bands,
            time_windows=time_windows,
            sfts_specs=sfts_specs,
            ensemble=top_members,
            preprocessor=self.preprocessor,

        )

        print("Training complete.")
        return self

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class probabilities for new data.
        Returns: (n_trials, 2)
        """
        if self.state is None:
            raise RuntimeError("Model not fitted.")
            
        # Preprocess
        # ds_proc = preprocess_pipeline(dataset)
        ds_proc = self.state.preprocessor.transform(dataset)
        
        # Precompute freq bands
        use_gpu = getattr(cfg.train_cfg, 'use_gpu', False)
        X_fband = precompute_freq_bands(ds_proc, self.state.freq_bands, use_gpu=use_gpu)

        
        n_trials = ds_proc.X.shape[0]
        n_classes = 2 # Binary
        probs_sum = np.zeros((n_trials, n_classes))
        
        # Ensemble voting
        # Algorithm 3: Average probabilities
        
        for member in self.state.ensemble:
            # Extract features for this member
            feats_list = []
            for s_id in member.sfts_ids:
                spec = next(s for s in self.state.sfts_specs if s.id == s_id)
                
                # Get data
                X_sfts = get_sfts_data(
                    X_fband, spec, self.state.channel_groups, self.state.time_windows
                )
                
                # Transform using stored CSP params
                # Transform using stored CSP params
                # Check if we should use GPU for transform
                use_gpu = getattr(cfg.train_cfg, 'use_gpu', False)
                
                if use_gpu:
                     divcsp = DivCSPTorch(device="cuda")
                else:
                     divcsp = DivCSP()
                     
                divcsp.set_params(member.csp_params[s_id])
                f = divcsp.transform(X_sfts)

                feats_list.append(f)
            
            F_concat = np.concatenate(feats_list, axis=1)
            
            # Predict proba
            # LinearSVC doesn't support predict_proba by default unless calibrated, 
            # but we can use decision_function and softmax or CalibratedClassifierCV.
            # However, for simplicity and since we used LinearSVC directly:
            # We can use decision_function and sigmoid.
            
            d = member.svm.decision_function(F_concat) # (n_trials,)
            # Sigmoid for binary
            prob_1 = 1 / (1 + np.exp(-d))
            prob_0 = 1 - prob_1
            
            probs = np.vstack([prob_0, prob_1]).T
            probs_sum += probs
            
        return probs_sum / len(self.state.ensemble)

    def predict(self, dataset: Dataset) -> np.ndarray:
        probs = self.predict_proba(dataset)
        return np.argmax(probs, axis=1)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.state, f)
            
    def load(self, path: str):
        with open(path, 'rb') as f:
            self.state = pickle.load(f)
