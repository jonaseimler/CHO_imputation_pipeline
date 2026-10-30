"""
impute_methods.py
─────────────────────────────────────────────────────────────────────────────
Imputation methods. Each imputer has impute(X_missing) -> DataFrame, where
X_missing is a time-indexed DataFrame with NaN in the gaps.

  BaselineImputerLinear / BaselineImputerSpline (PCHIP)
  BaselineImputerLinearSmoothed / BaselineImputerSplineSmoothed
      interpolation between the edges of a rolling mean
  FLkNNImputer   Fourier + lagged k-NN (Rahman et al. 2015) with endpoint stitching
  HankelImputer  iterative low-rank block-Hankel completion (SSA / Cadzow)
"""

import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from scipy.signal import correlate


class BaselineImputerLinear:
    def __init__(self):
        pass

    def impute(self, X_missing):
        df = X_missing.copy()
        # time-weighted for a DatetimeIndex, index-weighted otherwise
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.interpolate(method='time', limit_direction='both')
        else:
            df = df.interpolate(method='index', limit_direction='both')

        df = df.bfill().ffill()
        return df


class BaselineImputerSpline:
    def __init__(self, method='pchip'):
        """method: 'pchip' (shape-preserving, used in the study) or 'cubicspline'."""
        self.method = method

    def impute(self, X_missing):
        df = X_missing.copy()
        # both pass exactly through the observed edge values
        df = df.interpolate(method=self.method, limit_direction='both')
        df = df.bfill().ffill()
        return df


class BaselineImputerLinearSmoothed:
    def __init__(self, window_size=500):
        self.window_size = window_size

    def impute(self, X_missing):
        X_imputed = X_missing.copy()

        for col in X_missing.columns:
            series = X_missing[col]
            if not series.isna().any():
                continue

            is_nan = series.isna()
            not_nan = ~is_nan

            smoothed_series = series.copy()
            segment_id = (not_nan != not_nan.shift()).cumsum()

            for name, group in series[not_nan].groupby(segment_id[not_nan]):
                # rolling mean of the observed segment; edges are not pinned to the raw values
                temp_smoothed = group.rolling(window=self.window_size, min_periods=1, center=True).mean()

                smoothed_series.loc[group.index] = temp_smoothed

            # interpolate between the smoothed edges
            if isinstance(smoothed_series.index, pd.DatetimeIndex):
                imputed_smoothed = smoothed_series.interpolate(method='time', limit_direction='both')
            else:
                imputed_smoothed = smoothed_series.interpolate(method='index', limit_direction='both')

            # only gap values are replaced; observed values stay raw
            X_imputed.loc[is_nan, col] = imputed_smoothed.loc[is_nan]

        return X_imputed


class BaselineImputerSplineSmoothed:
    def __init__(self, window_size=500, method='pchip'):
        self.window_size = window_size
        self.method = method

    def impute(self, X_missing):
        X_imputed = X_missing.copy()

        for col in X_missing.columns:
            series = X_missing[col]
            if not series.isna().any():
                continue

            is_nan = series.isna()
            not_nan = ~is_nan

            smoothed_series = series.copy()
            segment_id = (not_nan != not_nan.shift()).cumsum()

            for name, group in series[not_nan].groupby(segment_id[not_nan]):
                # rolling mean of the observed segment; edges are not pinned to the raw values
                temp_smoothed = group.rolling(window=self.window_size, min_periods=1, center=True).mean()

                smoothed_series.loc[group.index] = temp_smoothed

            # interpolate between the smoothed edges
            imputed_smoothed = smoothed_series.interpolate(method=self.method, limit_direction='both')

            # only gap values are replaced; observed values stay raw
            X_imputed.loc[is_nan, col] = imputed_smoothed.loc[is_nan]

        return X_imputed

class FLkNNImputer:
    def __init__(self, k=5, num_lags=3, max_lag=60, fourier_descriptor_percent=100,
                 random_state=42):
        """
        k: Number of nearest neighbors to average.
        num_lags: Number of lag matrices to build (top-n cross-correlation lags).
        max_lag: Maximum lag (in samples) to search.
        fourier_descriptor_percent: % of Fourier coefficients to retain.
        random_state: Seed for reproducible random candidate sampling.
        """
        self.k = k
        self.num_lags = num_lags
        self.max_lag = max_lag
        self.fd_percent = fourier_descriptor_percent
        self.random_state = random_state

    def impute(self, X_missing):
        """Combine Fourier and Lagged k-NN imputation (FLk-NN, Rahman et al. 2015)."""
        X_fourier = self._fourier_impute(X_missing)
        X_knn = self._knn_lag_impute(X_missing)

        val_f = X_fourier.values
        val_k = X_knn.values
        # free the DataFrames early (memory)
        del X_fourier, X_knn

        if val_f.shape != val_k.shape:
            imputed_vals = val_f if val_f.shape == X_missing.shape else val_k
        else:
            stacked = np.dstack((val_f, val_k))
            imputed_vals = np.nanmean(stacked, axis=2)
            del stacked
        del val_f, val_k

        mask = X_missing.isna()
        if imputed_vals.shape == X_missing.shape:
            X_combined = pd.DataFrame(imputed_vals, index=X_missing.index,
                                      columns=X_missing.columns)
            del imputed_vals
            X_combined[~mask] = X_missing[~mask]

            # ── Endpoint stitching ──
            # Add a linear correction across each gap so that the imputed segment
            # joins the observed edges without a step (offsets mainly from Fourier).
            for col in X_missing.columns:
                is_nan_col = mask[col].values
                if not is_nan_col.any():
                    continue

                y_vals = X_combined[col].values.copy()
                diff = np.diff(is_nan_col.astype(int), prepend=0, append=0)
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]

                for st, en in zip(starts, ends):
                    n_gap = en - st
                    if n_gap < 1:
                        continue

                    imp_seg = y_vals[st:en].copy()

                    val_before = y_vals[st-1] if st > 0 else None
                    val_after = y_vals[en] if en < len(y_vals) else None

                    if val_before is None and val_after is None:
                        continue

                    if n_gap >= 2:
                        pred_before = imp_seg[0] - (imp_seg[1] - imp_seg[0])
                        pred_after = imp_seg[-1] + (imp_seg[-1] - imp_seg[-2])
                    else:
                        pred_before = imp_seg[0]
                        pred_after = imp_seg[-1]

                    E_start = val_before - pred_before if val_before is not None else (val_after - pred_after)
                    E_end = val_after - pred_after if val_after is not None else (val_before - pred_before)

                    E_interp = np.linspace(E_start, E_end, n_gap + 2)[1:-1]
                    y_vals[st:en] = imp_seg + E_interp

                X_combined[col] = y_vals
        else:
            X_combined = X_missing.copy()
            X_combined = X_combined.interpolate()

        if X_combined.isna().any().any():
            X_combined = X_combined.interpolate(method='linear', limit_direction='both')
            X_combined = X_combined.bfill().ffill()

        return X_combined

    def _fourier_impute(self, X_in):
            """
            Fourier imputation for block gaps with linear detrending, which avoids
            Gibbs ringing for growing signals such as biomass.
            """
            from scipy.fftpack import fft, ifft
            X = X_in.copy()
            n = len(X)
            t = np.arange(n)

            for col in X.columns:
                series = X[col].values.copy()
                if not np.isnan(series).any():
                    continue

                # 1. temporary linear fill
                series_temp = pd.Series(series).interpolate(method='linear', limit_direction='both').bfill().ffill().values

                # subtract the line through first and last value, so the signal starts
                # and ends near 0 (no spectral leakage / ringing)
                start_val = series_temp[0]
                end_val = series_temp[-1]
                trend_line = start_val + (end_val - start_val) * (t / (n - 1 if n > 1 else 1))

                series_detrended = series_temp - trend_line

                # 2. FFT of the detrended signal
                f_sig = fft(series_detrended)

                # 3. keep the lowest fd_percent of the coefficients
                desc_len = int(np.ceil(len(f_sig) * (self.fd_percent / 100.0)))

                coeffs = np.zeros_like(f_sig)
                if desc_len > 0:
                    coeffs[:desc_len] = f_sig[:desc_len]
                    if desc_len > 1:
                        coeffs[-desc_len+1:] = f_sig[-desc_len+1:]

                # 4. inverse FFT
                rec_sig_detrended = np.real(ifft(coeffs))

                # add the trend back
                rec_sig = rec_sig_detrended + trend_line

                # 5. fill only the gap points
                missing_mask = np.isnan(series)
                series[missing_mask] = rec_sig[missing_mask]

                X[col] = series

            return X


    def _knn_lag_impute(self, X_in):
        """
        Lagged k-NN imputation (Rahman et al. 2015).
          - Correlation-weighted distance (Eq. 6 of the paper).
          - Global candidate pool across all lag matrices; top-k selected from the pool.
          - Deterministic sampling via self.random_state.
        """
        X = X_in.copy()
        values = X.values.astype(float)
        n_rows, n_cols = values.shape
        rng = np.random.default_rng(self.random_state)

        # ── Linearly prefilled copy, used ONLY for lags and correlation weights ──
        # (dropping NaN rows instead would compress time and bias the lags)
        values_interp = (pd.DataFrame(values)
                         .interpolate(method='linear', limit_direction='both')
                         .bfill().ffill().fillna(0.0).values)

        # ── 1. Compute top-num_lags cross-correlation lags for every variable pair ──
        lags = np.zeros((n_cols, n_cols, self.num_lags), dtype=int)

        for i in range(n_cols):
            lags[i, i, :] = 0
            for j in range(i + 1, n_cols):
                s1 = values_interp[:, i]
                s2 = values_interp[:, j]

                cc = correlate(s1, s2, method='fft')

                # Restrict to ±max_lag around the zero-lag index.
                center = len(s2) - 1
                start = max(0, center - self.max_lag)
                end = min(len(cc), center + self.max_lag + 1)
                cc_window = cc[start:end]
                abs_cc = np.abs(cc_window)

                num_to_pick = min(self.num_lags, len(abs_cc))
                top_idx = np.argpartition(abs_cc, -num_to_pick)[-num_to_pick:]
                top_idx = top_idx[np.argsort(abs_cc[top_idx])[::-1]]

                for k_idx, raw_idx in enumerate(top_idx):
                    lag_val = (raw_idx + start) - center  # actual lag in samples
                    lags[i, j, k_idx] = lag_val
                    lags[j, i, k_idx] = -lag_val

        # ── 2. Normalisation constants (computed once on original data) ──
        v_min = np.nanmin(values, axis=0)
        v_max = np.nanmax(values, axis=0)
        range_vals = v_max - v_min
        range_vals[range_vals == 0] = 1.0

        # ── 3. Imputation loop ──
        for col_idx in range(n_cols):
            missing_indices = np.where(np.isnan(values[:, col_idx]))[0]
            if len(missing_indices) == 0:
                continue

            # Correlation weights |corr(target, j)| (Eq. 6), on the prefilled signals.
            # Normalized later (w_norm) over the variables available at each query.
            corr_weights = np.zeros(n_cols)
            for j in range(n_cols):
                if j == col_idx:
                    continue
                r = np.corrcoef(values_interp[:, col_idx], values_interp[:, j])[0, 1]
                corr_weights[j] = abs(r) if not np.isnan(r) else 0.0
            # no measurable correlation → equal weights
            if corr_weights.sum() == 0:
                corr_weights[:] = 1.0

            valid_target_rows = np.where(~np.isnan(values[:, col_idx]))[0]

            for row_idx in missing_indices:
                # global candidate pool across all lag matrices
                all_dists = []
                all_vals = []

                for l in range(self.num_lags):
                    current_lags = lags[col_idx, :, l]

                    # Build query feature vector: observed other-variable values at
                    # their lag-shifted times relative to row_idx.
                    indices_j = []
                    for j in range(n_cols):
                        if j == col_idx:
                            continue
                        t_shifted = row_idx + int(current_lags[j])
                        if 0 <= t_shifted < n_rows and not np.isnan(values[t_shifted, j]):
                            indices_j.append(j)

                    if not indices_j:
                        continue

                    indices_j = np.array(indices_j)
                    offsets = np.array([int(current_lags[j]) for j in indices_j])
                    query_t = row_idx + offsets
                    query_vec = values[query_t, indices_j]
                    query_norm = (query_vec - v_min[indices_j]) / range_vals[indices_j]

                    # random subset of at most 500 candidate rows
                    if len(valid_target_rows) > 500:
                        search_rows = rng.choice(valid_target_rows, 500, replace=False)
                    else:
                        search_rows = valid_target_rows

                    cand_t_matrix = search_rows[:, None] + offsets  # (len(search_rows), len(indices_j))

                    # drop candidates whose lagged times fall outside the series
                    in_bounds_mask = ~np.any((cand_t_matrix < 0) | (cand_t_matrix >= n_rows), axis=1)
                    if not np.any(in_bounds_mask):
                        continue

                    valid_search_rows = search_rows[in_bounds_mask]
                    valid_cand_t = cand_t_matrix[in_bounds_mask]

                    cand_vecs = np.zeros((len(valid_search_rows), len(indices_j)))
                    for idx_j_local, j in enumerate(indices_j):
                        cand_vecs[:, idx_j_local] = values[valid_cand_t[:, idx_j_local], j]

                    # drop candidates with missing features
                    nan_mask = ~np.any(np.isnan(cand_vecs), axis=1)
                    if not np.any(nan_mask):
                        continue

                    final_search_rows = valid_search_rows[nan_mask]
                    final_cand_vecs = cand_vecs[nan_mask]

                    # min-max normalization and weighted distance
                    cand_norms = (final_cand_vecs - v_min[indices_j]) / range_vals[indices_j]

                    w = corr_weights[indices_j]
                    w_norm = w / (w.sum() + 1e-10)

                    diffs = query_norm - cand_norms
                    dists = np.sqrt(np.sum(w_norm * (diffs ** 2), axis=1))

                    cand_values = values[final_search_rows, col_idx]
                    all_dists.append(dists)
                    all_vals.append(cand_values)

                # mean of the top-k from the global pool
                if all_dists:
                    pool_dists = np.concatenate(all_dists)
                    pool_vals = np.concatenate(all_vals)

                    valid_mask = ~np.isnan(pool_vals)
                    pool_dists = pool_dists[valid_mask]
                    pool_vals = pool_vals[valid_mask]

                    if len(pool_vals) > 0:
                        n_select = min(self.k, len(pool_dists))
                        if n_select < len(pool_dists):
                            top_k_idx = np.argpartition(pool_dists, n_select)[:n_select]
                        else:
                            top_k_idx = np.arange(n_select)
                        imputed_val = float(np.mean(pool_vals[top_k_idx]))
                        X.iloc[row_idx, col_idx] = imputed_val
                        values[row_idx, col_idx] = imputed_val  # keep array in sync

        return X


class HankelImputer:
    def __init__(self, lag=None, max_lag=100, max_iter=50, tol=1e-3,
                 max_window=400, lam=1e3, rank=3,
                 lag_buffer=None, context_factor=None):
        """
        Hankel matrix completion (iterative SSA / Cadzow) for block gaps.
        Data are z-scored per column; each gap is completed in a local window.

        Two parametrizations:
          • Relative (used in the study): set lag_buffer and context_factor,
            max_lag=None, max_window=None. For a gap of length L:
                k      = L + lag_buffer
                margin = context_factor * L   (context per side)
            The parameters do not depend on the absolute gap length.
          • Absolute: lag_buffer=None, context_factor=None.
            k = L + max(0.3·L, 10), capped at max_lag;
            margin = max(2k, 2L), capped at max_window // 2.

        lag:            fixed Hankel lag k (overrides lag_buffer / adaptive rule).
        max_lag:        cap for k (None → no cap).
        max_iter:       maximum SSA iterations.
        tol:            convergence tolerance (relative change of the gap values).
        max_window:     cap for the margin (max_window // 2 per side; None → no cap).
        lam:            unused, kept for compatibility.
        rank:           rank of the low-rank approximation.
        lag_buffer:     lag buffer b in samples: k = L + b.
        context_factor: context per side as a multiple of the gap length.

        After impute(), self.diagnostics_ holds one dict per gap: window length,
        k, Hankel dimensions, rank used, iterations, convergence flag and the
        change from the linear initialization.
        """
        self.lag = lag
        self.max_lag = max_lag
        self.max_iter = max_iter
        self.tol = tol
        self.max_window = max_window
        self.lam = lam
        self.rank = rank
        self.lag_buffer = lag_buffer
        self.context_factor = context_factor
        self.diagnostics_ = []

    # ── Hankel construction / reconstruction ─────────────────────────────────

    @staticmethod
    def _build_hankel(values, k):
        n, d = values.shape
        rows = n - k + 1
        idx = np.arange(rows)[:, None] + np.arange(k)[None, :]  # (rows, k)
        return values[idx].transpose(0, 2, 1).reshape(rows, d * k)

    @staticmethod
    def _reconstruct_from_hankel(H, n, d, k):
        rows = H.shape[0]
        result = np.zeros((n, d))
        H_3d = H.reshape(rows, d, k)
        for l in range(k):
            result[l:l + rows, :] += H_3d[:, :, l]
        t = np.arange(n)
        counts = np.minimum(np.minimum(t + 1, k), np.minimum(rows, n - t))
        return result / counts[:, None]

    # ── Solver: iterative SSA (Cadzow) ───────────────────────────────────────

    def _complete_iterative_ssa(self, window_missing, k):
        """
        Iterative Hankel matrix completion: alternate low-rank projection of the
        Hankel matrix and re-insertion of the observed values.
        """
        n, d = window_missing.shape

        # 1. linear interpolation as initial estimate (all-NaN columns → 0)
        df_temp = pd.DataFrame(window_missing)
        X_current = df_temp.interpolate(method='linear', limit_direction='both').bfill().ffill().fillna(0.0).values
        mask_missing = np.isnan(window_missing)
        X_init_missing = X_current[mask_missing].copy()

        info = {"n_iter": 0, "converged": False, "final_rel_change": np.nan,
                "rank_used": np.nan}

        for iteration in range(self.max_iter):
            old_missing_vals = X_current[mask_missing].copy()

            # 2. block-Hankel matrix
            H = self._build_hankel(X_current, k)

            # 3. rank-r approximation (SVD)
            target_rank = min(self.rank, min(H.shape) - 1)
            target_rank = max(1, target_rank)

            try:
                # randomized SVD for large matrices (memory, speed); seed varies per iteration
                if min(H.shape) > 100 and target_rank < min(H.shape) - 1:
                    from sklearn.utils.extmath import randomized_svd
                    U_k, s_k, Vt_k = randomized_svd(H, n_components=target_rank, random_state=42 + iteration)
                else:
                    U, s, Vt = np.linalg.svd(H, full_matrices=False)
                    actual_rank = min(self.rank, len(s))

                    U_k = U[:, :actual_rank]
                    s_k = s[:actual_rank]
                    Vt_k = Vt[:actual_rank, :]
                    del U, s, Vt
            except Exception as e:
                import scipy.linalg
                U, s, Vt = scipy.linalg.svd(H, full_matrices=False, lapack_driver='gesvd')
                actual_rank = min(self.rank, len(s))

                U_k = U[:, :actual_rank]
                s_k = s[:actual_rank]
                Vt_k = Vt[:actual_rank, :]
                del U, s, Vt

            info["rank_used"] = len(s_k)
            H_low_rank = (U_k * s_k) @ Vt_k
            del H, U_k, s_k, Vt_k

            # 4. back to a time series (anti-diagonal averaging)
            X_reconstructed = self._reconstruct_from_hankel(H_low_rank, n, d, k)
            del H_low_rank

            # 5. keep the observed values
            X_current = np.where(mask_missing, X_reconstructed, window_missing)
            del X_reconstructed

            # 6. convergence check
            norm_old = np.linalg.norm(old_missing_vals)
            if norm_old == 0:
                norm_old = 1e-10
            diff = np.linalg.norm(X_current[mask_missing] - old_missing_vals) / norm_old
            del old_missing_vals

            info["n_iter"] = iteration + 1
            info["final_rel_change"] = float(diff)
            if diff < self.tol:
                info["converged"] = True
                break

        # change from the linear initialization (z units): RMS and relative norm
        delta = X_current[mask_missing] - X_init_missing
        info["rms_change_from_linear_z"] = float(np.sqrt(np.mean(delta ** 2))) if delta.size else np.nan
        norm_init = np.linalg.norm(X_init_missing)
        info["rel_change_from_linear"] = float(np.linalg.norm(delta) / norm_init) if norm_init > 0 else np.nan

        return X_current, info

    # ── Main imputation ───────────────────────────────────────────────────────

    def impute(self, X_missing):
        values = X_missing.values.copy().astype(float)
        n, d = values.shape

        # z-score per column, so large-valued sensors do not dominate
        col_means = np.nanmean(values, axis=0)
        col_stds = np.nanstd(values, axis=0)
        col_stds[col_stds == 0] = 1.0  # constant columns

        values = (values - col_means) / col_stds

        result = values.copy()

        self.diagnostics_ = []

        any_missing = np.isnan(values).any(axis=1)
        if not any_missing.any():
            return X_missing.copy()

        diff = np.diff(any_missing.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        gap_sizes = ends - starts
        order = np.argsort(gap_sizes)

        for idx in order:
            st, en = starts[idx], ends[idx]
            gap_len = en - st

            if self.lag is not None:
                k = self.lag
            elif self.lag_buffer is not None:
                k = gap_len + int(self.lag_buffer)
            else:
                k = gap_len + max(int(gap_len * 0.3), 10)

            if self.max_lag is not None:
                k = min(k, self.max_lag)
            k = max(2, k)

            if self.context_factor is not None:
                margin = int(round(self.context_factor * gap_len))
            else:
                margin = max(k * 2, gap_len * 2)
            if self.max_window is not None:
                margin = min(margin, self.max_window // 2)

            w_st = max(0, st - margin)
            w_en = min(n, en + margin)

            window = values[w_st:w_en, :].copy()
            wn, wd = window.shape

            k = min(k, wn // 2)

            if wn - k + 1 < 2:
                self.diagnostics_.append({"gap_start": int(st), "gap_len": int(gap_len),
                                          "skipped": True})
                continue

            gap_rel_st = st - w_st
            gap_rel_en = en - w_st

            other_nans = np.isnan(window).copy()
            other_nans[gap_rel_st:gap_rel_en, :] = False

            if other_nans.any():
                df_prefill = pd.DataFrame(window)
                prefilled = (df_prefill.interpolate(method='linear', limit_direction='both')
                             .bfill().ffill().fillna(0.0).values)
                window[other_nans] = prefilled[other_nans]

            window_imputed, info = self._complete_iterative_ssa(window, k)
            self.diagnostics_.append({
                "gap_start": int(st), "gap_len": int(gap_len), "skipped": False,
                "margin_left": int(st - w_st), "margin_right": int(w_en - en),
                "window_len": int(wn), "k": int(k), "d": int(wd),
                "hankel_rows": int(wn - k + 1), "hankel_cols": int(k * wd),
                **info,
            })

            for j in range(wd):
                col_missing_in_gap = np.isnan(values[st:en, j])
                if col_missing_in_gap.any():
                    abs_idx = st + np.where(col_missing_in_gap)[0]
                    rel_idx = gap_rel_st + np.where(col_missing_in_gap)[0]
                    result[abs_idx, j] = window_imputed[rel_idx, j]
                    values[abs_idx, j] = window_imputed[rel_idx, j]

        # back to physical units
        result = (result * col_stds) + col_means

        df = pd.DataFrame(result, index=X_missing.index, columns=X_missing.columns)
        if df.isna().any().any():
            df = df.interpolate(method='linear', limit_direction='both').bfill().ffill()
        return df


