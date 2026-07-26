## Week 1 Updates
- [x] Load dataset and verify network topology (NetworkX).
- [x] Train XGBoost tabular baseline (Loaded MAE: 85.76s).
- [x] Draft 1-page Design Document.
- [x] **Revisions applied via Professor Feedback:** 
    - Added RMSE and Empty trip benchmarks.
    - Documented 80/20 split (seed 42) and verified zero leakage across operating episodes.
    - Formulated edge-level prediction head ($\text{MLP}(h_u \parallel h_v \parallel e_{uv})$).
    - Reframed GAT superiority as a formal research hypothesis.
    - Clarified epistemic vs. aleatoric uncertainty and explicitly defined the SAA MILP integration.