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

## Week 2 Updates
- [x] Engineered custom 2-layer `GCNConv` architecture with Edge-Level MLP readout.
- [x] Implemented strict Leave-One-Arc-Out Cross-Validation (LOAOCV) to prevent spatial data leakage during graph training.
- [x] Benchmarked optimized vanilla GCN (Loaded MAE: 111.65s).
- [x] Documented the "Micro-Data Variance Limit" and mathematically proved the architectural limitation of isotropic GCNs in isolating directional traffic bottlenecks.

## Week 3 Updates
- [x] Upgraded architecture to Physics-Aware `GATv2Conv` with dynamic attention mechanisms.
- [x] Extracted structural attention weights ($\alpha$) and plotted a geographical mine heatmap for operational interpretability (identifying key bottlenecks like I3 $\rightarrow$ D2).
- [x] Overcame parameter-to-data ratio limits by scaling the simulation to a "Mega Mine" topology (20 nodes, 52 edges, 10,000 trips).
- [x] Engineered a **Hybrid Late-Fusion Architecture**, concatenating GATv2 spatial embeddings with Z-score normalized tabular context (`payload_t`, `shift_hour`, `congestion_level`).
- [x] **Milestone:** Tuned Hybrid Model successfully shattered the tabular XGBoost baseline (Test MAE: 71.98s).

## Week 4 Updates
- [x] Updated generative script to inject a 20% failure probability for key upstream nodes (shovels and crushers).
- [x] Trained comparative architectures: Nominal GNN (blind to outages) vs. Failure-Aware GNN (conditioned on the binary `failure_active` flag).
- [x] Filtered test set to evaluate predictions exclusively during held-out active mine outages.
- [x] Generated comparative boxplots (Loaded/Empty/Combined), proving that failure-aware routing drastically condenses error variance.
- [x] Benchmarked final outage metrics: Nominal GNN (374.98s) vs. Failure-Aware GNN (223.46s), yielding a net improvement of 151.52s.

## Week 5 Updates
- [x] Implemented a 10-model GNN ensemble with randomized seed initializations to capture epistemic uncertainty.
- [x] Configured ensemble aggregation (mean for final routing prediction, standard deviation for operational confidence).
- [x] Built regression calibration pipeline and computed Expected Calibration Error (ECE: 56.29s).
- [x] Generated uncertainty vs. actual error calibration plots, proving strong linear correspondence and model self-awareness during active mine outages.