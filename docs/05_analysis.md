

At aggressive feature reduction (k=10), methods diverge significantly — LIME is the most isolated with maximum Jaccard of 0.18, which explains its weaker performance. As k increases to 30, methods converge — RFE and SHAP reach 0.82 similarity, and three features (LON, LAT, BASIN_NI) achieve full consensus. Geographic coordinates dominate across all methods, confirming spatial position as the primary predictor of tropical cyclone landfall.

"At k=10, feature selection methods diverge significantly in their choices, with LIME being the most isolated. At k=30, methods converge toward a consensus set dominated by geographic coordinates, suggesting that spatial position is the most robust predictor of landfall regardless of selection strategy."


MLP (left):
SHAP (teal, AUC=0.9392) is the clear winner, visibly above all other curves in the top-left region. PSO is the only method below baseline — consistent with what we saw in the heatmap. The curves are relatively tightly clustered, confirming MLP is moderately stable across methods.
Random Forest (middle):
This is your most dramatic plot. DE (orange, AUC=0.9745) and SHAP (teal, AUC=0.9768) are far above the baseline dashed line — the visual gap is large and immediately obvious. Chi2 (blue) is the worst performer, barely above baseline. This plot alone makes the case for feature selection being highly beneficial for Random Forest.
XGBoost (right):
All curves are tightly bunched together near the top — confirming again that XGBoost barely needs feature selection. The baseline is already competitive and no method dramatically improves it.
One small thing to note — the legend shows different best k values per method (e.g. shap k=50 for XGBoost, shap k=10 for RF). This is correct behaviour since it's pulling the best k per model from best_per_method.csv, but worth being aware of when writing your thesis so you don't accidentally claim "SHAP always selects k=10".