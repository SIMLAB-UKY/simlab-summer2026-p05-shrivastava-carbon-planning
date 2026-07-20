# R01 GNN Dataset — Mine Haulage Network

## Files

### graph_nodes.csv (10 nodes)
Mine locations: 4 shovels (S1-S4), 3 intersections (I1-I3 + I4), 1 crusher (CR), 1 waste dump (D2).
Columns: node_id, node_type, x, y, elevation_m, capacity_tph, queue_capacity, failure_rate_per_shift

### graph_edges.csv (24 directed edges)
Road segments connecting nodes. Both directions included (gradient flips sign).
Columns: edge_id, from_node, to_node, distance_km, gradient_pct, surface_quality, speed_limit_kmh, lanes

### cycle_times.csv (4,800 observations)
Individual truck traversals of routes through the network.
Columns:
- obs_id, truck_id, route, origin, destination
- loaded (1=hauling ore/waste, 0=returning empty)
- payload_t (tonnes; 0 when empty)
- total_distance_km, max_gradient_pct, surface_factor (route aggregates)
- shift_hour (0-12), congestion_level (0-8, queue units at intersections)
- failure_active (1 = an upstream node failure was active during this trip)
- travel_time_min  <- PREDICTION TARGET

## Ground-truth generative structure (for your reference AFTER modelling)
Travel time = sum over edges of (distance/speed) * gradient_penalty * surface_factor
            + congestion * 0.8 min
            * failure_multiplier (1.15-1.40 when failure_active=1)
            * noise N(1.0, 0.06)

## Suggested first steps
1. Load nodes+edges into NetworkX; verify the graph is connected.
2. Baseline: XGBoost on tabular features (ignore graph structure).
3. GNN: PyTorch Geometric — node features + edge features, predict travel_time_min.
4. The failure_active flag is your handle for failure-aware training (Week 4).

Generated with seed=42. Regenerate: python generate_gnn_data.py
