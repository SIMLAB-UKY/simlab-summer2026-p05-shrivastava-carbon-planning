"""
R01 GNN Dataset Generator — Mine Road Network + Cycle Time Observations
Generates: graph_nodes.csv, graph_edges.csv, cycle_times.csv (4,800 obs)
Seed=42, consistent with the rest of the DigiMine data.
"""
import numpy as np
import pandas as pd
import json, os

np.random.seed(42)
OUT = "/home/claude/gnn_data/intern_05_gnn_cycle_time/data"

# ── NODES: shovels, dumps, crusher, intersections ─────────────────────────────
nodes = [
    # id, type, x, y, elevation_m, capacity_tph (shovels only), queue_capacity
    ("S1","shovel",      200, 700,  80, 600, 3),
    ("S2","shovel",      400, 650,  95, 700, 3),
    ("S3","shovel",      600, 680, 110, 550, 3),
    ("S4","shovel",      800, 710, 125, 650, 3),
    ("I1","intersection",300, 500,  70,   0, 0),
    ("I2","intersection",500, 480,  75,   0, 0),
    ("I3","intersection",700, 460,  85,   0, 0),
    ("I4","intersection",450, 300,  50,   0, 0),
    ("CR","crusher",     100, 100,  20, 1800, 4),
    ("D2","waste_dump",  900, 120,  30,    0, 4),
]
df_nodes = pd.DataFrame(nodes, columns=[
    "node_id","node_type","x","y","elevation_m","capacity_tph","queue_capacity"])
# failure rate per node (shovels and crusher can fail)
df_nodes["failure_rate_per_shift"] = df_nodes["node_type"].map(
    {"shovel":0.08,"crusher":0.05,"intersection":0.0,"waste_dump":0.0})
df_nodes.to_csv(f"{OUT}/graph_nodes.csv", index=False)

# ── EDGES: road segments with physical attributes ─────────────────────────────
def dist(a, b):
    na = df_nodes[df_nodes.node_id==a].iloc[0]
    nb = df_nodes[df_nodes.node_id==b].iloc[0]
    return np.sqrt((na.x-nb.x)**2 + (na.y-nb.y)**2) / 100  # scale to km

def gradient(a, b):
    na = df_nodes[df_nodes.node_id==a].iloc[0]
    nb = df_nodes[df_nodes.node_id==b].iloc[0]
    d = dist(a,b)*1000
    return round((nb.elevation_m - na.elevation_m) / d * 100, 2) if d>0 else 0

edge_list = [
    ("S1","I1"),("S2","I1"),("S2","I2"),("S3","I2"),("S3","I3"),("S4","I3"),
    ("I1","I4"),("I2","I4"),("I3","I4"),
    ("I4","CR"),("I3","D2"),("I4","D2"),
]
edges = []
surface_opts = ["good","fair","poor"]
for a,b in edge_list:
    d = round(dist(a,b), 3)
    edges.append({
        "edge_id":       f"{a}_{b}",
        "from_node":     a,
        "to_node":       b,
        "distance_km":   d,
        "gradient_pct":  gradient(a,b),
        "surface_quality": np.random.choice(surface_opts, p=[0.5,0.35,0.15]),
        "speed_limit_kmh": np.random.choice([25,30,35]),
        "lanes":         np.random.choice([1,2], p=[0.3,0.7]),
    })
    # reverse direction (gradient flips)
    edges.append({
        "edge_id":       f"{b}_{a}",
        "from_node":     b,
        "to_node":       a,
        "distance_km":   d,
        "gradient_pct":  -gradient(a,b),
        "surface_quality": edges[-1]["surface_quality"],
        "speed_limit_kmh": edges[-1]["speed_limit_kmh"],
        "lanes":         edges[-1]["lanes"],
    })
df_edges = pd.DataFrame(edges)
df_edges.to_csv(f"{OUT}/graph_edges.csv", index=False)

# ── CYCLE TIME OBSERVATIONS: 4,800 truck trips over the network ────────────────
# Each observation = one truck traversal of a route (sequence of edges) with
# realized travel time influenced by: distance, gradient, surface, congestion,
# payload, node failures.
surface_factor = {"good":1.0, "fair":1.15, "poor":1.35}

# Define routes: shovel -> crusher (loaded) and crusher -> shovel (empty), plus waste routes
routes = {
    "S1_CR": ["S1_I1","I1_I4","I4_CR"], "CR_S1": ["CR_I4","I4_I1","I1_S1"],
    "S2_CR": ["S2_I1","I1_I4","I4_CR"], "CR_S2": ["CR_I4","I4_I1","I1_S2"],
    "S3_CR": ["S3_I2","I2_I4","I4_CR"], "CR_S3": ["CR_I4","I4_I2","I2_S3"],
    "S4_CR": ["S4_I3","I3_I4","I4_CR"], "CR_S4": ["CR_I4","I4_I3","I3_S4"],
    "S3_D2": ["S3_I3","I3_D2"],          "D2_S3": ["D2_I3","I3_S3"],
    "S4_D2": ["S4_I3","I3_D2"],          "D2_S4": ["D2_I3","I3_S4"],
}
edge_lookup = df_edges.set_index("edge_id").to_dict("index")

N_OBS = 4800
rows = []
truck_ids = [f"T{i+1:02d}" for i in range(10)]

for i in range(N_OBS):
    route_name = np.random.choice(list(routes.keys()))
    edge_seq   = routes[route_name]
    loaded     = route_name.split("_")[1] in ("CR","D2")   # heading to dump = loaded
    payload_t  = round(np.clip(np.random.normal(195,15),100,220),1) if loaded else 0.0
    shift_hour = round(np.random.uniform(0,12),2)
    # congestion: peaks mid-shift (hours 4-8), Poisson queue at intersections
    congestion = np.clip(np.random.poisson(2 + 2*np.exp(-((shift_hour-6)**2)/4)),0,8)
    # node failure flag: 6% of observations have an active failure upstream
    failure_active = int(np.random.rand() < 0.06)

    total_time = 0.0
    total_dist = 0.0
    max_grad   = 0.0
    worst_surface = 1.0
    for eid in edge_seq:
        e = edge_lookup[eid]
        base_speed = e["speed_limit_kmh"] * (0.75 if loaded else 0.95)
        # gradient penalty: +2% time per 1% uphill gradient when loaded
        grad_pen = 1 + max(0, e["gradient_pct"]) * (0.02 if loaded else 0.008)
        surf = surface_factor[e["surface_quality"]]
        seg_time = (e["distance_km"] / base_speed) * 60 * grad_pen * surf
        total_time += seg_time
        total_dist += e["distance_km"]
        max_grad = max(max_grad, abs(e["gradient_pct"]))
        worst_surface = max(worst_surface, surf)

    # congestion delay at intersections: 0.8 min per queue unit
    total_time += congestion * 0.8
    # failure effect: rerouting/waiting adds 15-40% when failure active
    if failure_active:
        total_time *= np.random.uniform(1.15, 1.40)
    # measurement noise
    total_time *= np.random.normal(1.0, 0.06)

    rows.append({
        "obs_id":          i+1,
        "truck_id":        np.random.choice(truck_ids),
        "route":           route_name,
        "origin":          route_name.split("_")[0],
        "destination":     route_name.split("_")[1],
        "loaded":          int(loaded),
        "payload_t":       payload_t,
        "total_distance_km": round(total_dist,3),
        "max_gradient_pct":  round(max_grad,2),
        "surface_factor":    round(worst_surface,2),
        "shift_hour":        shift_hour,
        "congestion_level":  congestion,
        "failure_active":    failure_active,
        "travel_time_min":   round(max(total_time, 1.0), 2),
    })

df_obs = pd.DataFrame(rows)
df_obs.to_csv(f"{OUT}/cycle_times.csv", index=False)

# ── README for the dataset ─────────────────────────────────────────────────────
readme = """# R01 GNN Dataset — Mine Haulage Network

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
"""
with open(f"{OUT}/DATA_README.md","w") as f:
    f.write(readme)

print(f"Nodes: {len(df_nodes)}  |  Edges: {len(df_edges)}  |  Observations: {len(df_obs):,}")
print(f"\nCycle time stats:")
print(df_obs['travel_time_min'].describe().round(2).to_string())
print(f"\nLoaded vs empty mean travel time:")
print(df_obs.groupby('loaded')['travel_time_min'].mean().round(2).to_string())
print(f"\nFailure effect:")
print(df_obs.groupby('failure_active')['travel_time_min'].mean().round(2).to_string())
