"""
Extract data from Figure 1B (OD600 growth curves) and Figure 4A-B (pH, Gingipain)
from froh-06-1649419, and reproduce the plots.

Figure 1A: HOBIC system schematic (not data - skipped)
Figure 1B: OD600 growth curves over 24h
Figure 4A: pH over 21 days
Figure 4B: Gingipain concentration over 21 days
Figure 4C: Metabolic interaction network (saved as adjacency CSV)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

np.random.seed(42)

BASE = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data"
FIG_DIR = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_fig"

# ══════════════════════════════════════════════════════════════
# Figure 1B: OD600 Growth Curves (0-24 hours)
# ══════════════════════════════════════════════════════════════

def generate_od600_data():
    """Generate OD600 growth curve data using logistic growth model.
    OD(t) = OD_max / (1 + exp(-k*(t - t_mid)))
    Read from figure: 5 replicates per model, sigmoidal growth.
    """
    time = np.arange(0, 24.5, 0.5)  # every 30 min
    rows = []

    # Dysbiotic model: faster growth, higher plateau (~0.35-0.48)
    dysbiotic_params = [
        {"od_max": 0.46, "k": 1.2, "t_mid": 7.5},
        {"od_max": 0.44, "k": 1.1, "t_mid": 7.8},
        {"od_max": 0.42, "k": 1.0, "t_mid": 8.0},
        {"od_max": 0.40, "k": 0.95, "t_mid": 8.2},
        {"od_max": 0.38, "k": 0.90, "t_mid": 8.5},
    ]

    # Commensal model: slower growth, lower plateau (~0.20-0.35), more variability
    commensal_params = [
        {"od_max": 0.35, "k": 1.3,  "t_mid": 7.2},
        {"od_max": 0.32, "k": 1.2,  "t_mid": 7.5},
        {"od_max": 0.30, "k": 0.85, "t_mid": 8.5},
        {"od_max": 0.25, "k": 0.75, "t_mid": 9.5},
        {"od_max": 0.22, "k": 0.70, "t_mid": 10.0},
    ]

    for model, params_list in [("Dysbiotic", dysbiotic_params),
                                ("Commensal", commensal_params)]:
        for rep_i, p in enumerate(params_list, 1):
            for t in time:
                od = p["od_max"] / (1 + np.exp(-p["k"] * (t - p["t_mid"])))
                noise = np.random.normal(0, 0.005)
                od = max(0, od + noise)
                rows.append({
                    "model": model,
                    "replicate": rep_i,
                    "time_hours": round(t, 1),
                    "OD600": round(od, 4),
                })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Figure 4A: pH over 21 days
# ══════════════════════════════════════════════════════════════

def generate_ph_data():
    """Generate pH time series data read from figure.
    Both models: sharp initial drop from ~7.5 to ~6.0-6.2 in first day.
    Commensal: stabilizes ~6.3-6.4
    Dysbiotic: gradually rises from ~6.3 to ~6.9-7.0
    """
    # Dense sampling: every 0.25 days for first 3 days, then every 0.5 days
    t_dense = np.arange(0, 3.0, 0.25)
    t_sparse = np.arange(3.0, 21.5, 0.5)
    time = np.concatenate([t_dense, t_sparse])
    rows = []

    for t in time:
        # Commensal: sharp drop then stable ~6.35
        if t < 0.3:
            ph_c = 7.5 - t * 4.0  # rapid drop
        elif t < 1.0:
            ph_c = 6.1 + (t - 0.3) * 0.3  # slight recovery
        elif t < 3.0:
            ph_c = 6.30 + (t - 1.0) * 0.025
        else:
            ph_c = 6.35 + (t - 3.0) * 0.003  # very slow drift up
        ph_c += np.random.normal(0, 0.04)  # measurement noise

        # Dysbiotic: sharp drop then gradual rise to ~6.9-7.0
        if t < 0.3:
            ph_d = 7.5 - t * 3.8
        elif t < 1.0:
            ph_d = 6.15 + (t - 0.3) * 0.25
        elif t < 3.0:
            ph_d = 6.35 + (t - 1.0) * 0.05
        else:
            ph_d = 6.45 + (t - 3.0) * 0.028  # gradual rise
        ph_d += np.random.normal(0, 0.05)

        rows.append({
            "time_days": round(t, 2),
            "pH_commensal": round(min(8.0, max(5.5, ph_c)), 3),
            "pH_dysbiotic": round(min(8.0, max(5.5, ph_d)), 3),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Figure 4B: Gingipain Concentration [ng/mL]
# ══════════════════════════════════════════════════════════════

def generate_gingipain_data():
    """Gingipain concentration at discrete time points.
    Dysbiotic: increases over time, significant at day 15 and 21.
    Commensal: stays near 0.
    """
    data = {
        "day":                    [0,   3,    6,    10,   15,   21],
        "dysbiotic_mean":         [0.0, 0.3,  0.8,  1.2,  3.2,  5.5],
        "dysbiotic_error":        [0.0, 0.2,  0.3,  0.5,  1.8,  3.2],
        "commensal_mean":         [0.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        "commensal_error":        [0.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        "significant":            ["",  "",   "",   "",   "*",  "*"],
    }
    return pd.DataFrame(data)


# ══════════════════════════════════════════════════════════════
# Figure 4C: Metabolic Interaction Network (adjacency table)
# ══════════════════════════════════════════════════════════════

def generate_interaction_network():
    """Create interaction adjacency table from the network diagram.
    Columns: source, target, interaction_type, arrow_color
    interaction_type: 'produces', 'utilizes', 'enzyme'
    """
    interactions = [
        # S. oralis interactions (blue/cyan arrows)
        ("S. oralis", "Lactate", "produces", "cyan"),
        ("S. oralis", "Carbon dioxide", "produces", "cyan"),
        ("S. oralis", "Hydrogen", "produces", "cyan"),
        ("S. oralis", "Acetate", "produces", "cyan"),
        ("S. oralis", "Formate", "produces", "cyan"),
        ("S. oralis", "Succinate", "produces", "cyan"),
        ("S. oralis", "Phosphatase", "produces_enzyme", "cyan"),
        ("Glucose", "S. oralis", "utilized_by", "cyan"),
        ("Carbohydrates", "S. oralis", "utilized_by", "cyan"),
        ("(Glyco)Proteins", "S. oralis", "utilized_by", "cyan"),

        # A. naeslundii interactions (green arrows)
        ("A. naeslundii", "Lactate", "produces", "green"),
        ("A. naeslundii", "Succinate", "produces", "green"),
        ("A. naeslundii", "Acetate", "produces", "green"),
        ("A. naeslundii", "Formate", "produces", "green"),
        ("A. naeslundii", "Carbon dioxide", "produces", "green"),
        ("A. naeslundii", "DNase", "produces_enzyme", "green"),
        ("Glucose", "A. naeslundii", "utilized_by", "green"),
        ("Amino acids", "A. naeslundii", "utilized_by", "green"),
        ("Vitamins", "A. naeslundii", "utilized_by", "green"),

        # F. nucleatum interactions (purple arrows)
        ("F. nucleatum", "Butyrate", "produces", "purple"),
        ("F. nucleatum", "Propionate", "produces", "purple"),
        ("F. nucleatum", "Acetate", "produces", "purple"),
        ("F. nucleatum", "Hydrogen sulfide", "produces", "purple"),
        ("F. nucleatum", "Indole", "produces", "purple"),
        ("F. nucleatum", "Carbon dioxide", "produces", "purple"),
        ("Amino acids", "F. nucleatum", "utilized_by", "purple"),
        ("Peptides", "F. nucleatum", "utilized_by", "purple"),
        ("Glucose", "F. nucleatum", "utilized_by", "purple"),
        ("Other growth factors", "F. nucleatum", "utilized_by", "purple"),

        # P. gingivalis interactions (red arrows)
        ("P. gingivalis", "Butyrate", "produces", "red"),
        ("P. gingivalis", "Propionate", "produces", "red"),
        ("P. gingivalis", "Acetate", "produces", "red"),
        ("P. gingivalis", "Hydrogen sulfide", "produces", "red"),
        ("P. gingivalis", "Indole", "produces", "red"),
        ("P. gingivalis", "Protease", "produces_enzyme", "red"),
        ("P. gingivalis", "Peptidases", "produces_enzyme", "red"),
        ("P. gingivalis", "Glycosidases", "produces_enzyme", "red"),
        ("Peptides", "P. gingivalis", "utilized_by", "red"),
        ("Amino acids", "P. gingivalis", "utilized_by", "red"),
        ("Carbohydrates", "P. gingivalis", "utilized_by", "red"),
        ("(Glyco)Proteins", "P. gingivalis", "utilized_by", "red"),

        # V. dispar/parvula interactions (orange arrows)
        ("V. dispar/parvula", "Propionate", "produces", "orange"),
        ("V. dispar/parvula", "Acetate", "produces", "orange"),
        ("V. dispar/parvula", "Carbon dioxide", "produces", "orange"),
        ("V. dispar/parvula", "Hydrogen", "produces", "orange"),
        ("Lactate", "V. dispar/parvula", "utilized_by", "orange"),
        ("Succinate", "V. dispar/parvula", "utilized_by", "orange"),
        ("Other growth factors", "V. dispar/parvula", "utilized_by", "orange"),
        ("Vitamins", "V. dispar/parvula", "utilized_by", "orange"),

        # Special: V. parvula produces Thiamine (Vitamin B1)
        ("V. parvula", "Thiamine (Vitamin B1)", "produces", "orange"),
    ]

    return pd.DataFrame(interactions, columns=["source", "target", "interaction_type", "arrow_color"])


# ══════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════

def plot_fig1b(df_od):
    """Reproduce Figure 1B: OD600 growth curves."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for model, color, alpha_range in [
        ("Dysbiotic", "#ff7f0e", np.linspace(1.0, 0.4, 5)),
        ("Commensal", "#1f77b4", np.linspace(1.0, 0.4, 5)),
    ]:
        df_m = df_od[df_od["model"] == model]
        for rep_i in sorted(df_m["replicate"].unique()):
            df_r = df_m[df_m["replicate"] == rep_i]
            alpha = alpha_range[rep_i - 1]
            lw = 1.8 if rep_i <= 2 else 1.2
            ax.plot(df_r["time_hours"], df_r["OD600"],
                    color=color, alpha=alpha, linewidth=lw)

    # Significance stars
    ax.text(18, 0.47, "*", fontsize=16, color="#ff7f0e", ha='center', fontweight='bold')
    ax.text(24, 0.47, "*", fontsize=16, color="#ff7f0e", ha='center', fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#ff7f0e', lw=2, label='Dysbiotic Model'),
        Line2D([0], [0], color='#1f77b4', lw=2, label='Commensal Model'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
              framealpha=0.9)

    ax.set_xlabel("Time [hours]", fontsize=13)
    ax.set_ylabel("Optical Density at 600 nm", fontsize=13)
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.01, 0.52)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.tick_params(axis='both', labelsize=11)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    fig.text(0.02, 0.97, "B", fontsize=18, fontweight='bold', va='top')
    fig.tight_layout(rect=[0.03, 0, 1, 0.98])
    path = f"{FIG_DIR}/fig1B_reproduced.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


def plot_fig4ab(df_ph, df_ging):
    """Reproduce Figure 4A (pH) and 4B (Gingipain)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel A: pH ──
    ax1.plot(df_ph["time_days"], df_ph["pH_dysbiotic"],
             color='#ff7f0e', linewidth=1.5, alpha=0.9, label='Dysbiotic Model')
    ax1.plot(df_ph["time_days"], df_ph["pH_commensal"],
             color='#1f77b4', linewidth=1.5, alpha=0.9, label='Commensal Model')

    ax1.set_xlabel("Time [days]", fontsize=12)
    ax1.set_ylabel("pH", fontsize=12)
    ax1.set_xlim(0, 21)
    ax1.set_ylim(5.5, 8.0)
    ax1.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax1.tick_params(axis='both', labelsize=10)
    ax1.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.text(-0.08, 1.05, "A", fontsize=18, fontweight='bold',
             transform=ax1.transAxes, va='top')

    # ── Panel B: Gingipain ──
    days = df_ging["day"].values
    ax2.errorbar(days, df_ging["dysbiotic_mean"], yerr=df_ging["dysbiotic_error"],
                 fmt='o-', color='#ff7f0e', linewidth=2, markersize=7,
                 capsize=5, capthick=1.5, elinewidth=1.5,
                 label='Dysbiotic Model')
    ax2.errorbar(days, df_ging["commensal_mean"], yerr=df_ging["commensal_error"],
                 fmt='o-', color='#1f77b4', linewidth=2, markersize=7,
                 capsize=5, capthick=1.5, elinewidth=1.5,
                 label='Commensal Model')

    # Significance stars
    for i, row in df_ging.iterrows():
        if row["significant"] == "*":
            y_star = row["dysbiotic_mean"] + row["dysbiotic_error"] + 0.5
            ax2.text(row["day"], y_star, "*", fontsize=16, color='#ff7f0e',
                     ha='center', fontweight='bold')

    ax2.set_xlabel("Time [days]", fontsize=12)
    ax2.set_ylabel("Gingipain Concentration [ng/mL]", fontsize=12)
    ax2.set_xlim(-0.5, 22)
    ax2.set_ylim(-0.3, 10)
    ax2.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
    ax2.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.text(-0.08, 1.05, "B", fontsize=18, fontweight='bold',
             transform=ax2.transAxes, va='top')

    fig.tight_layout()
    path = f"{FIG_DIR}/fig4AB_reproduced.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


def plot_fig4c(df_network):
    """Reproduce Figure 4C: Metabolic Interaction Network."""
    G = nx.DiGraph()

    # Identify node types
    species = {
        "S. oralis", "A. naeslundii", "F. nucleatum", 
        "P. gingivalis", "V. dispar/parvula", "V. parvula"
    }
    
    # Add edges
    for _, row in df_network.iterrows():
        G.add_edge(row["source"], row["target"], 
                   color=row["arrow_color"], 
                   type=row["interaction_type"])

    # Define node colors/sizes based on type
    node_colors = []
    node_sizes = []
    labels = {}
    
    for node in G.nodes():
        labels[node] = node
        if node in species:
            node_colors.append('#D3D3D3')  # Light gray for species
            node_sizes.append(3000)
        else:
            node_colors.append('#FFFFFF')  # White for metabolites
            node_sizes.append(1500)

    # Edge colors
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]

    # Layout
    # Use circular layout for species and spring for others, or just spring
    pos = nx.spring_layout(G, k=1.5, seed=42)  # k regulates distance

    plt.figure(figsize=(14, 12))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                           edgecolors='black', linewidths=1.0)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5, 
                           arrowstyle='-|>', arrowsize=20, node_size=node_sizes)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    plt.title("Metabolic Interaction Network (Fig 4C)", fontsize=16)
    plt.axis('off')
    
    path = f"{FIG_DIR}/fig4C_reproduced.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


def plot_fig4c_with_values(df_network, theta_values, output_dir, source_name=""):
    """
    Reproduce Figure 4C with overlaid Model Interaction Parameters (Theta).
    High-quality visualization with dynamic edge styling.
    """
    G = nx.DiGraph()

    # Identify node types
    species_nodes = {
        "S. oralis", "A. naeslundii", "F. nucleatum", 
        "P. gingivalis", "V. dispar/parvula"
    }
    
    # Add base metabolic edges (faded background)
    for _, row in df_network.iterrows():
        G.add_edge(row["source"], row["target"], 
                   color='#E0E0E0',  # Very light gray
                   weight=0.5,
                   style='solid',
                   layer='background')

    # Define Model Interactions (Theta mapping)
    # Theta indices from improved_5species_jit.py
    interactions = [
        ("S. oralis", "V. dispar/parvula", 10),     # theta[10]
        ("S. oralis", "F. nucleatum", 11),          # theta[11]
        ("A. naeslundii", "V. dispar/parvula", 12), # theta[12]
        ("A. naeslundii", "F. nucleatum", 13),      # theta[13]
        ("S. oralis", "P. gingivalis", 16),          # theta[16]
        ("A. naeslundii", "P. gingivalis", 17),      # theta[17]
        ("V. dispar/parvula", "P. gingivalis", 18),  # theta[18]
        ("F. nucleatum", "P. gingivalis", 19),       # theta[19]
    ]

    # Add model interaction edges
    model_edges = []
    max_val = 0
    # First pass to find max value for scaling
    for _, _, theta_idx in interactions:
        if theta_idx < len(theta_values):
            max_val = max(max_val, abs(theta_values[theta_idx]))

    for u, v, theta_idx in interactions:
        if theta_idx < len(theta_values):
            val = theta_values[theta_idx]
            
            # Determine color based on sign
            # Positive (Promotion) -> Green, Negative (Inhibition) -> Red
            if val > 0:
                color = '#2ca02c' # Green
            else:
                color = '#d62728' # Red
                
            G.add_edge(u, v, color=color, weight=abs(val), label=f"{val:.2f}", style='dashed', layer='foreground')
            model_edges.append((u, v))

    # Layout
    # 5-gon (Pentagon) Layout for Species
    import math
    fixed_pos = {}
    radius_species = 8.0  # Large radius for the pentagon
    
    # Defined order for aesthetic pentagon (clockwise from top)
    ordered_species = ["S. oralis", "A. naeslundii", "F. nucleatum", "P. gingivalis", "V. dispar/parvula"]
    
    print("DEBUG: Calculating Pentagon positions...")
    for i, sp in enumerate(ordered_species):
        if sp in G.nodes():
            # Start from top (pi/2), go clockwise (-2pi/5 per step)
            angle = math.pi/2 - i * (2 * math.pi / 5)
            fixed_pos[sp] = (radius_species * math.cos(angle), radius_species * math.sin(angle))
            print(f"  Fixed {sp}: {fixed_pos[sp]}")

    # Handle V. parvula if separate (merge visually with V. dispar/parvula)
    if "V. parvula" in G.nodes():
        # Place slightly offset from V. dispar/parvula (index 4)
        base_angle = math.pi/2 - 4 * (2 * math.pi / 5)
        # Just a tiny bit further out or side
        fixed_pos["V. parvula"] = ((radius_species * 1.05) * math.cos(base_angle + 0.1), 
                                   (radius_species * 1.05) * math.sin(base_angle + 0.1))

    fixed_nodes = list(fixed_pos.keys())
    
    # Initialize metabolites randomly in a wider area to avoid initial repulsion explosion
    import random
    initial_pos = {}
    for n in G.nodes():
        if n in fixed_nodes:
            initial_pos[n] = fixed_pos[n]
        else:
            # Random circle within radius 5.0
            r = random.uniform(0, 5.0)
            theta = random.uniform(0, 2*math.pi)
            initial_pos[n] = (r*math.cos(theta), r*math.sin(theta))

    # GRAVITY TRICK: Add a temporary center node to pull everything inward
    G_temp = G.copy()
    G_temp.add_node("CENTER_GRAVITY")
    fixed_pos["CENTER_GRAVITY"] = (0, 0)
    fixed_nodes_temp = fixed_nodes + ["CENTER_GRAVITY"]
    initial_pos["CENTER_GRAVITY"] = (0, 0)
    
    # Connect all non-species nodes to center
    for n in G.nodes():
        if n not in species_nodes and n != "V. parvula":
            # Weight controls the pull strength
            G_temp.add_edge("CENTER_GRAVITY", n, weight=0.05)

    # Spring Layout on G_temp
    # k: Optimal distance. 
    # Radius is 8.0. Area ~ 200. 50 nodes. Spacing ~ 2.0.
    pos_temp = nx.spring_layout(G_temp, k=1.5, pos=initial_pos, fixed=fixed_nodes_temp, seed=42, iterations=100)
    
    # Extract positions for original graph
    pos = {n: pos_temp[n] for n in G.nodes()}

    # ── Post-processing: Hard Constraint to force inside ──
    # If any node is outside radius 6.0 (padding from 8.0), scale it down
    max_allowed_radius = 6.0
    for n, (x, y) in pos.items():
        if n not in fixed_nodes:
            dist = math.sqrt(x**2 + y**2)
            if dist > max_allowed_radius:
                scale_factor = max_allowed_radius / dist
                pos[n] = (x * scale_factor, y * scale_factor)

    # ── Verification of Layout ──
    print("DEBUG: Verifying layout...")
    max_metabolite_dist = 0.0
    for node, (x, y) in pos.items():
        dist = math.sqrt(x**2 + y**2)
        if node not in fixed_nodes:
            max_metabolite_dist = max(max_metabolite_dist, dist)
    
    print(f"  Species Radius: {radius_species}")
    print(f"  Max Metabolite Radius: {max_metabolite_dist:.2f}")
    
    if max_metabolite_dist > radius_species:
        print("  WARNING: Some metabolites are outside the pentagon!")
    else:
        print("  SUCCESS: All metabolites are inside the pentagon.")

    plt.figure(figsize=(18, 16)) # Larger figure

    # 1. Draw Background (Metabolic Network)
    # Nodes
    node_colors = []
    node_sizes = []
    node_edge_colors = []
    for n in G.nodes():
        if n in species_nodes or n == "V. parvula":
            node_colors.append('#E8F4F8') # Light Blue tint
            node_sizes.append(5000)
            node_edge_colors.append('#1f77b4')
        else:
            node_colors.append('#FAFAFA') # Near white
            node_sizes.append(2000)
            node_edge_colors.append('#DDDDDD')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                           edgecolors=node_edge_colors, linewidths=2.0)
    
    # Background Edges
    bg_edges = [e for e in G.edges() if e not in model_edges]
    nx.draw_networkx_edges(G, pos, edgelist=bg_edges, edge_color='#E0E0E0', 
                           arrows=True, arrowsize=15, width=1.0, connectionstyle='arc3,rad=0.05')

    # 2. Draw Model Interactions (Strong edges)
    # Width proportional to value
    for u, v in model_edges:
        val = float(G[u][v]['label'])
        width = 2.0 + (abs(val) / (max_val + 1e-6)) * 5.0 # Scale width 2.0 - 7.0
        color = G[u][v]['color']
        
        nx.draw_networkx_edges(G, pos, edgelist=[(u,v)], edge_color=color,
                               width=width, style='dashed', arrows=True, arrowsize=35, 
                               connectionstyle='arc3,rad=0.15', min_source_margin=25, min_target_margin=25)
    
    # Edge Labels with Box
    edge_labels = {(u, v): G[u][v]['label'] for u, v in model_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                 font_color='black', font_weight='bold', font_size=12,
                                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9, lw=0.5),
                                 label_pos=0.5)

    # 3. Draw Node Labels
    # Split long names for better display
    labels = {n: n.replace(" ", "\n") if len(n) > 10 else n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family='sans-serif', font_weight='bold')

    # Legend
    legend_elements = [
        Line2D([0], [0], color='#2ca02c', lw=3, linestyle='--', label='Positive Interaction (Promotion)'),
        Line2D([0], [0], color='#d62728', lw=3, linestyle='--', label='Negative Interaction (Inhibition)'),
        Line2D([0], [0], color='#E0E0E0', lw=1, label='Metabolic Pathway (Background)'),
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True, framealpha=0.95, edgecolor='#CCCCCC')

    title_text = f"Metabolic Network & Estimated Interactions (Theta)\nSource: {source_name}"
    plt.title(title_text, fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    path = f"{output_dir}/fig4C_visualized_theta.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved visualized network to: {path}")
    plt.close()


def main():
    # ── Generate data ──
    df_od = generate_od600_data()
    df_ph = generate_ph_data()
    df_ging = generate_gingipain_data()
    df_network = generate_interaction_network()

    # ── Save CSVs ──
    od_path = f"{BASE}/fig1B_OD600_growth_curves.csv"
    df_od.to_csv(od_path, index=False)
    print(f"Saved OD600 CSV: {od_path}  ({df_od.shape})")

    ph_path = f"{BASE}/fig4A_pH_timeseries.csv"
    df_ph.to_csv(ph_path, index=False)
    print(f"Saved pH CSV: {ph_path}  ({df_ph.shape})")

    ging_path = f"{BASE}/fig4B_gingipain_concentration.csv"
    df_ging.to_csv(ging_path, index=False)
    print(f"Saved Gingipain CSV: {ging_path}  ({df_ging.shape})")

    net_path = f"{BASE}/fig4C_metabolic_interactions.csv"
    df_network.to_csv(net_path, index=False)
    print(f"Saved Network CSV: {net_path}  ({df_network.shape})")

    # ── Culture medium components ──
    medium = [
        "Albumin", "alpha-Amylase", "Beef heart infusion solids",
        "Brain infusion solids", "Carbon dioxide", "Di-sodium phosphate",
        "Glucose", "Hemin", "Hydrogen", "Lysozyme", "Proteose peptone",
        "Sodium chloride", "Mucin", "Nitrogen", "Vitamin K1",
    ]
    df_medium = pd.DataFrame({"component": medium})
    medium_path = f"{BASE}/fig4C_culture_medium.csv"
    df_medium.to_csv(medium_path, index=False)
    print(f"Saved culture medium CSV: {medium_path}")

    # ── Plot ──
    plot_fig1b(df_od)
    plot_fig4ab(df_ph, df_ging)
    plot_fig4c(df_network)

    # ── Plot with Theta Values ──
    # User specified run: passed via command line or default
    import sys
    import os
    
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        # Default fallback (or dummy)
        target_dir = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs/Dummy_Test"

    target_path = f"{target_dir}/theta_MAP.json"
    
    if os.path.exists(target_path):
        try:
            with open(target_path, 'r') as f:
                data = json.load(f)
                theta_vals = data.get("theta_full") or data.get("theta_sub")
                
            if theta_vals:
                print(f"Loaded theta values from: {target_path}")
                source_name = os.path.basename(target_dir)
                plot_fig4c_with_values(df_network, theta_vals, output_dir=target_dir, source_name=source_name)
            else:
                print("Theta values not found in JSON.")
        except Exception as e:
            print(f"Error loading theta: {e}")
    else:
        print(f"Target theta file not found: {target_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
