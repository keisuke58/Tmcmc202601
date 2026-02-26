import json
import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(json_path, output_path):
    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create graph
    G = nx.Graph()

    # Add nodes with attributes
    for node in data["nodes"]:
        G.add_node(node["id"], label=node["short"], color=node["color"], full_name=node["name"])

    # Add active edges
    for edge in data["active_edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            type="active",
            color="black",
            style="solid",
            weight=2,
            label=f"Idx:{edge['param_idx']}",
        )

    # Add locked edges (optional, maybe dotted lines to show what's removed)
    for edge in data["locked_edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            type="locked",
            color="gray",
            style="dashed",
            weight=1,
            label=f"Locked:{edge['param_idx']}",
        )

    # Layout
    pos = nx.circular_layout(G)  # Circular layout is good for interaction networks

    plt.figure(figsize=(10, 8))

    # Draw nodes
    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)

    # Draw active edges
    active_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "active"]
    nx.draw_networkx_edges(
        G, pos, edgelist=active_edges, width=2, edge_color="black", style="solid"
    )

    # Draw locked edges
    locked_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "locked"]
    nx.draw_networkx_edges(
        G, pos, edgelist=locked_edges, width=1, edge_color="gray", style="dashed"
    )

    # Draw labels
    node_labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_weight="bold")

    # Draw edge labels (active only for clarity, or minimal info)
    # edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True) if d['type'] == 'active'}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(
        "5-Species Interaction Network (Nishioka Algorithm)\nSolid: Active / Dashed: Locked (Zero)",
        fontsize=15,
    )
    plt.axis("off")

    # Add legend manually
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="black", lw=2, label="Active Interaction"),
        Line2D([0], [0], color="gray", lw=1, ls="--", label="Locked to 0 (No Interaction)"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Graph visualization saved to {output_path}")


if __name__ == "__main__":
    json_path = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/interaction_graph.json"
    output_path = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/interaction_network.png"
    visualize_graph(json_path, output_path)
