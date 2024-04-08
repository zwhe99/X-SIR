import os
import json
import random
import argparse
import networkx as nx
from transformers import AutoTokenizer

# set seed
random.seed(0)

# Maximum size of connected components
# Note: The returned connected components are not guaranteed to be smaller than the max_size constraint.
#       We only use max_size to determine the number of clusters.
CC_MAX_SIZE=250

def main():
    parser = argparse.ArgumentParser(description='Generate mappings.')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--dictionary', type=str, required=True, help='Dictionary path. One line per entry.')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab = tokenizer.get_vocab()

    # Load dictionary as edges
    edges = []
    with open(args.dictionary) as f:
        for line in f:
            src_token, tgt_token = line.split()
            edges.append((src_token, tgt_token))

    # Add self-loop for each token
    for token in vocab:
        edges.append((token, token))

    # Build graph & find connected components
    graph = nx.Graph(edges)
    connected_components_node = list(nx.connected_components(graph)) # list of nodes
    connected_components_node.sort(key=lambda x: len(x), reverse=True) # sort by size
    connected_components_graph = [graph.subgraph(ccn) for ccn in connected_components_node] # list of graphs

    # Split connected components into clusters
    clusters = []
    for ccg in connected_components_graph:
        if len(ccg) <= CC_MAX_SIZE:
            clusters.append(list(ccg))
        else:
            resolution = 1
            if len(ccg) > 10000: # use higher resolution for extremely large connected components to find smaller clusters
                resolution = 10
            cs = nx.community.louvain_communities(ccg, seed=0, resolution=resolution)
            clusters.extend(cs)
            print(f"Splitted {len(ccg)} nodes into {len(cs)} clusters")

    # Valid clusters: filter tokens that are not in the vocabulary
    valid_clusters = []
    for c in clusters:
        valid_c = set()
        for token in c:
            if token in vocab:
                valid_c.add(token)
        if len(valid_c) > 0:
            valid_clusters.append(list(valid_c))

    # Check if all tokens are included
    all_valid_tokens = []
    for c in valid_clusters:
        all_valid_tokens.extend(c)
    assert len(all_valid_tokens) == len(vocab), f"len(all_valid_tokens)={len(all_valid_tokens)} len(vocab)={len(vocab)}"
    assert len(all_valid_tokens) == tokenizer.vocab_size, f"len(all_valid_tokens)={len(all_valid_tokens)} tokenizer.vocab_size={tokenizer.vocab_size}"

    # Generate mappings
    clustre_mapping = [random.randint(0, 300 - 1) for _ in range(len(valid_clusters))]
    mapping = [None for _ in range(tokenizer.vocab_size)]
    for i, c in enumerate(valid_clusters):
        for token in c:
            token_id = tokenizer.convert_tokens_to_ids(token)
            assert isinstance(token_id, int)
            assert mapping[token_id] is None
            mapping[token_id] = clustre_mapping[i]
    assert all(x is not None for x in mapping)

    # Save mappings
    if os.path.dirname(args.output_file) != '' and (not os.path.exists(os.path.dirname(args.output_file))):
        os.makedirs(os.path.dirname(args.output_file))
    with open(args.output_file, 'w') as f:
        json.dump(mapping, f, indent=4)

    # Save clusters
    output_cluster_file = ".".join(args.output_file.split(".")[:-1]) + "_clusters.json"
    with open(output_cluster_file, "w") as f:
        # sort by size
        valid_clusters.sort(key=lambda x: len(x), reverse=True)
        json.dump(valid_clusters, f, indent=4, ensure_ascii=False)

    # Print statistics
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Number of clusters: {len(valid_clusters)}")
    print(f"Number of clusters with size >= 2: {len([cc for cc in valid_clusters if len(cc) >= 2])}")
    print(f"Number of tokens in the cc with size >= 2: {sum([len(cc) for cc in valid_clusters if len(cc) >= 2])}")
    print(f"Top 20 largest cc: {[len(cc) for cc in sorted(connected_components_graph, key=lambda x: len(x), reverse=True)[:20]]}")
    print(f"Top 20 largest clusters: {[len(cc) for cc in sorted(clusters, key=lambda x: len(x), reverse=True)[:20]]}")
    print(f"Top 20 largest valid clusters: {[len(cc) for cc in sorted(valid_clusters, key=lambda x: len(x), reverse=True)[:20]]}")
    print(f"Vocab coverage: {sum([len(cc) for cc in valid_clusters if len(cc) >= 2]) / tokenizer.vocab_size * 100:.2f}%")

    # Viualize the largest connected component
    # print(f"Visualizing the largest connected component to {output_cluster_file}")
    # net = Network(height="100%", width="100%", notebook=True, cdn_resources="remote")
    # net.from_nx(connected_components_graph[0])
    # net.write_html(".".join(args.output_file.split(".")[:-1]) + "_largest_cc.html")

if __name__ == '__main__':
    main()