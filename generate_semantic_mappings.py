import json
import random
import argparse
from transformers import AutoTokenizer

# set seed
random.seed(0)

def build_undirected_graph(edges):
    """
    Build an undirected graph.

    Args:
        edges (list): A list containing tuples, where each tuple represents a pair of adjacent nodes.

    Returns:
        dict: A dictionary representing the graph, where the keys are nodes and the values are sets of adjacent nodes.
    """
    graph = {}
    for node1, node2 in edges:
        if node1 not in graph:
            graph[node1] = set()
        if node2 not in graph:
            graph[node2] = set()
        graph[node1].add(node2)
        graph[node2].add(node1)
    return graph

def dfs(graph, start, visited):
    """
    Performs a depth-first search on a graph starting from a given node.
    
    Args:
        graph (dict): The graph represented as a dictionary where the keys are nodes and the values are lists of neighbors.
        start: The starting node for the depth-first search.
        visited (set): A set to keep track of visited nodes.

    Returns:
        set: A set of visited nodes during the depth-first search.
    """
    visited.add(start)
    for neighbour in graph[start]:
        if neighbour not in visited:
            dfs(graph, neighbour, visited)
    return visited

def find_connected_components(graph):
    """
    Finds the connected components in a graph.

    Parameters:
    graph (dict): A dictionary representing the graph, where the keys are the nodes and the values are the adjacent nodes.

    Returns:
    list: A list of sets, where each set represents a connected component in the graph.
    """
    visited = set()
    components = []
    for node in graph.keys():
        if node not in visited:
            component = set()
            dfs(graph, node, component)
            components.append(component)
            visited.update(component)
    return components

def main():
    parser = argparse.ArgumentParser(description='Generate mappings.')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--dictionary', type=str, required=True, help='Chinese-English dictionary path. One line per entry.')
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
    graph = build_undirected_graph(edges)
    connected_components = find_connected_components(graph)

    # Validate connected components: filter tokens that are not in the vocabulary
    valid_connected_components = []
    for cc in connected_components:
        valid_cc = set()
        for token in cc:
            if token in vocab:
                valid_cc.add(token)
        if len(valid_cc) > 0:
            valid_connected_components.append(valid_cc)

    # Validate connected components: check if all tokens are included
    all_valid_tokens = []
    for cc in valid_connected_components:
        all_valid_tokens.extend(cc)
    assert len(all_valid_tokens) == len(vocab)
    assert len(all_valid_tokens) == tokenizer.vocab_size

    # Generate mappings
    clustre_mapping = [random.randint(0, 300 - 1) for _ in range(len(valid_connected_components))]
    mapping = [None for _ in range(tokenizer.vocab_size)]
    for i, cc in enumerate(valid_connected_components):
        for token in cc:
            token_id = tokenizer.convert_tokens_to_ids(token)
            assert isinstance(token_id, int)
            assert mapping[token_id] is None
            mapping[token_id] = clustre_mapping[i]
    assert all(x is not None for x in mapping)

    # Save mappings
    with open(args.output_file, 'w') as f:
        json.dump(mapping, f, indent=4)

    # Save connected components
    output_cluster_file = f"{args.output_file.split('.')[0]}_cluster.json"
    with open(output_cluster_file, "w") as f:
        valid_connected_components_serializable = []
        for cc in valid_connected_components:
            valid_connected_components_serializable.append(list(cc))
        json.dump(valid_connected_components_serializable, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()