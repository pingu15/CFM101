import sys

def weighted_max_bipartite_matching(N, M, A, C):
    """
    Finds the weighted maximum bipartite matching for sectors and stocks.
    
    Args:
    - N: Number of sectors.
    - M: Number of stocks.
    - A: List of length N where A[i] is the number of stocks required by sector i.
    - C: 2D list (N x M) of weights (correlations) between sectors and stocks.
    
    Returns:
    - match: List of tuples (sector, stock) representing the matching.
    - total_weight: Total weight of the matching.
    """
    # Expand the graph: create dummy nodes for each sector demand
    total_sectors = sum(A)
    expanded_C = [[-sys.maxsize] * M for _ in range(total_sectors)]
    
    sector_mapping = []
    index = 0
    for i in range(N):
        for _ in range(A[i]):
            expanded_C[index] = C[i]
            sector_mapping.append(i)  # Map expanded sector to original sector
            index += 1

    # Hungarian algorithm for max-weight matching
    match = [-1] * M  # Stores which sector is assigned to each stock
    sector_label = [0] * total_sectors
    stock_label = [0] * M
    slack = [0] * M
    slack_x = [-1] * M
    parent = [-1] * M
    
    def dfs(x, visited_x, visited_y):
        visited_x[x] = True
        for y in range(M):
            if visited_y[y]:
                continue
            delta = sector_label[x] + stock_label[y] - expanded_C[x][y]
            if delta == 0:  # Tight edge
                visited_y[y] = True
                if match[y] == -1 or dfs(match[y], visited_x, visited_y):
                    match[y] = x
                    return True
            else:  # Update slack
                if slack[y] > delta:
                    slack[y] = delta
                    slack_x[y] = x
        return False

    # Initialize labels
    for x in range(total_sectors):
        sector_label[x] = max(expanded_C[x])

    # Augmenting path search
    for x in range(total_sectors):
        slack = [sys.maxsize] * M
        slack_x = [-1] * M
        while True:
            visited_x = [False] * total_sectors
            visited_y = [False] * M
            if dfs(x, visited_x, visited_y):
                break
            # Update labels
            delta = min(slack[y] for y in range(M) if not visited_y[y])
            for i in range(total_sectors):
                if visited_x[i]:
                    sector_label[i] -= delta
            for y in range(M):
                if visited_y[y]:
                    stock_label[y] += delta
                else:
                    slack[y] -= delta

    # Extract results
    total_weight = 0
    final_match = []
    for y in range(M):
        if match[y] != -1:
            sector_idx = sector_mapping[match[y]]
            final_match.append((sector_idx, y))
            total_weight += C[sector_idx][y]

    return final_match, total_weight

# Example Usage
N = 3  # Number of sectors
M = 5  # Number of stocks
A = [2, 1, 1]  # Sector demands
C = [  # Correlation matrix
    [1, 2, 4, 4, 5],  # Sector 0
    [1, 2, 2, 1, 1],  # Sector 1
    [3, 2, 3, 4, 1]   # Sector 2
]

match, total_weight = weighted_max_bipartite_matching(N, M, A, C)
print("Matching:", match)
print("Total Weight:", total_weight)