import numpy as np

def hits_algorithm(adj_matrix, num_iterations):
    n = adj_matrix.shape[0]
    
    auth_scores = np.ones(n)
    hub_scores = np.ones(n)
    
    for i in range(num_iterations):
        auth_scores = np.dot(adj_matrix.T, hub_scores)
        hub_scores = np.dot(adj_matrix, auth_scores)
        
        if auth_scores.max() != 0:
            auth_scores /= auth_scores.max()
        if hub_scores.max() != 0:
            hub_scores /= hub_scores.max()
            
        print(f"Iteration {i+1}:")
        print(f"  Authorities: {auth_scores}")
        print(f"  Hubs:        {hub_scores}")
        print("-" * 20)
        
    return auth_scores, hub_scores

A = np.array([
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
])

num_iters = 5

print("Initial Network:")
print(f"Nodes: {A.shape[0]}")
print(f"Adjacency Matrix:\n{A}\n")

print("Running HITS:")
final_auth, final_hub = hits_algorithm(A, num_iters)

print("\nFinal Normalized Scores:")
print(f"Authorities: {final_auth}")
print(f"Hubs:        {final_hub}")

top_auth_node = np.argmax(final_auth) + 1
top_hub_node = np.argmax(final_hub) + 1
print(f"\nPage {top_auth_node} is the most authoritative page.")
print(f"Page {top_hub_node} is the best hub page.")
