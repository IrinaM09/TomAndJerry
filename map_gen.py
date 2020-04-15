# from collections import defaultdict
#
#
# class Graph:
#     def __init__(self):
#         self.graph = defaultdict(list)
#
#     # Add edge to graph
#     def addEdge(self, u, v):
#         self.graph[u].append(v)
#
#     # BFS function to find path from source to sink
#     def BFS(self, s, d):
#         # Base case
#         if s == d:
#             return True
#
#         # Mark all the vertices as not visited
#         visited = [False] * (len(self.graph) + 1)
#
#         # Create a queue for BFS
#         queue = [s]
#
#         visited[s] = True
#         while queue:
#
#             s = queue.pop(0)
#
#             # Get all adjacent vertices
#             for i in self.graph[s]:
#                 # destination
#                 if i == d:
#                     return True
#
#                 # Continue BFS
#                 if not visited[i]:
#                     queue.append(i)
#                     visited[i] = True
#
#         return False
#
#
# def is_safe(i, j, matrix):
#     return True if 0 <= i < len(matrix) and 0 <= j < len(matrix[0]) \
#                    and matrix[i][j] != 'X' else False
#
#
# # Returns true if there is a path
# # from a source to a destination
# def find_path(matrix, source_row, source_col, dest_row, dest_col):
#     s, d = None, None  # source and destination
#     N = len(matrix)
#     M = len(matrix[0])
#     g = Graph()
#
#     k = 1  # current vertex
#     for i in range(N):
#         for j in range(M):
#             if is_safe(i, j + 1, matrix):
#                 g.addEdge(k, k + 1)
#             if is_safe(i, j - 1, matrix):
#                 g.addEdge(k, k - 1)
#             if is_safe(i + 1, j, matrix):
#                 g.addEdge(k, k + N)
#             if is_safe(i - 1, j, matrix):
#                 g.addEdge(k, k - M)
#
#             # source index
#             if i == source_row and j == source_col:
#                 s = k
#
#             # destination index
#             if i == dest_row and j == dest_col:
#                 d = k
#             k += 1
#
#     # find path Using BFS
#     return g.BFS(s, d)


def is_safe(matrix, visited, i, j):
    return not (matrix[i][j] == 'X' or visited[i][j] != 0)


def is_valid(matrix, i, j):
    return 0 <= i < len(matrix) and 0 <= j < len(matrix[0])


def find_path(matrix, visited, i, j, dest_row, dest_col, min_dist, dist):
    if i == dest_row and j == dest_col:
        return min(dist, min_dist)

    visited[i][j] = 'V'

    # down
    if is_valid(matrix, i + 1, j) and is_safe(matrix, visited, i + 1, j):
        min_dist = find_path(matrix, visited, i + 1, j, dest_row, dest_col, min_dist, dist + 1)

    # right
    if is_valid(matrix, i, j + 1) and is_safe(matrix, visited, i, j + 1):
        min_dist = find_path(matrix, visited, i, j + 1, dest_row, dest_col, min_dist, dist + 1)

    # top
    if is_valid(matrix, i - 1, j) and is_safe(matrix, visited, i - 1, j):
        min_dist = find_path(matrix, visited, i - 1, j, dest_row, dest_col, min_dist, dist + 1)

    # left
    if is_valid(matrix, i, j - 1) and is_safe(matrix, visited, i, j - 1):
        min_dist = find_path(matrix, visited, i, j - 1, dest_row, dest_col, min_dist, dist + 1)

    visited[i][j] = 0

    return min_dist
