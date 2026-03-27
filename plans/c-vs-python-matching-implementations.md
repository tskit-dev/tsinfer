# Divergence Analysis: Python vs C `AncestorMatcher`

## 1. Traceback allelic_state traversal — SEGFAULT RISK

**C (line 1812):**
```c
v = u;
while (allelic_state[v] == TSK_NULL) {
    v = parent[v];  // parent was memset to 0xff (NULL_NODE = -1)
}
```

**Python (line 544-546):**
```python
v = u
while self.allelic_state[v] == -1:
    v = self.parent[v]
```

**Divergence:** The Python traceback rebuilds the full tree (parent, left_child, right_child, left_sib, right_sib) via `insert_edge`/`remove_edge` (lines 523-530). The C traceback only reconstructs `parent` (lines 1776, 1786, 1791) — it does a raw `parent[edge.child] = edge.parent` without using `insert_edge`.

This means the C traceback's `parent` array is correct. BUT: if the allelic_state traversal reaches a node whose parent is `NULL_NODE` (-1), the C code indexes `parent[-1]` → **segfault**. The Python code would raise an IndexError. This can happen if the mutation is on the root node (node 0) but `u` starts at a non-root node that doesn't have a mutation on the path to root.

**Specifically:** at line 1824, the upward walk for recombination stops at node 0 (`while u != 0`). But at line 1812, the allelic_state walk has no such guard — it walks `v = parent[v]` until it finds a mutation. If no mutation marks are found before reaching a node with `parent[v] == NULL_NODE`, it reads `allelic_state[-1]` which is out of bounds.

## 2. Traceback tree reconstruction order — SEMANTIC DIVERGENCE

**Python (lines 523-530):**
```python
# Reverse edges: REMOVE edges from left_index (insertion order), INSERT from right_index
while k >= 0 and Il[k].left == pos:    # left_index, traverse backwards
    self.remove_edge(Il[k])             # REMOVE
    k -= 1
while j >= 0 and Ir[j].right == pos:   # right_index, traverse backwards
    self.insert_edge(Ir[j])             # INSERT
    j -= 1
```

**C (lines 1783-1791):**
```c
// Uses swapped variable names: 'in' = right_index_edges, 'out' = left_index_edges
while (out_index >= 0 && out[out_index].left == pos) {   // left_index
    parent[edge.child] = NULL_NODE;                       // REMOVE (set to -1)
    out_index--;
}
while (in_index >= 0 && in[in_index].right == pos) {     // right_index
    parent[edge.child] = edge.parent;                     // INSERT
    in_index--;
}
```

These match — both remove from left_index and insert from right_index during reverse traversal. The naming is confusing (`in`/`out` are swapped vs the forward pass) but the logic is equivalent.

## 3. Forward pass edge removal: decompression divergence

**C (lines 2002-2023):** When removing an edge during the forward pass, if the child has `NULL_LIKELIHOOD` (compressed), the C code decompresses by traversing up through the L_cache to find the parent's likelihood. It then sets `L[edge.child] = L_child` and adds it to `likelihood_nodes`.

**Concern:** The traversal at line 2004-2006 walks `u = parent[u]` looking for a node with either `L[u] != NULL_LIKELIHOOD` or `L_cache[u] != CACHE_UNSET`. If the edge has just been removed, `parent[edge.child]` is already set to `NULL_NODE`. But the code uses `edge.parent` at line 2003 to start the walk — not `parent[edge.child]`. This is correct because the edge was just removed.

However: the walk at line 2006 uses `parent[u]` where `u` starts at `edge.parent`. If `edge.parent` is a root node, `parent[edge.parent] == NULL_NODE`. The walk would then read `L[NULL_NODE]` = `L[-1]` → **segfault**.

**Python handles this differently:** it checks `self.likelihood[u]` inline and uses Python dict lookups.

## 4. Forward pass initial tree loading bounds

**C (line 1893):**
```c
while (in_index < M && out_index < M && in[in_index].left <= start_pos) {
```

**Python (line 303):** Similar loop but the condition differs slightly in how it handles the edge cases when `start_pos == 0`.

## 5. `num_nodes` in transition probability

**C (line 1484):**
```c
const double n = (double) self->matcher_indexes->num_nodes;
```

**Python (test_lshmm.py line 371):**
```python
n = self.num_nodes
```

Both use the total number of nodes in the tree sequence builder (not the current tree's node count). This is consistent.

## 6. Traceback edge output — site index vs position

**Python (lines 560-563):** Uses site INDEX for edge boundaries:
```python
output_edge.left = site_index        # site index
output_edge = Edge(right=site_index, parent=u)  # site index
```
Then converts to positions at lines 583-584.

**C (lines 1831-1835):** Uses site POSITION directly:
```c
path_left[path_length] = sites_position[site];    // position
path_right[path_length] = sites_position[site];   // position
```

This is fine as long as `sites_position[site]` is valid, but skips the intermediate index representation.

## 7. `parent` conversion — Python subtracts 1

**Python (line 588):**
```python
parent -= 1
```

The Python reference works with 1-indexed nodes internally (from the MatcherIndexes) and converts to 0-indexed at the end. The C code works with the same MatcherIndexes but may or may not need this offset. If the C code doesn't do this subtraction, all parent IDs are off by 1.

## Summary: Most Likely Segfault Sources

1. **Allelic state traversal in traceback (line 1812):** No guard against `parent[v] == NULL_NODE`. If the walk reaches the root without finding a mutation, it indexes `allelic_state[-1]`.

2. **L_cache traversal during edge removal (line 2004-2006):** Walking `parent[u]` can reach `NULL_NODE` if starting from a root node, causing `L[-1]` access.

3. **Recombination traversal in traceback (line 1824-1826):** The `assert(u != NULL_NODE)` at line 1826 would catch this in debug builds but is stripped in release. If `parent[u]` reaches `NULL_NODE` before reaching node 0, it reads `recombination_required[-1]`.

## Files

- `lib/ancestor_matcher.c` — C implementation (critical lines: 1508-1510, 1812-1814, 2004-2006, 1824-1826)
- `tests/test_lshmm.py` — Python reference (lines 115-593)
- `tests/algorithm.py` — Alternative Python reference (lines 719-1158)
