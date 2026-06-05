# Knowledge Graph Measures

This report lists the global graph measures used for the Neo4j-based video knowledge graph.

## Assumptions

- The graph is **directed**.
- Relationships are **unweighted**.
- The measures are computed on the **entire graph**.
- If a metric is undefined on a disconnected directed graph, the standard network-science convention used here is:
  - use the **largest strongly connected component (GSCC)** for exact shortest-path metrics like average shortest path length and diameter
  - use all ordered node pairs and count unreachable pairs as `0` contribution for **global efficiency**

## GDS Setup

Some measures require the **Graph Data Science (GDS)** plugin. These graph projections are in-memory only and do not modify stored data.

Check if GDS is installed:

```cypher
RETURN gds.version() AS gds_version;
```

Project the full directed graph:

```cypher
MATCH (source)
OPTIONAL MATCH (source)-[r]->(target)
WITH gds.graph.project(
  'global_metrics_graph',
  source,
  target
) AS g
RETURN g.graphName, g.nodeCount, g.relationshipCount;
```

Project the largest strongly connected component:

```cypher
CALL gds.scc.stream('global_metrics_graph')
YIELD nodeId, componentId
WITH componentId, collect(nodeId) AS nodeIds
ORDER BY size(nodeIds) DESC
LIMIT 1
WITH [id IN nodeIds | gds.util.asNode(id)] AS gsccNodes
UNWIND gsccNodes AS source
OPTIONAL MATCH (source)-[r]->(target)
WHERE target IN gsccNodes
WITH gds.graph.project(
  'global_metrics_gscc',
  source,
  target
) AS g
RETURN g.graphName, g.nodeCount, g.relationshipCount;
```

---

## 1. Number of Nodes

**Definition**  
Total number of nodes in the graph.

**Formula**  
\[
n = |V|
\]

**Interpretation**  
- Higher value means more entities are represented in the graph.
- In this project, nodes may include characters, objects, locations, actions, emotions, and topics.

**Cypher**

```cypher
MATCH (n)
RETURN count(n) AS node_count;
```

**GDS**  
Not required.

---

## 2. Number of Relationships

**Definition**  
Total number of directed relationships in the graph.

**Formula**  
\[
m = |E|
\]

**Interpretation**  
- Higher value means more connections between entities.
- Because the graph is directed, `A -> B` and `B -> A` count as two different relationships.

**Cypher**

```cypher
MATCH ()-[r]->()
RETURN count(r) AS relationship_count;
```

**GDS**  
Not required.

---

## 3. Edge Density for a Directed Graph

**Definition**  
Fraction of all possible directed edges that actually exist.

**Standard formula**  
For a simple directed graph without self-loops:

\[
D = \frac{m}{n(n-1)}
\]

**Interpretation**  
- Near `0`: sparse graph
- Near `1`: almost every node points to almost every other node
- Video knowledge graphs are usually expected to be sparse.

**Cypher**

```cypher
MATCH (n)
WITH count(n) AS n
MATCH ()-[r]->()
WHERE startNode(r) <> endNode(r)
WITH n, count(r) AS m
RETURN
  n,
  m,
  CASE
    WHEN n < 2 THEN 0.0
    ELSE toFloat(m) / (n * (n - 1))
  END AS edge_density_directed;
```

**GDS**  
Not required.

---

## 4. Average Degree

**Definition**  
Average number of incident relationships per node.

**Standard formula for directed graphs**

- Average total degree:
\[
\bar{k} = \frac{2m}{n}
\]
- Average in-degree:
\[
\bar{k}_{in} = \frac{m}{n}
\]
- Average out-degree:
\[
\bar{k}_{out} = \frac{m}{n}
\]

**Interpretation**  
- Higher value means nodes are more connected on average.
- In directed graphs, average in-degree and out-degree are always equal.

**Cypher**

```cypher
MATCH (n)
WITH count(n) AS n
MATCH ()-[r]->()
WITH n, count(r) AS m
RETURN
  CASE WHEN n = 0 THEN 0.0 ELSE 2.0 * m / n END AS average_total_degree,
  CASE WHEN n = 0 THEN 0.0 ELSE 1.0 * m / n END AS average_in_degree,
  CASE WHEN n = 0 THEN 0.0 ELSE 1.0 * m / n END AS average_out_degree;
```

**GDS**  
Not required.

---

## 5. Number of Weakly Connected Components

**Definition**  
Number of maximal groups of nodes that are connected when direction is ignored.

**Interpretation**  
- `1` means the entire graph is weakly connected.
- Larger values mean the graph breaks into disconnected regions.

**Cypher**  
No practical exact built-in Cypher algorithm.

**GDS**

```cypher
CALL gds.wcc.stats('global_metrics_graph')
YIELD componentCount
RETURN componentCount AS weakly_connected_components;
```

**Directionality assumption**  
Weak connectivity ignores direction by definition.

---

## 6. Size of the Largest Weakly Connected Component

**Definition**  
Number of nodes in the largest weakly connected component.

**Interpretation**  
- Large value means most nodes are part of one broad connected region.
- Small value means the graph is fragmented.

**Cypher**  
No practical exact built-in Cypher algorithm.

**GDS**

```cypher
CALL gds.wcc.stats('global_metrics_graph')
YIELD componentDistribution
RETURN componentDistribution.max AS largest_wcc_size;
```

---

## 7. Ratio of Largest Weakly Connected Component Size to Total Graph Size

**Definition**  
Fraction of nodes that belong to the largest weakly connected component.

**Formula**

\[
\frac{|LWCC|}{n}
\]

**Interpretation**  
- Near `1`: almost all nodes are in the same weakly connected region
- Near `0`: graph is highly fragmented

**Cypher**  
No practical exact built-in Cypher algorithm.

**GDS**

```cypher
CALL gds.wcc.stats('global_metrics_graph')
YIELD componentDistribution
MATCH (n)
RETURN
  componentDistribution.max AS largest_wcc_size,
  count(n) AS total_nodes,
  CASE
    WHEN count(n) = 0 THEN 0.0
    ELSE toFloat(componentDistribution.max) / count(n)
  END AS largest_wcc_ratio;
```

---

## 8. Number of Strongly Connected Components

**Definition**  
Number of maximal groups of nodes where every node can reach every other node following edge direction.

**Interpretation**  
- `1` means the entire graph is strongly connected.
- Large values usually mean one-way structure and directional fragmentation.

**Cypher**  
No practical exact built-in Cypher algorithm.

**GDS**

```cypher
CALL gds.scc.stats('global_metrics_graph')
YIELD componentCount
RETURN componentCount AS strongly_connected_components;
```

**Directionality assumption**  
Strong connectivity respects direction exactly.

---

## 9. Size of the Largest Strongly Connected Component

**Definition**  
Number of nodes in the largest strongly connected component.

**Interpretation**  
- Large value means a large part of the graph is mutually reachable.
- Small value means directed reachability is limited.

**Cypher**  
No practical exact built-in Cypher algorithm.

**GDS**

```cypher
CALL gds.scc.stats('global_metrics_graph')
YIELD componentDistribution
RETURN componentDistribution.max AS largest_scc_size;
```

---

## 10. Ratio of Largest Strongly Connected Component Size to Total Graph Size

**Definition**  
Fraction of nodes that belong to the largest strongly connected component.

**Formula**

\[
\frac{|LSCC|}{n}
\]

**Interpretation**  
- Near `1`: most nodes are mutually reachable
- Near `0`: the graph is directionally fragmented

**Cypher**  
No practical exact built-in Cypher algorithm.

**GDS**

```cypher
CALL gds.scc.stats('global_metrics_graph')
YIELD componentDistribution
MATCH (n)
RETURN
  componentDistribution.max AS largest_scc_size,
  count(n) AS total_nodes,
  CASE
    WHEN count(n) = 0 THEN 0.0
    ELSE toFloat(componentDistribution.max) / count(n)
  END AS largest_scc_ratio;
```

---

## 11. Average Shortest Path Length

**Definition**  
Average shortest directed path length between all ordered pairs of nodes in the chosen connected set.

**Standard definition used here**  
For a disconnected directed graph, this is computed on the **largest strongly connected component**.

\[
L = \frac{1}{n(n-1)} \sum_{i \ne j} d(i,j)
\]

where `d(i,j)` is the shortest directed path length.

**Interpretation**  
- Lower value means information or connectivity can move across the graph in fewer steps.
- Higher value means the graph is more spread out.

**Cypher**  
Exact all-pairs shortest paths are not production-friendly in plain Cypher.

**GDS**

```cypher
CALL gds.allShortestPaths.stream('global_metrics_gscc')
YIELD sourceNodeId, targetNodeId, distance
WHERE sourceNodeId <> targetNodeId
  AND gds.util.isFinite(distance)
RETURN avg(distance) AS average_shortest_path_length;
```

**Approximate Cypher fallback**

```cypher
MATCH (s)
WITH s ORDER BY rand() LIMIT 100
MATCH (t)
WITH s, t
WHERE s <> t AND rand() < 0.01
CALL {
  WITH s, t
  OPTIONAL MATCH p = shortestPath((s)-[*]->(t))
  RETURN CASE WHEN p IS NULL THEN NULL ELSE length(p) END AS d
}
WITH d
WHERE d IS NOT NULL
RETURN avg(d) AS approx_average_shortest_path_length;
```

**Directionality assumption**  
Shortest paths follow edge direction.

---

## 12. Graph Diameter

**Definition**  
Maximum finite shortest-path distance between any ordered pair of nodes.

**Standard definition used here**  
Computed on the **largest strongly connected component** for a directed disconnected graph.

\[
\text{diameter} = \max_{i \ne j} d(i,j)
\]

**Interpretation**  
- Small value means even the most distant reachable nodes are still relatively close.
- Large value means the graph has long directed chains.

**Cypher**  
Exact computation is not production-friendly in plain Cypher.

**GDS**

```cypher
CALL gds.allShortestPaths.stream('global_metrics_gscc')
YIELD sourceNodeId, targetNodeId, distance
WHERE sourceNodeId <> targetNodeId
  AND gds.util.isFinite(distance)
RETURN max(distance) AS graph_diameter;
```

**Approximate Cypher fallback**

```cypher
MATCH (s)
WITH s ORDER BY rand() LIMIT 100
MATCH (t)
WITH s, t
WHERE s <> t AND rand() < 0.01
CALL {
  WITH s, t
  OPTIONAL MATCH p = shortestPath((s)-[*]->(t))
  RETURN CASE WHEN p IS NULL THEN NULL ELSE length(p) END AS d
}
WITH d
WHERE d IS NOT NULL
RETURN max(d) AS approximate_diameter_lower_bound;
```

---

## 13. Global Efficiency

**Definition**  
Average inverse shortest-path distance over all ordered node pairs.

**Standard formula**

\[
E = \frac{1}{n(n-1)} \sum_{i \ne j} \frac{1}{d(i,j)}
\]

Unreachable pairs contribute `0`.

**Interpretation**  
- Higher value means the graph is globally efficient: nodes are reachable in fewer directed steps.
- Lower value means the graph is harder to traverse directionally.

**Cypher**  
Exact all-pairs evaluation is not practical in plain Cypher.

**GDS**

```cypher
MATCH (n)
WITH count(n) AS n
CALL gds.allShortestPaths.stream('global_metrics_graph')
YIELD sourceNodeId, targetNodeId, distance
WITH n, sourceNodeId, targetNodeId, distance
WHERE sourceNodeId <> targetNodeId
  AND gds.util.isFinite(distance)
RETURN
  CASE
    WHEN n < 2 THEN 0.0
    ELSE sum(1.0 / distance) / (n * (n - 1))
  END AS global_efficiency;
```

**Directionality assumption**  
Uses directed shortest paths over ordered pairs.

---

## 14. Reciprocity

**Definition**  
Fraction of directed relationships that have a reverse relationship.

**Standard formula**

\[
R = \frac{\text{number of directed edges that have a reverse edge}}{m}
\]

**Interpretation**  
- `0`: no edge has a reverse counterpart
- `1`: every directed edge is reciprocated
- Higher reciprocity means more mutual relationships

**Cypher**

```cypher
MATCH (a)-[r]->(b)
WITH
  count(r) AS total_edges,
  sum(
    CASE
      WHEN EXISTS { MATCH (b)-[]->(a) }
      THEN 1
      ELSE 0
    END
  ) AS reciprocated_directed_edges
RETURN
  total_edges,
  reciprocated_directed_edges,
  CASE
    WHEN total_edges = 0 THEN 0.0
    ELSE toFloat(reciprocated_directed_edges) / total_edges
  END AS reciprocity;
```

**GDS**  
Not required.

---

## Notes on Cypher vs GDS

- **Direct Cypher is enough** for:
  - number of nodes
  - number of relationships
  - edge density
  - average degree
  - reciprocity

- **GDS is recommended** for:
  - weakly connected components
  - strongly connected components
  - average shortest path length
  - diameter
  - global efficiency

Reason: these require full-graph traversal algorithms that Cypher does not provide efficiently as built-in global metrics.

## If GDS Is Not Installed

Neo4j Desktop:

1. Open the database in Neo4j Desktop.
2. Open the `Plugins` tab.
3. Install `Graph Data Science`.
4. Restart the database.

Official docs:

- Cypher concepts: https://neo4j.com/docs/cypher-manual/current/queries/concepts/
- GDS installation: https://neo4j.com/docs/graph-data-science/current/installation/
- Neo4j Desktop GDS install: https://neo4j.com/docs/graph-data-science/current/installation/neo4j-desktop/
