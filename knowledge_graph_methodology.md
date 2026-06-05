# Knowledge Graph Methodology

This section describes the flow used to construct the video knowledge graph in Kairos. The graph is built per video. One processed video produces one directed knowledge graph.

## 1. Node Inventory Extraction

During the first phase of Scene-Level Description Generation, the pipeline produces short scene descriptions. These short descriptions are then used to build a reusable video-level node inventory before the longer scene descriptions are generated.

Let the video contain \(N\) scenes, and let the short description for scene \(i\) be \(s_i^{\text{short}}\). The concatenated short-description context is:

\[
S^{\text{short}} = \bigoplus_{i=1}^{N} s_i^{\text{short}}
\]

where \(\bigoplus\) denotes ordered concatenation.

The node inventory is then extracted with an LLM:

\[
\mathcal{V}^{*} = \operatorname{LLM}_{\text{nodes}}\!\left(S^{\text{short}}\right)
\]

where \(\mathcal{V}^{*}\) is the normalized candidate node set for the full video.

The graph uses a fixed node ontology:

\[
\mathcal{C}_V = \{\text{Character}, \text{Object}, \text{Location}, \text{Action}, \text{Emotion}, \text{Topic}\}
\]

Each node \(v \in \mathcal{V}^{*}\) has a category:

\[
c(v) \in \mathcal{C}_V
\]

The purpose of this stage is to create a stable node inventory that can be reused when extracting relationships from later scene descriptions.

## 2. Relationship Extraction After Scene-Level Description

After the long Scene-Level Descriptions are generated, relationship extraction is performed. This happens independently of synopsis generation, although it occurs after scene descriptions are available. The existing video-level node inventory \(\mathcal{V}^{*}\) is used as the node reference set for all relationship extraction passes.

Each relationship is typed and directed:

\[
e = (u, r, v), \quad u,v \in \mathcal{V}, \; r \in \mathcal{C}_E
\]

where \(\mathcal{C}_E\) is the set of allowed relationship types and \(\mathcal{V}\) is the final node set used in the graph and
where all nodes are drawn from the previously generated node inventory.

## 3. Narrative Relationship Extraction

Narrative relationships are extracted per scene from the long scene description.

Let \(s_i^{\text{long}}\) denote the long scene description for scene \(i\). Then the narrative relationship set for scene \(i\) is:

\[
\mathcal{E}^{\text{nar}}_i =
\operatorname{LLM}_{\text{nar}}\!\left(s_i^{\text{long}}, \mathcal{V}^{*}\right)
\]

The full narrative relationship set for the video is:

\[
\mathcal{E}^{\text{nar}} = \bigcup_{i=1}^{N} \mathcal{E}^{\text{nar}}_i
\]

These relationships cover the narrative-semantic part of the graph, such as scene membership, actions, targets, dialogue, emotions, and mentions.

## 4. Spatial Relationship Extraction

Spatial relationships are also extracted per scene, but this pass uses both the long scene description and YOLO-based object context. Only the relevant node categories are provided as context: Character, Object, and Location.

Let \(y_i\) denote the YOLO summary for scene \(i\). Then the spatial relationship set for scene \(i\) is:

\[
\mathcal{E}^{\text{spa}}_i =
\operatorname{LLM}_{\text{spa}}\!\left(s_i^{\text{long}}, y_i, \mathcal{V}^{*}_{\{\text{Character,Object,Location}\}}\right)
\]

The full spatial relationship set for the video is:

\[
\mathcal{E}^{\text{spa}} = \bigcup_{i=1}^{N} \mathcal{E}^{\text{spa}}_i
\]

This pass is used to model location membership and relative spatial structure, such as containment, proximity, left-right relations, front-back relations, and support relations.

## 5. Temporal Interval Relationship Extraction

Temporal interval relationships are extracted with a rolling window rather than from isolated scenes. Each temporal call receives a fixed-size window of scene descriptions with overlap from the previous window. The first scene in each window is used only as context, and temporal relationships are produced only for the remaining output scenes in that window.

Let the temporal window size be \(w\). Then the temporal window beginning at scene \(k\) is:

\[
W_k = \left(s_k^{\text{long}}, s_{k+1}^{\text{long}}, \dots, s_{k+w-1}^{\text{long}}\right)
\]

Let \(\mathcal{A}^{\text{ctx}}_k\) denote the existing action-related relationships already extracted for those same scenes. Then the temporal relationships produced for the output portion of the window are:

\[
\mathcal{E}^{\text{tmp}}_{k} =
\operatorname{LLM}_{\text{tmp}}\!\left(W_k, \mathcal{A}^{\text{ctx}}_k, \mathcal{V}^{*}_{\{\text{Action}\}}\right)
\]

The temporal pass emits only action-to-action interval relations, such as continuation, ordering, and overlap. The global temporal relationship set is:

\[
\mathcal{E}^{\text{tmp}} = \bigcup_{k \in \mathcal{K}} \mathcal{E}^{\text{tmp}}_{k}
\]

where \(\mathcal{K}\) is the set of valid rolling-window start indices.

For this system, the temporal configuration used in practice is:

\[
w = 6
\]

with the first scene in the window acting as overlap context only, so each call produces temporal interval relationships for the remaining five scenes.

## 6. Relationship Groups

The full edge set is composed of three groups:

\[
\mathcal{E} =
\mathcal{E}^{\text{nar}}
\;\cup\;
\mathcal{E}^{\text{spa}}
\;\cup\;
\mathcal{E}^{\text{tmp}}
\]

where:

\[
\mathcal{E}^{\text{nar}} = \text{narrative relationships}
\]

\[
\mathcal{E}^{\text{spa}} = \text{spatial relationships}
\]

\[
\mathcal{E}^{\text{tmp}} = \text{temporal interval relationships}
\]

## 7. Final Graph Per Video

For each processed video, the final knowledge graph is:

\[
G_{\text{video}} = (\mathcal{V}, \mathcal{E})
\]

with:

\[
\mathcal{V} = \mathcal{V}^{*} \cup \mathcal{V}^{\text{scene}}
\]

and:

\[
\mathcal{E} =
\mathcal{E}^{\text{nar}}
\cup
\mathcal{E}^{\text{spa}}
\cup
\mathcal{E}^{\text{tmp}}
\]

Here, \(\mathcal{V}^{\text{scene}}\) includes the scene nodes added to anchor scene-level properties and ordering relations. Therefore, one video corresponds to one directed knowledge graph containing its node inventory, scene nodes, and all extracted relationship groups.
