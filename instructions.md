# Network Analysis Assignment

**The Art of Analyzing Big Data — Network Analysis, Visualization, Graph Embeddings, and Link Prediction**

Required tasks: **100 points** · Bonus: **+5 points**

## Point distribution

| Part | Topic | Points |
|------|-------|--------|
| Part A | Movie / Social Network Analysis | 40 |
| Part B | Directed Link Prediction | 25 |
| Part C | Enron Manager Detection and Local LLM Analysis | 35 |
| Bonus | Future Link Prediction | +5 |

> **Note on point totals (inconsistencies in the source PDF):**
> - Part A header reads "35 points" but the distribution table says **40**, and the task breakdown sums to 40 if A1 = 10 (A1.1 5 + A1.2 5). Treat Part A as **40**.
> - A5 header says 5 pts but its sub-steps sum to 7 (2 + 1.5 + 1.5 + 2).
> - "Task C1 — 25 points" header conflicts with its sub-tasks, which sum to **35** (matching the Part C total). Treat Part C as **35**.
> Confirm exact weights with the instructor if grading precision matters.

## General submission instructions

Submit a single **Jupyter Notebook or Markdown report** that includes:

1. Clear answers to all required questions.
2. Well-documented Python code.
3. Visualizations and screenshots where relevant.
4. Short explanations of your methods and results.
5. A short discussion of any limitations, assumptions, or sampling choices you made.

For **each task**, include a section titled **"How I solved this task"** explaining, in your own words:

- What you did.
- Which algorithm, method, or tool you used.
- Why you selected this method.
- What the main result means.

---

# Part A — Movie / Social Network Analysis (40 points)

Dataset: **Movie Dynamics Networks** — https://www.kaggle.com/datasets/michaelfire/movie-dynamics-over-15000-movie-social-networks

Select **one** movie or show network that interests you for Tasks A1–A5, A7. For Task **A1.2**, you will use **multiple** movie networks from the dataset.

## Task A1: Degree Distribution and Graph Embeddings

### A1.1 Degree Distribution — 5 points
For the selected movie/show network, calculate and visualize the degree distribution of the vertices. Include:
- The code used to calculate the degree of each vertex.
- A plot of the degree distribution.
- A short explanation of what the distribution tells you about the network.

### A1.2 Graph Embeddings of Movie Networks — 5 points
Use a graph embedding algorithm to create an embedding for **each** movie/show network. Use one of the following approaches (or another justified method):
- Graph2Vec
- NetLSD
- Weisfeiler-Lehman graph kernel
- Basic graph-level feature vectors (number of nodes, number of edges, density, average clustering coefficient, degree statistics, etc.)
- Any other graph embedding method you can explain clearly.

After creating one embedding vector per graph:
1. Use t-SNE, UMAP, PCA, or another dimensionality reduction method to display all networks in 2D.
2. Visualize the movie/show networks as points in the 2D space.
3. Find interesting groups of movies.
4. Identify several interesting pairs or groups of similar movies.
5. Explain why these movies may be similar according to the network structure.

Include: a description of the graph embedding method, the 2D visualization, an analysis of interesting clusters or similar movies, and a discussion of what the embedding captures and what it may miss.

## Task A2: Top-12 Character Subgraph — 5 points
Choose one centrality algorithm and use it to identify the top 12 characters in the selected network. Create a subgraph containing these 12 characters and draw it using a **circular layout**. Include the centrality algorithm selected, a short explanation of why, and the visualization.

## Task A3: PageRank, Triangles, and Shortest Paths — 5 points
For each vertex in the selected graph, calculate:
- PageRank
- Number of triangles
- Average shortest path length

Include a table or dataframe with the calculated values and a short discussion of the most interesting vertices according to these measures.

## Task A4: Visualization with Cytoscape and Gephi — 5 points
Use Cytoscape and Gephi to visualize the selected network. The visualization should:
- Have vertex size correlated with vertex degree.
- Be readable and visually clear.
- Include screenshots from **both** Cytoscape and Gephi in your submission.

## Task A5: Ego Network Function — 5 points
Write a function that receives a selected vertex and creates a subgraph containing:
- The selected vertex.
- All of its incoming neighbors.
- All of its outgoing neighbors.

Then:
1. Draw the resulting subgraph — 2 points.
2. Calculate the number of vertices in the subgraph — 1.5 points.
3. Calculate the number of edges in the subgraph — 1.5 points.
4. Explain what this ego network reveals about the selected vertex — 2 points.

## Task A6: Large Chess Network — 5 points
Dataset: **Free Internet Chess Server network** — http://dynamics.cs.washington.edu/nobackup/chess/fcis.tar.gz

> **Important:** This network has approximately **429,747,476 edges**. You are not expected to load the full network into memory using standard NetworkX methods.

Find the top 10 most central players in the network and visualize part of the network.

Recommended approaches: **Vaex**, **igraph**, **Polars**, or generate and analyze a meaningful subgraph/sample.

Include: a clear explanation of how you handled the network size, the centrality algorithm used, the top-10 most central players, a visualization of a meaningful part of the network, and a short explanation of the sampling/filtering method (if used).

## Task A7: Lord of the Rings Couples Network — 5 points
Use Cytoscape to draw the Lord of the Rings Couples network discussed in Lecture 2. The visualization should:
- Color vertices according to gender.
- Use vertex shapes to represent character race.
- Include a screenshot of the final visualization.
- Briefly explain the design choices in the visualization.

---

# Part B — Directed Link Prediction (25 points)

## Task B1: Directed Link Prediction Classifier — 25 points
Select a **directed** network and develop a simple link prediction classifier based on the network's directed topology.

You may use one of the following data sources:
- **Reddit community networks** — http://dynamics.cs.washington.edu/data.html
- **The Colorado Index of Complex Networks (ICON)** — https://icon.colorado.edu/
- Another directed network, with approval or a clear explanation of the source.

Deliverables:
1. A short description of the selected network — 2 points.
2. A definition of positive and negative link examples — 3 points.
3. A train/test split strategy — 3 points.
4. A simple baseline classifier based on directed topology — 7 points.
5. An improved classifier using node embeddings or link embedding features — 5 points.
6. Evaluation using appropriate metrics such as AUC, accuracy, precision, recall, or F1 — 3 points.
7. A short comparison between the baseline and the improved model — 2 points.

Examples of possible directed topology features:
- In-degree and out-degree of the source and target nodes.
- Common successors.
- Common predecessors.
- Directed Adamic-Adar style features.
- Directed Jaccard similarity.
- Preferential attachment using in-degree and out-degree.
- Shortest directed path features.

---

# Part C — Enron Manager Detection and Local LLM Analysis (35 points)

> **Local LLM requirement:** The local LLM must run **locally** on your machine or the faculty cluster.

Possible local LLM tools: **Ollama**, **LM Studio**, **llama.cpp**, a local **Hugging Face** model, or any other local model with a clear explanation.

## Task C1: Identifying Managers Using Centrality and Local LLMs

Use the Enron email network and apply three different centrality algorithms to identify managers in the organization. In addition, use a local LLM to help identify which users are actually managers based on their email content.

### C1.1 Network and Label Description — 5 points
Provide a short description of:
- The Enron email network.
- The nodes and edges in the network.
- The available managerial labels, if such labels are available.
- Any preprocessing you performed.

### C1.2 Centrality-Based Manager Detection — 5 points
Apply three different centrality algorithms to identify possible managers. Examples of possible centrality algorithms:
- In-degree centrality
- Out-degree centrality
- Betweenness centrality
- Closeness centrality
- PageRank
- HITS authority score
- HITS hub score
- Eigenvector centrality

Include: the three centrality algorithms you used, the top-10 ranked vertices according to each algorithm, the **precision@10** of each algorithm (precision@10 = number of true managers in the top 10 / 10), and a short comparison between the algorithms.

### C1.3 Local LLM-Based Manager Identification — 10 points
For users that appear likely to be managers, examine their emails and use a local LLM to classify whether each user appears to be a manager. For each analyzed user, provide the local LLM with a sample of emails written by or sent to that user.

Include: the local LLM model used, the prompt you used, how many emails were analyzed per user, the LLM's classification for each user, and a short explanation of whether you agree with the LLM's classification.

Important:
- Do **not** send private or sensitive email text to an external LLM.
- You may summarize or truncate emails before giving them to the local LLM.
- You should explain how you selected the emails for each user.

### C1.4 Summarizing Manager Roles with a Local LLM — 5 points
For each manager you identified, use the local LLM to summarize their likely position or role in the company. The summary should be based only on the email evidence. For each identified manager, include:
- The person's name or node identifier.
- A short local-LLM-based summary of their likely role.
- 2–3 examples of evidence from the emails, such as recurring topics, communication patterns, or people they interact with.
- A short note about the uncertainty of the conclusion.

### C1.5 Network Visualization — 5 points
Create a visualization of the Enron network, or a meaningful subgraph of it, using one of the centrality measures. The visualization should include:
- Node size based on a selected centrality measure.
- A clear layout.
- Highlighting of users identified as likely managers.
- A short explanation of the visualization.

### C1.6 Discussion — 5 points
Discuss the differences between:
- Managers identified by centrality algorithms.
- Managers identified using the local LLM.
- Managers identified in the ground-truth labels, if available.

Your discussion should address: which method worked best, cases where the methods disagreed, possible reasons for mistakes, and limitations of using email content and local LLMs for organizational role detection.

---

# Bonus — Future Link Prediction (+5 points)

Evaluate the classifier's ability to predict future links. Include:
- A temporal train/test split, if the data supports timestamps.
- A comparison between random link prediction and informed future link prediction.
- A short discussion of whether your classifier generalizes to future links.

---

# Optional Practice Questions (not part of the required 100-point submission)

These are for additional practice unless stated otherwise by the instructor.

1. **Connected Components** — Visualize the distribution of the network's strongly connected components and weakly connected components.
2. **Maximal Strongly Connected Component** — Using Cytoscape, visualize the network's maximal strongly connected component, or a meaningful part of it.
3. **Reciprocal Links** — Draw a subgraph of all vertices that have at least one reciprocal link (include every vertex v for which there exists another vertex u such that both links (u, v) and (v, u) exist).
4. **Communities** — Split the network into communities and find the second most central vertex in each community.
5. **Chess Network Extension** — Find the top-10 most central players in the Free Internet Chess Server network and visualize part of it.

---

# General Notes

- Clearly state which libraries and tools you used.
- For large networks, sampling is allowed, but you must justify your sampling method.
- Every visualization should include a title and a short explanation.
- When using external datasets, include a link to the data source.
- Make sure your analysis is reproducible as much as possible.
- For every task, include a short explanation of how you solved it.
- In Part C, you are required to use a local LLM as part of the analysis method.
