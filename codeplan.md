# CCKS2025 Complex Knowledge Reasoning Challenge

## Code Plan v1.0

### Goal

Build a unified reasoning framework for:

* Knowledge Graph QA (100)
* Multi-hop Text QA (100)
* Table QA (50)

Input:

```json
{
  "id": 1,
  "task_type": "...",
  ...
}
```

Output:

```json
{"id":1,"answer":"xxx"}
```

Generate:

```bash
submit.jsonl
```

---

# Stage 0: Environment Setup

## Python

```bash
python>=3.10
```

## Dependencies

```bash
pip install openai
pip install pandas
pip install numpy
pip install rank-bm25
pip install networkx
pip install tqdm
pip install pydantic
pip install rapidfuzz
pip install sentence-transformers
```

Optional:

```bash
pip install vllm
pip install faiss-cpu
```

---

# Stage 1: Project Structure

```text
project/

├── configs/
│   └── config.yaml
│
├── data/
│   └── contest_data.json
│
├── outputs/
│   └── submit.jsonl
│
├── prompts/
│   ├── kg_prompt.txt
│   ├── text_prompt.txt
│   ├── table_prompt.txt
│   └── verify_prompt.txt
│
├── src/
│
│   ├── main.py
│   │
│   ├── router.py
│   │
│   ├── llm/
│   │   └── client.py
│   │
│   ├── utils/
│   │   ├── io.py
│   │   ├── logger.py
│   │   └── metrics.py
│   │
│   ├── solvers/
│   │   ├── kg_solver.py
│   │   ├── text_solver.py
│   │   └── table_solver.py
│   │
│   ├── kg/
│   │   ├── parser.py
│   │   ├── graph_builder.py
│   │   └── graph_reasoner.py
│   │
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── embed.py
│   │   └── reranker.py
│   │
│   ├── table/
│   │   ├── parser.py
│   │   └── executor.py
│   │
│   └── verifier/
│       └── answer_verifier.py
│
└── experiments/
```

---

# Stage 2: Unified Data Schema

Create unified sample object.

```python
class Sample:
    id: int
    task_type: str
    question: str

    input: str | None
    contexts: list | None
    table: dict | None
```

Goal:

Hide dataset differences.

Every solver receives:

```python
solve(sample)
```

---

# Stage 3: Task Router

Implement:

```python
def route(sample):
```

Logic:

```python
if task_type == "knowledge_graph":
    return kg_solver.solve(sample)

if task_type == "multi_hop_qa":
    return text_solver.solve(sample)

if task_type == "table_qa":
    return table_solver.solve(sample)
```

---

# Stage 4: LLM Layer

Create unified interface.

```python
answer = llm.generate(prompt)
```

Support:

* GPT
* Qwen
* DeepSeek
* Local vLLM

Config driven.

Example:

```yaml
model:
  provider: openai
  model_name: qwen-max
```

---

# Stage 5: Knowledge Graph Module

## Objective

Convert triples into graph.

Input:

```text
(A, spouse, B)
(B, born_in, C)
```

Output:

```python
graph[A] = [
    ("spouse", B)
]
```

---

## Components

### parser.py

Parse triples.

```python
triples = [
    (h,r,t)
]
```

### graph_builder.py

Build graph.

Use:

```python
networkx.MultiDiGraph
```

### graph_reasoner.py

Implement:

```python
BFS
DFS
Path Search
```

Support:

* 1-hop
* 2-hop
* 3-hop

reasoning.

---

## KG Solver Workflow

```text
Question
    ↓
Entity Detection
    ↓
Graph Search
    ↓
Candidate Answer
    ↓
LLM Refinement
    ↓
Final Answer
```

---

# Stage 6: Multi-Hop Text QA

## Objective

Avoid sending all paragraphs directly.

Use retrieval first.

---

## Step 1

Build paragraph corpus.

```python
paragraphs = [...]
```

---

## Step 2

BM25 retrieval.

Retrieve:

```python
TopK=5
```

---

## Step 3

Construct reasoning context.

```python
context =
top paragraphs
```

---

## Step 4

LLM reasoning.

Prompt:

```text
Use ONLY the provided evidence.

Question:
...

Evidence:
...

Answer:
```

---

## Future Upgrade

Add:

```text
Dense Retrieval

BGE-large
E5-large
```

Hybrid retrieval.

---

# Stage 7: Table QA

## Objective

Avoid pure LLM reasoning.

Use program execution.

---

## Table Parser

Convert:

```json
{
  "header": [...],
  "rows": [...]
}
```

to

```python
DataFrame
```

---

## Question Classifier

Classify into:

### Aggregation

```text
max
min
avg
sum
```

### Count

```text
how many
count
```

### Compare

```text
greater than
less than
```

### Lookup

```text
who
which
```

---

## Executor

Run directly:

```python
df.max()
df.min()
df.sum()
```

instead of LLM.

Fallback:

```python
LLM
```

---

# Stage 8: Evidence Verification

## Motivation

Handle K-Stress.

Avoid hallucination.

---

Prompt:

```text
Question:
...

Evidence:
...

Answer:
...

Is answer fully supported?

Return:
YES
NO
```

If:

```text
NO
```

Return:

```text
Unknown
```

---

# Stage 9: Self Consistency

Run 3 reasoning styles.

```python
answer1
answer2
answer3
```

Voting:

```python
Counter
```

Return majority answer.

Applicable only for:

* Text QA
* Complex KG QA

---

# Stage 10: Submission Generator

Generate:

```jsonl
{"id":1,"answer":"..."}
{"id":2,"answer":"..."}
```

Validation:

* no missing ids
* answer is string
* preserve ordering

---

# Stage 11: Offline Evaluation

Implement:

```python
evaluate.py
```

Metrics:

### Exact Match

EM

### F1

Token-level F1

### ROUGE-L

Final:

```python
score =
(EM+F1+ROUGE)/3
```

---

# Stage 12: Logging

For each sample save:

```json
{
  "id":1,
  "question":"...",
  "evidence":"...",
  "prediction":"..."
}
```

Benefits:

Easy error analysis.

---

# Stage 13: Error Mining

Create:

```bash
experiments/error_analysis.ipynb
```

Analyze:

* KG path errors
* retrieval failures
* table parsing failures
* hallucinations

Generate statistics.

---

# Milestone

## M1

Data loading

Router

Submission generation

Expected time:

0.5 day

---

## M2

KG baseline

Text baseline

Table baseline

Expected time:

1 day

---

## M3

BM25 retrieval

Graph reasoning

Table executor

Expected time:

1 day

---

## M4

Verification

Self consistency

Expected time:

1 day

---

## M5

Error-driven optimization

Expected time:

continuous

---

# Expected Performance

Baseline:

```text
LLM Only
```

Expected:

50~65%

---

Hybrid Reasoning:

```text
KG Search
+
BM25
+
Table Executor
```

Expected:

65~80%

---

Advanced Version:

```text
Hybrid Retrieval
+
Verifier
+
Self Consistency
```

Expected:

Top leaderboard competitive solution.
