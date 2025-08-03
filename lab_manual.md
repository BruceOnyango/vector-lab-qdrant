# MIT Vector Database Lab Manual (Qdrant)

## 🧠 Overview

This hands-on lab introduces students to **Vector Databases** using **Qdrant**, an open-source, high-performance engine for vector similarity search. You'll learn how to:

* Set up Qdrant with Docker
* Use sentence-transformers to embed text
* Perform CRUD operations with semantic vectors
* Execute advanced queries with filters and thresholds
* Apply vector DBs to a real-world semantic search use case

Estimated time: **45–60 minutes**
Skill level: **Beginner to Intermediate**

---

## ⚙️ 1. Setup Instructions (Docker)

### Prerequisites

* Docker installed
* Python 3.8+
* Git

### Step 1: Clone the Lab Repository

```bash
git clone https://github.com/BruceOnyango/vector-lab-qdrant.git
cd vector-lab-qdrant
```
### If using wsl or linux use 2.1
### Step 2.1: Start Qdrant via Docker(linux/wsl users only)

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.9.1
```
### If using windows powershell use 2.2
```bash
Note powershell and cmd( commandline )are not the same thing

```
### Step 2.2: Start Qdrant via Docker(windows powershell users only)

```bash
docker run -p 6333:6333 -p 6334:6334 `
  -v ${PWD}/qdrant_storage:/qdrant/storage `
  qdrant/qdrant:v1.9.1
```

> 🧪 REST API → `http://localhost:6333`

### Step 3.1: Install Python Requirements Linux users only

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3.2: Install Python Requirements Windows (Powershell) users only

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ 2. CRUD Operations + Sample Data

Run the script:

```bash
python main.py
```

This script:

* Initializes the collection
* Embeds 5 movie plot summaries
* Stores them with their genre metadata
* Performs semantic search, filtering, updates, and deletion

### Sample Data

| ID | Plot                                              | Genre   |
| -- | ------------------------------------------------- | ------- |
| 0  | A young wizard attends a magic school.            | Fantasy |
| 1  | A hobbit sets out to destroy a powerful ring.     | Fantasy |
| 2  | A group of superheroes save the world.            | Action  |
| 3  | A space crew travels through a wormhole.          | Sci-Fi  |
| 4  | A detective solves crimes with a genius roommate. | Mystery |

---

## 🧩 3. Applied Scenario: Semantic Movie Search

### Problem

Users of a movie recommendation engine often describe movies vaguely (e.g., "wizard school", "superhero teamwork"). Traditional keyword search fails here.

### Vector DB Solution

* Embed each movie plot using sentence transformers
* Represent user queries as high-dimensional vectors
* Use cosine similarity to find closest plot vectors

---

## 💻 4. Code Snippets & Advanced Features

### 🔍 Basic Similarity Search

```python
results = client.search(
    collection_name="movies",
    query_vector=model.encode("magic and school").tolist(),
    limit=3
)
```

### 🎯 Metadata Filtering

```python
Filter(must=[FieldCondition(key="genre", match=MatchValue(value="Fantasy"))])
```

### 🔗 Multi-Vector Query

```python
avg_vec = np.mean(model.encode(["magic", "teamwork"]), axis=0)
```

### ✅ Update Payload

```python
client.set_payload(payload={"updated": True}, points=[0])
```

### 🗑️ Delete Point

```python
client.delete(points_selector={"points": [4]})
```

### 🧮 Similarity Thresholding

```python
if hit.score > 0.85:
    print(hit.payload["plot"])
```

---

## 📸 5. Visuals and Outputs

Provide screenshots for:

* Collection creation
* Basic query output
* Filtered results
* Multi-query results
* Metadata update confirmation

Store in `./screenshots/`

---

## 🧼 6. Clarity and Reproducibility

* Clear repo layout
* Commented code
* All dependencies in `requirements.txt`
* Optional large dataset support (CSV)

---

## 🤝 7. Group Collaboration Summary

| Member         | Contribution                                  |
| -------------- | --------------------------------------------- |
| Alice Kimani   | Docker setup, data model explanation          |
| Brian Otieno   | CRUD operations, embedding logic              |
| Carol Mwangi   | Markdown formatting, visuals, filtering logic |
| Daniel Njoroge | Multi-query logic, optional dataset ingestion |

---

## 📂 Repository Structure

```
vector-lab-qdrant/
├── main.py
├── lab_manual.md
├── requirements.txt
├── data/
│   └── movie_plots.csv  (optional)
├── screenshots/
│   ├── collection.png
│   ├── results.png
└── qdrant_storage/ (generated)
```

---

## ✅ To Submit

* Push all files to GitHub
* Ensure `lab_manual.md`, `main.py`, `requirements.txt`, and screenshots are included
