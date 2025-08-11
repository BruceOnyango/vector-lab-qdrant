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

### Prerequisites and versions used for this tutorial

* Docker installed (27.0.3)- check version using the command 
  ```bash 
  docker --version
  ```
* Python 3.8+ - check version using the following command
```bash
python --version   # or python3 --version
```

* Git (2.46.0)- check version using the following command
```bash
git --version
```



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
    limit=3 #shows only three results maximum
)
```
### 🔍 Basic Similarity Search — Deep Dive

#### 📌 What is Similarity Search?
Similarity search in a vector database finds **items whose embeddings are closest** to a given query vector.  
Instead of relying on keyword matching, it measures **semantic closeness** between high-dimensional vectors using metrics like cosine similarity or Euclidean distance.

In Qdrant, the `.search()` method performs this operation and returns the top matches, optionally with their metadata.

---

#### 📚 Why We Can Call `client.search()` — OOP Principles in Action

The `client` object is an **instance** of the `QdrantClient` class.  
Because `QdrantClient` is implemented using **Object-Oriented Programming (OOP)** principles, we can access its methods (like `.search()`) directly.

Here’s how the main OOP principles apply:

1. **Encapsulation**  
   - The connection details (`host`, `port`, authentication) are stored **inside** the `client` object.  
   - We don’t have to manually manage HTTP requests — `.search()` internally handles them.

2. **Abstraction**  
   - Instead of writing raw API calls, we use a **simple Python method**.  
   - The complexity of:
     - Forming HTTP payloads  
     - Sending requests  
     - Parsing JSON responses  
     - Handling errors  
     is hidden from the user.

3. **Inheritance**  
   - The `QdrantClient` class **inherits shared functionality** (e.g., HTTP communication, serialization) from a **base client class** in the Qdrant library.  
   - This prevents rewriting the same logic for every database operation.

4. **Polymorphism**  
   - Methods like `.search()` can accept different input types:
     - Python lists
     - NumPy arrays
     - Other iterable formats  
   - This flexibility allows the same method to work with multiple data representations without changing the code.



### 🎯 Metadata Filtering

```python

Filter(must=[FieldCondition(key="genre", match=MatchValue(value="Fantasy"))])
```

### 🎯 Metadata Filtering — Deep Dive

#### 📌 What is Metadata in a Vector Database?
In a vector database like **Qdrant**, **metadata** (also called *payload*) refers to **non-vector information attached to a point**.  
While the **vector** is used for similarity search, metadata provides **structured context** that allows filtering or faceting results.

**Example:**

| Data Type   | Example Value                                                       | Purpose                                   |
|-------------|---------------------------------------------------------------------|-------------------------------------------|
| **Vector**  | `[-0.012, 0.233, ..., 0.832]`                                       | Encodes the meaning of the document/plot |
| **Metadata**| `{"genre": "Fantasy", "release_year": 2001, "rating": 8.5}`         | Adds structured fields for filtering     |

With metadata, you can:
- Restrict search results (e.g., only `genre = Fantasy`)
- Combine **semantic search + structured filtering**
- Enable **faceted search** and analytics

---

#### 📌 Filtering Syntax in Qdrant

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

query_filter = Filter(
    must=[
        FieldCondition(
            key="genre",
            match=MatchValue(value="Fantasy")
        )
    ]
)
```

### 🔗 Multi-Vector Query

```python
avg_vec = np.mean(model.encode(["magic", "teamwork"]), axis=0)
```
### 🔗 Multi-Vector Query — Deep Dive

#### 📌 What is a Multi-Vector Query?
In a vector database, a **multi-vector query** allows you to search using more than one concept at a time by **combining their embeddings** into a single query vector.  
This is useful when:
- A single term is too vague.
- You want to **blend multiple semantic concepts**.
- You need **results that capture the intersection** of multiple ideas.

---

#### 📌 Example Syntax

```python
import numpy as np

# Encode two different concepts into vectors
magic_vector = model.encode(["magic"])
teamwork_vector = model.encode(["teamwork"])

# Average the vectors to create a blended representation
avg_vec = np.mean(model.encode(["magic", "teamwork"]), axis=0)
```

### ✅ Update Payload

```python
client.set_payload(payload={"updated": True}, points=[0])
```

### ✅ Update Payload — Deep Dive

#### 📌 What is a Payload Update?
In Qdrant, the **payload** (metadata) is the additional structured information attached to a point alongside its vector.  
An **update payload** operation modifies this metadata **without changing the vector itself**.  
This is useful for:
- Tagging results after indexing.
- Marking items as updated, verified, or inactive.
- Adding new attributes after the vector is already stored.


### 🗑️ Delete Point

```python
client.delete(points_selector={"points": [4]})
```

### 🗑️ Delete Point — Deep Dive

#### 📌 What is Point Deletion in Qdrant?
In Qdrant, a **point** represents a single record consisting of:
- A **vector** (embedding for similarity search)
- **Payload/metadata** (structured fields like genre, year, etc.)
- A **unique ID** (used to identify and manage the point)

The `.delete()` operation **removes** these records entirely from the collection.  
Once deleted:
- The vector is removed from the index.
- The payload is permanently deleted.
- The point ID becomes available for reuse.


### 🧮 Similarity Thresholding

```python
if hit.score > 0.85:
    print(hit.payload["plot"])
```

### 🧮 Similarity Thresholding — Deep Dive

#### 📌 What is a Similarity Threshold?
A **similarity threshold** is a **cutoff value** used to decide whether a retrieved result is *similar enough* to the query to be considered relevant.  
In vector databases like Qdrant, this value is based on the **similarity score** between:
- The **query vector** (representation of the user’s search)
- The **stored point vector** (representation of an item in the database)

If the score is **above** the threshold → we keep the result.  
If it’s **below** → we discard it.

---

#### 📌 Example Syntax

```python
if hit.score > 0.85:
    print(hit.payload["plot"])
```
#### 📚 How We Get From Text to a Threshold Score

1. **Tokenization**  
   - The input text (e.g., `"magic and school"`) is split into smaller units called **tokens**.  
   - These tokens can be:
     - Words → `"magic"`, `"school"`
     - Subwords → `"ma"`, `"##gic"` (common in BERT-based models)
     - Special tokens like `[CLS]` or `[SEP]` added by the model.
   - Example:  
     ```
     "magic and school" → ["magic", "and", "school"]
     ```

2. **Embedding Generation**  
   The goal of embedding generation is to turn human-readable text into **numerical vectors** that preserve meaning so they can be compared mathematically.

   Historically and in modern NLP, this has evolved in three main stages:

   **a) Bag of Words (BoW)**  
   - **How it works:**  
     1. Create a vocabulary of all unique words in the dataset.  
     2. Represent each sentence as a vector where each dimension counts how many times each word appears.  
     - Example: Vocabulary = `["magic", "school", "teamwork"]`  
       ```
       "magic and school" → [1, 1, 0]
       ```
   - **Limitations:**  
     - Ignores word order.  
     - Treats all words as unrelated — "magic" and "wizard" have no relationship in BoW space.  
     - Vectors grow huge for large vocabularies.

   **b) Word2Vec and Similar Embedding Models**  
   - **How it works:**  
     - Words are represented as **dense, low-dimensional vectors** (e.g., 300 dimensions) learned from large text corpora.  
     - Words with similar meanings have vectors close together in this space.  
     - Example: `"magic"` and `"wizard"` will be near each other in vector space.  
     - Sentence vectors can be created by **averaging word vectors**.  
   - **Benefit:** Captures some semantic relationships between words.  
   - **Limitation:** Context-independent — `"bank"` in “river bank” and “money bank” has the same vector.

   **c) Modern Contextual Embeddings (e.g., BERT, Sentence Transformers)**  
   - **How it works:**  
     - Uses **transformer models** to create **contextual embeddings** where the meaning of a word depends on its surrounding words.  
     - For `"magic and school"`, the vector for `"school"` will be different in the sentence `"magic and school"` vs `"mathematics at school"`.  
     - Produces a **single sentence embedding** (e.g., 768 dimensions for BERT) that captures the overall meaning.  
       Example:
       ```
       "magic and school" → [0.21, -0.55, 0.89, ..., -0.12]
       ```
   - **Benefit:** Handles polysemy (multiple meanings), word order, and richer semantics.
   - **Why it matters for similarity search:**  
     - Vectors from modern embeddings are much better at grouping semantically similar texts together, even if they have different words.

---

**Summary Flow:**  
Text → Tokens → Word-level vectors (BoW or Word2Vec) → Sentence vector (average or transformer pooling) → Stored in vector database for similarity comparison.


3. **Storage in the Vector Database**  
   - This vector is stored in Qdrant along with its **payload** (metadata like title, genre, plot).

4. **Embedding the Query**  
   - When searching, the query text is **processed the same way**:
     - Tokenized
     - Embedded into a vector using the same model
     - Example: `"magic"` → `[0.19, -0.53, 0.87, ..., -0.10]`

5. **Similarity Calculation**  
   - Qdrant compares the query vector to stored vectors using a **distance metric**:
     - **Cosine Similarity** (most common in NLP):
       \[
       \text{similarity} = \frac{A \cdot B}{||A|| \times ||B||}
       \]
       where \( A \) is the query vector and \( B \) is a stored vector.
     - Result range: `-1` (opposite meaning) to `1` (identical meaning).
   - Example:  
     ```
     similarity("magic", "magic and school") = 0.86
     ```

## 📐 Cosine Similarity — Deep Dive

### 📌 What is Cosine Similarity?
Cosine similarity measures **how close two vectors point in the same direction**, regardless of their length.  
It is based on the **angle** between the vectors in a multi-dimensional space.
This is ideal for text embeddings, where the direction encodes meaning.

#### 🔢 Formula
\[
\text{similarity}(A, B) = \frac{A \cdot B}{||A|| \times ||B||}
\]

Where:
- \( A \cdot B \) → **Dot product** of vectors \( A \) and \( B \).
- \( ||A|| \) → Magnitude of vector \( A \).
- \( ||B|| \) → Magnitude of vector \( B \).

---

#### 🧮 Step-by-Step Example
Given:
A = [0.2, 0.5, 0.7]
B = [0.3, 0.45, 0.8]

1. **Dot product (A · B):**
(0.2 × 0.3) + (0.5 × 0.45) + (0.7 × 0.8)
= 0.06 + 0.225 + 0.56
= 0.845

2. **Magnitude of A (||A||):**

sqrt(0.2² + 0.5² + 0.7²)
= sqrt(0.04 + 0.25 + 0.49)
= sqrt(0.78) ≈ 0.883

3. **Magnitude of B (||B||):**

sqrt(0.3² + 0.45² + 0.8²)
= sqrt(0.09 + 0.2025 + 0.64)
= sqrt(0.9325) ≈ 0.965


4. **Cosine similarity:**
0.845 / (0.883 × 0.965) ≈ 0.992


**Interpretation:**  
A score of `0.992` means the vectors are almost identical in direction.

---

#### 📊 Range
- **1.0** → Perfect similarity (same direction)
- **0.0** → No similarity (orthogonal)
- **-1.0** → Opposite meaning (opposite direction)

---

#### ✅ Advantages
- Ignores magnitude — works well for semantic embeddings.
- Effective for text and NLP-based searches.
- Computationally efficient for high dimensions.

#### ⚠️ Disadvantages
- Magnitude information is lost.
- Requires normalized, high-quality embeddings.

---

### 2️⃣ Euclidean Distance

#### 📌 What is Euclidean Distance?
Euclidean distance measures the **straight-line distance** between two vectors in space.  
Unlike cosine similarity, it considers **both magnitude and direction**

---

#### 🔢 Formula
\[
\text{distance}(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}
\]

Where:
- \( A_i \) and \( B_i \) are the components of vectors \( A \) and \( B \) in the \( i^{th} \) dimension.

---

#### 🧮 Step-by-Step Example
Using the same vectors:
A = [0.2, 0.5, 0.7]
B = [0.3, 0.45, 0.8]


1. **Component-wise differences:**
(0.2 - 0.3)² = 0.01
(0.5 - 0.45)² = 0.0025
(0.7 - 0.8)² = 0.01


2. **Sum of squares:**
0.01 + 0.0025 + 0.01 = 0.0225


3. **Square root:**
sqrt(0.0225) = 0.15


**Interpretation:**  
A smaller Euclidean distance indicates greater similarity.

---

#### 📊 Range
- **0** → Identical vectors.
- **Higher values** → Greater dissimilarity.

---

#### ✅ Advantages
- Takes both direction and magnitude into account.
- Easy to understand geometrically.
- Works well when absolute differences matter.

#### ⚠️ Disadvantages
- Sensitive to magnitude — longer vectors can appear less similar even if direction is the same.
- Can be less meaningful for very high-dimensional embeddings without normalization.

---

### 3️⃣ Which to Use in Qdrant?
- **Cosine Similarity** → Best for **semantic search**, NLP tasks, and when vector magnitude is not important.  
- **Euclidean Distance** → Best when magnitude differences matter (e.g., in sensor data or physical measurements).

---

**Example in Qdrant with Cosine Similarity:**
```python
results = client.search(
    collection_name="movies",
    query_vector=model.encode("magic").tolist(),
    limit=3
)
```
**Example in Qdrant with Euclidean Distance:**

```python
results = client.search(
    collection_name="movies",
    query_vector=model.encode("magic").tolist(),
    limit=3,
    search_params={"hnsw_ef": 128, "exact": False, "metric_type": "euclid"}
)
```


6. **Threshold Comparison**  
   - The computed result's similarity score is compared to a **predefined threshold**:
     - If `score >= threshold` → Keep the result.
     - If `score < threshold` → Discard the result.
   - Example:
     ```
     Threshold = 0.85
     Score = 0.86 → Keep result
     Score = 0.79 → Discard result
     ```


**Key Insight:**  
The threshold is not arbitrary — it is a **semantic closeness cutoff** derived from how the embedding model transforms raw text into numerical vectors and how the chosen similarity metric measures their relationship.

**Summary similarity threshold Flow**
Text → Tokenization → Embedding (BoW / Word2Vec / Transformer) → Store in DB
Query Text → Tokenization → Embedding
Stored Vector ↔ Query Vector → Cosine Similarity → Compare to Threshold → Keep/Discard




## 📸 5. Visuals and Outputs

Provided screenshots for:

* Filtered results
* Multi-query results

Stored in `./screenshots/`

---

## 🧼 6. Clarity and Reproducibility

* Clear repo layout
* Commented code
* All dependencies in `requirements.txt`


---

## 🤝 7. Group Collaboration Summary

| Member        |Admission Number | Contribution                                                      | 
| --------------| --------------- | ----------------------------------------------------------------- |
| Lucy Njoroge  |   224177        | Docker setup, data model explanation, optional dataset ingestion  |
| Emmanuel Anasi|   224341        | CRUD operations, embedding logic, multiquery, Multi-query logic   |
| Bruce Onyango |   121063        | Markdown formatting, visuals, filtering logic                     |


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
