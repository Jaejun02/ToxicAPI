# 🚀 Toxic Comment Classification API

This project provides a **FastAPI**-based web service for classifying toxic comments using a **unitary's Toxic BERT model**. The API allows users to analyze text for toxicity and submit feedback on classification results.

---
## 🧐 About
This API detects toxic content in text using **natural language processing (NLP)**. It leverages **BERT** for classification and includes functionality for **storing user feedback** in an SQLite database. 

---
## ✨ Key Features
- **FastAPI-powered** for high-performance request handling  
- **BERT-based toxic comment classification**  
- **Database storage** for user feedback (SQLite)  
- **Swagger UI and ReDoc** for interactive API documentation  
- **Security middleware** for enhanced protection  
- **Pagination-enabled feedback retrieval**  

---
## ⚙️ Installation
### Prerequisites
1. **Clone the repository** from GitHub:
```bash
git clone <https://github.com/Jaejun02/ToxicAPI.git>
cd ToxicAPI
```
2. **Create a Conda environment and install dependencies**:
```bash
conda env create -f environment.yml
conda activate toxic-comment-api
```
3. **Set up environment variables**:
   - Create a `.env` file in the project root.
   - Add the following line to the file:
   ```
   HF_TOKEN=hugging_face_token_here
   ```

---
## 🚀 Usage & API Endpoints
### Running the Server
Start the FastAPI server with:
```bash
uvicorn app:app --reload
```

### Accessing API Documentation
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### API Endpoints
#### Classification Endpoint
- **POST** `/classify`
  - **Input**: JSON object with a `comment` field, e.g.:
    ```json
    {
      "comment": "This is an example toxic comment."
    }
    ```
  - **Output**: JSON object with predicted labels and probabilities, e.g.:
    ```json
    {
      "labels": ["toxic", "insult"],
      "probabilities": {
        "toxic": 0.85,
        "severe_toxic": 0.02,
        "obscene": 0.15,
        "threat": 0.01,
        "insult": 0.78,
        "identity_hate": 0.03
      }
    }
    ```

#### Feedback Submission
- **POST** `/submit-feedback`
  - **Input**: JSON object with the original `comment` and `expected_labels`, e.g.:
    ```json
    {
      "comment": "This is an example toxic comment.",
      "expected_labels": ["toxic", "insult"]
    }
    ```
  - **Output**: Confirmation message.
    ```json
    {
      "message": "Feedback stored successfully"
    }
    ```

#### Feedback Retrieval
- **GET** `/feedback-stats`
  - **Output**: Total number of stored feedback entries.
    ```json
    {
      "total_feedback_entries": 123
    }
    ```
- **GET** `/view-feedback`
  - **Input**: Optional query parameters `limit` and `offset`.
  - **Output**: List of feedback entries.
    ```json
    [
      {
        "id": 1,
        "comment": "This is an example toxic comment.",
        "toxic": true,
        "severe_toxic": false,
        "obscene": false,
        "threat": false,
        "insult": true,
        "identity_hate": false,
        "timestamp": "2024-12-23T12:34:56"
      }
    ]
    ```

---
## 📜 License
This project is licensed under the **MIT License**.

---
📌 **Author:** Jaejun Shim  
📆 **Date:** January 12th, 2025

