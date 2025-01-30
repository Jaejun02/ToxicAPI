"""
Toxic Comment Classification API using FastAPI

This application provides endpoints for classifying toxic comments using a BERT model
and collecting user feedback on predictions.
"""

import os
import sqlite3
from datetime import datetime
from typing import List

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware import Middleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

# --------------------------
# Environment Configuration
# --------------------------
load_dotenv()
hf_token = os.getenv("HF_TOKEN", "")
MODEL_NAME = "unitary/toxic-bert"
DB_NAME = "feedback_data.db"
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# ------------------------
# FastAPI Initialization
# ------------------------
app = FastAPI(
    title="Toxic Comment Classifier",
    description="API for detecting toxic content using BERT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security Middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Consider restricting in production
)

# ----------------------
# Model Initialization
# ----------------------
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=hf_token)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Model initialization failed: {str(e)}")

# ----------------------
# Database Setup
# ----------------------
def init_db():
    """Initialize feedback database if it does not exist."""
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS feedback
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      comment TEXT,
                      toxic INTEGER,
                      severe_toxic INTEGER,
                      obscene INTEGER,
                      threat INTEGER,
                      insult INTEGER,
                      identity_hate INTEGER,
                      timestamp DATETIME)''')

init_db()

# ----------------------
# Data Models
# ----------------------
class ClassificationRequest(BaseModel):
    """Request schema for classification endpoint."""
    comment: str = Field(..., min_length=1, max_length=1000,
                         example="This is an example comment")

class ClassificationResponse(BaseModel):
    """Response schema for classification results"""
    labels: List[str]
    probabilities: dict

class FeedbackRequest(BaseModel):
    """Schema for submitting feedback"""
    comment: str
    expected_labels: List[str]

class FeedbackResponse(BaseModel):
    """Schema for feedback entry details."""
    id: int
    comment: str
    toxic: bool
    severe_toxic: bool
    obscene: bool
    threat: bool
    insult: bool
    identity_hate: bool
    timestamp: datetime

    class Config:
        orm_mode = True

# ----------------------
# Core Logic
# ----------------------
def classify_comment(comment: str, threshold: float = 0.5) -> dict:
    """
    Classify text using BERT model.

    Parameters:
    - comment: Text to analyze.
    - threshold: Confidence cutoff for labels (default: 0.5).

    Returns:
    - dict: Contains predicted labels and probability scores.
    """
    try:
        inputs = tokenizer(comment, 
                          return_tensors="pt", 
                          truncation=True, 
                          max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()
        predictions = {label: float(probs[i]) 
                      for i, label in enumerate(LABELS)}
        
        predicted_labels = [
            label for label, prob in predictions.items() 
            if prob >= threshold
        ]
        
        return {
            'labels': predicted_labels or ['none'],
            'probabilities': predictions
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Classification error: {str(e)}")

# ----------------------
# API Endpoints
# ----------------------
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    """Redirects root URL to API documentation."""
    return RedirectResponse(url="/docs")

@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """
    Analyze text for toxic content.

    Parameters:
    - request: Contains text to classify (ClassificationRequest).

    Returns:
    - ClassificationResponse: Labels and probability scores.
    """
    return classify_comment(request.comment)

@app.post("/submit-feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Store user feedback on classification results.

    Parameters:
    - feedback: Contains original comment and expected labels.

    Returns:
    - dict: Operation status confirmation.
    """
    invalid_labels = [
        label for label in feedback.expected_labels 
        if label not in LABELS and label != 'none'
    ]
    
    if invalid_labels:
        raise HTTPException(400, 
            detail=f"Invalid labels: {', '.join(invalid_labels)}")
    
    label_values = [
        1 if label in feedback.expected_labels else 0 
        for label in LABELS
    ]
    
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute('''INSERT INTO feedback 
                         (comment, toxic, severe_toxic, obscene, threat, insult, identity_hate, timestamp)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (feedback.comment, *label_values, datetime.now()))
            conn.commit()
        return {"message": "Feedback stored successfully"}
    except Exception as e:
        raise HTTPException(500, detail=f"Database error: {str(e)}")

@app.get("/feedback-stats")
async def get_feedback_stats():
    """
    Get feedback collection statistics.

    Returns:
    - dict: Total number of stored feedback entries.
    """
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM feedback")
            count = cursor.fetchone()[0]
        return {"total_feedback_entries": count}
    except Exception as e:
        raise HTTPException(500, detail=f"Database error: {str(e)}")
    
@app.get("/view-feedback", response_model=List[FeedbackResponse])
async def view_feedback(limit: int = 100, offset: int = 0):
    """
    Retrieve submitted feedback entries with pagination.
    
    Parameters:
    - limit: Maximum entries to return (default: 100).
    - offset: Number of entries to skip (default: 0).
    
    Returns:
    - List of feedback entries with full details.
    """
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, comment, 
                       toxic, severe_toxic, obscene, 
                       threat, insult, identity_hate,
                       timestamp 
                FROM feedback 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve feedback: {str(e)}"
        )