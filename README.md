# SWYNEX Intelligent Feature — Task 3

An intelligent feedback triage prototype extending SWYNEX Tasks 1 and 2.

## Feature
Classify a message into one of five categories, return a confidence score, and flag low-confidence predictions for human review. The CLI validates input and handles missing data/model errors.

Pipeline: TF-IDF → Logistic Regression (scikit-learn).

Categories: Positive Feedback, Negative Feedback, Question, Suggestion, Complaint.

## Quick start
Python 3.10+ recommended.

    pip install -r requirements.txt
    python src/app.py
    python src/app.py --text "Could you clarify the task deadline?"
    python src/evaluate.py

Optional interactive UI:

    streamlit run src/streamlit_app.py

## Included
- Confidence-aware prediction and configurable human-review threshold
- Empty/whitespace/overlong input validation
- Friendly errors for invalid thresholds and missing or malformed datasets
- Accuracy, macro-F1, precision, recall and per-class report
- Curated evaluation examples and deliberate failure cases
- Optional Streamlit interface

The sample dataset is small demonstration data, not a production corpus. Scores demonstrate the workflow, not real-world readiness. No API keys or secrets are required.
