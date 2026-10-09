# SWYNEX Task 3 — Intelligent Feature

## Summary
This project extends the Task 2 feedback classifier with confidence-aware predictions and human-review routing. It includes error handling, evaluation examples, deliberate failure cases, a CLI demo, and an optional Streamlit interface.

## Intelligent feature
For each non-empty message, the app returns the predicted category, top class probability, and a human-review flag when confidence is below a configurable threshold (default 0.45). Low confidence does not prove a prediction is wrong, and high confidence does not guarantee correctness.

## Error handling
- Empty and whitespace-only input is rejected.
- Input longer than 2,000 characters is rejected.
- Invalid confidence thresholds are rejected.
- Missing/malformed datasets and unknown labels produce clear errors.
- Unexpected runtime errors are caught at the CLI boundary.
- The Streamlit UI displays user-friendly errors.

## Evaluation
Run:

    python src/evaluate.py

The script reports held-out accuracy, macro-F1, precision, recall and per-class metrics. It also evaluates curated examples and prints qualitative failure cases. Ambiguous/out-of-domain input is not scored as correct because the model must choose one of the known labels.

## Demo interface
CLI:

    python src/app.py --text "Could you clarify the task deadline?"

Optional browser UI:

    streamlit run src/streamlit_app.py

## Limitations
The dataset is small and intended for learning only. The evaluation split is not a substitute for external validation. Short text can be ambiguous; probabilities may be poorly calibrated; out-of-domain text may receive a misleading category. Before real use, expand the dataset and evaluate on independently labeled data.

## Responsible use
The included sample data contains no real student records. Avoid entering sensitive personal information. Keep a human in the loop for consequential decisions.

## Files
- src/feedback_model.py — shared model, validation and prediction
- src/app.py — command-line demo and error handling
- src/evaluate.py — evaluation and failure-case runner
- src/streamlit_app.py — interactive UI
- data/sample_feedback.csv — demonstration dataset
- examples/evaluation_cases.json — expected outputs and failure cases

No API key or secret is required.
