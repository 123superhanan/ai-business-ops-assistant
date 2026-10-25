
---

AI Business Operations Assistant

An AI assistant for business operations, built in stages: a support-ticket classifier, then a RAG pipeline over policy documents, then a FastAPI service.

This README covers the dataset and the ML experiment plan. RAG and the API come later.

PROJECT STRUCTURE

- data/tickets.csv — labelled support tickets
- buisness_ML/training/ — PyTorch training notebooks
- business_rag/ — RAG pipeline (later)
- app.py, main.py — FastAPI service (later)

THE DATASET

data/tickets.csv has two columns. ticket_text is the customer message (the feature, X). category is the label (the target, y), one of billing, refund, technical, or shipping.

LABELING RULES

Tickets often overlap, so every row follows the same rules. Any new row must follow them too.

- Asks for money back, a return, a damaged item, or refund status: refund
- Disputes a charge, invoice, subscription, or extra fee: billing
- Login, password, app or site error, crash, feature broken: technical
- Delivery, tracking, courier, lost or late parcel: shipping

Edge cases, labelled by the rule above them:

- Technical problem on a refund or billing page (page shows error): technical
- Refund eligibility or "where is my refund": refund
- "Charged twice and want money back" (payment dispute): billing
- Cancel a subscription before the next payment: billing

Rule of thumb: money back for a product is refund, disputing a charge is billing, something broken is technical, a parcel involved is shipping.

These rules are written down before adding rows so labels stay consistent. Inconsistent labels cap accuracy no matter how good the model is.

DIFFICULTY

This is a Level 2 (realistic) dataset. It has ambiguous wording, overlapping categories, deliberate class imbalance, and some near-duplicate paraphrases. It does not include random typos or contradictory labels.

CLASS DISTRIBUTION (IMBALANCED ON PURPOSE)

- billing: 350 rows (35 percent)
- refund: 250 rows (25 percent)
- technical: 250 rows (25 percent)
- shipping: 150 rows (15 percent)
- total: 1000 rows

The imbalance forces macro-F1 and per-class recall to be reported, not just accuracy. A model that predicts billing for everything gets 35 percent accuracy but zero recall on the other classes.

Current status: working with a 160-row balanced seed (40 per class). The full 1000-row version comes from scripts/make_tickets.py once the pipeline is validated.

SPLITTING

Split the data 70 percent train, 15 percent validation, 15 percent test.

- Train fits the model.
- Validation tunes hyperparameters only.
- Test is evaluated once at the end, never tuned on.

Two rules:

- Stratify by category so every split has every class in the same proportion.
- Group near-duplicates (paraphrases of the same ticket) into the same split. Otherwise the test score is inflated by memorisation. Use a text fingerprint (sorted lowercase word set) as the group key with StratifiedGroupKFold.

EXPERIMENT PLAN

Work through these in order. Each finding informs the next.

1. Baseline. Train TF-IDF plus LogisticRegression. Report accuracy, macro-F1, per-class precision and recall, and the confusion matrix.
2. Error analysis. Dump every misclassified row with predicted probabilities. Group by the true and predicted pair. Expect confusions between billing and refund, technical and billing, shipping and refund.
3. Unigrams versus bigrams. Compare ngram_range (1,1) with (1,2). Bigrams catch phrases like "not arrived" and "charged twice".
4. Class weighting. Compare LogisticRegression() with class_weight set to balanced. Balanced raises macro-F1 for shipping at the cost of accuracy.
5. Regularisation. Sweep C in 0.01, 0.03, 0.1, 0.3, 1, 3, 10 on validation only. Pick by validation macro-F1.
6. Final evaluation. Freeze the config, fit on train plus validation, evaluate once on test.

BASELINES

- Dummy majority class: zero parameters. About 35 percent accuracy, about 0.1 macro-F1.
- TF-IDF plus LogisticRegression: about 2000 parameters. Strong baseline at this scale.
- PyTorch MLP: about 32000 parameters. Over-parameterised for 160 rows.

At around 160 training rows, logistic regression is expected to beat the MLP. The MLP is there to demonstrate the training loop, not to win until the dataset grows past about 1000 rows.

REPRODUCING

Create a virtual environment, activate it, then install pandas, scikit-learn, torch, and jupyter. Run scripts/make_tickets.py to generate the data, then open the notebook in buisness_ML/training.

All experiments use random_state=42. Pin it for comparability.

DONE WHEN

- Baseline reports accuracy, macro-F1, per-class metrics, and confusion matrix.
- Misclassified rows are inspected and the overlap pairs documented.
- Unigram versus bigram comparison is complete.
- class_weight balanced comparison is complete.
- C is tuned on validation only.
- Final model is evaluated once on test.
- Results are written up with a one-paragraph conclusion.

OUT OF SCOPE (LATER)

- RAG over policy documents
- FastAPI endpoints
- Time-based splits, label noise handling, drift monitoring

KEY TAKEAWAYS

- Write labeling rules before the data, or labels drift and no model can recover.
- Macro-F1, not accuracy, is the metric for an imbalanced four-class problem.
- Stratify splits and group near-duplicates into the same fold.
- Baselines first. The dummy gives the floor, logistic regression gives the ceiling at this size.
- Do not tune on test. Every peek spends that estimate.

---
