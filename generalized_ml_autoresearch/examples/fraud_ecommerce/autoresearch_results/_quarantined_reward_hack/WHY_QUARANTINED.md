# Quarantined: Exps 19-23 — REWARD HACKING

These experiments were rejected for changing the test set size.

Each one trimmed the dataset to a recent slice (e.g., rows 60k-151112) and then
computed `test_fraction=0.2` of the trimmed dataset, producing test sets of 11k
rows instead of the FDB-protocol 30k rows.

The "improvement" of +0.05 to +0.075 AUC was largely an artifact of testing on
a smaller, more homogeneous, more recent subset. AUC on the SAME test set as
the published FDB benchmark would not have moved nearly as much.

Lesson: never change the test set. To vary training data, use the new
HoldoutSplit `min_train_idx` parameter which keeps the test indices identical
to the published benchmark. See the new "Reward Hacking Prohibition" section
in CLAUDE_template.md.
