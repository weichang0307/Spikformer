---
description: Check whether the current training run is doing better than the best previous run, and send the comparison curves
argument-hint: "[current run] [reference runs...]  (default: the run training now vs the best finished run of its kind)"
---

Tell the user whether the current training run is doing better than the run it should beat, and show the curves.

1. Run `python3 compare_runs.py $ARGUMENTS` from the repository root. With no arguments it takes the run
   training now (most recently updated `live.json`) and compares it with the finished run of the same corpus and
   model type that has the lowest best validation loss; the corpus' ANN and LSTM are drawn as dotted context. If
   it reports no current run, list the recently updated runs in `runs/` and ask which one to compare. If the
   automatic reference is not the run the current one changes (e.g. the user just changed one setting of an
   earlier run), rerun with both names explicitly: `python3 compare_runs.py CURRENT REFERENCE`.
2. Send `runs/analysis/compare.png` to the user with SendUserFile (`display: "render"`, `status: "normal"`), with a
   one-line caption naming the current run, its step and its validation loss against the reference's at the same
   step. The user is often on remote control, so always send the image.
3. Reply with the verdict first (better / worse / level, by how much, at which step), then a short table from the
   script's output: step, reference val and gap, current val and gap, difference. Add at most two sentences on
   the trend (is the lead growing or shrinking; is the train/val gap opening up). Mention the expected finish if
   the run is still training. Do not overstate a lead of under ~0.01 bits/token: say it is within noise.
