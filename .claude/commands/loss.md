---
description: Plot the live training curves (recall and loss) and send the figure
argument-hint: "[run names...]  (default: runs updated in the last 10 minutes)"
---

Show the user the current training curves.

1. Run `python3 plot_curves.py --brief $ARGUMENTS` from the repository root (`--brief`: only the key panels, at high
   resolution; the user asked for this). With no arguments it plots every
   run whose `runs/<run>/live.json` changed in the last 10 minutes; if it reports none, rerun with
   `--brief --minutes 120`, or list `runs/` and ask which runs to plot.
2. Send `runs/analysis/curves.png` to the user with SendUserFile (`display: "render"`,
   `status: "normal"`), with a one-line caption naming the runs and their latest recall. The user is
   often on remote control and cannot see the in-app browser, so always send the image.
3. Reply with a short table from the script's output: run, step, recall at the largest T and at T=4,
   eval loss. Then add at most two sentences on what changed since the last check (plateau,
   jump, collapse), using the "every 1,000 steps" lines.
