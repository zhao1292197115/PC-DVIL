# Real-Robot Battery-Assembly Results

This file records the latest 30-trial results supplied for the two real-robot test conditions. Condition 1 is the normal laboratory-lighting environment; Condition 2 is the natural-lighting environment.

| Condition | Successful trials | Total trials | Complete-task success |
|---|---:|---:|---:|
| Normal environment (laboratory lighting) | 27 | 30 | 90.0% |
| Natural-lighting environment | 25 | 30 | 83.3% |

## Stage-wise failure counts

The task is evaluated over four ordered stages: grasping, placement, stabilization/pressing, and final insertion/pushing.

| Condition | Stage 1 failures | Stage 2 failures | Stage 3 failures | Stage 4 failures |
|---|---:|---:|---:|---:|
| Normal environment | 0 | 0 | 2 | 1 |
| Natural-lighting environment | 0 | 0 | 3 | 2 |

## Trial-level failure indices

- Normal environment: trials 6 and 27 failed at Stage 3; trial 16 failed at Stage 4.
- Natural-lighting environment: trial 6 failed at Stage 3; trial 10 failed at Stage 4; trial 16 failed at Stage 3; trial 18 failed at Stage 3; trial 27 failed at Stage 4.

The latest results in this file supersede earlier draft values of 93.3% and 86.7%. Do not use those earlier values in the manuscript.

## Deployment configuration

The corresponding v7 real-robot inference implementation uses three-view RGB observations, a 40 Hz command-publishing loop, and inference-only feature-level Temporal Feature Cache with K_c = 2. Visual features are refreshed every two policy calls; current proprioception and ACT action decoding are processed at every policy query. The cache interval does not reduce the policy-query or action-publication frequency.

The feature-cache code path and the two 30-trial outcome lists are recorded separately here. The trial lists alone do not establish which exact software commit or launch script was used during each physical run; retain the run logs alongside the experimental record when available.
