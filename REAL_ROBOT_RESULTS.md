# Real-Robot Battery-Assembly Results

This file reports the 30-trial battery-assembly results for the two real-robot lighting conditions.

| Test environment | Successful trials | Total trials | Complete-task success |
|---|---:|---:|---:|
| Normal laboratory lighting | 27 | 30 | 90.0% |
| Natural lighting | 25 | 30 | 83.3% |

## Stage-wise failure counts

The task is evaluated over four ordered stages: grasping, placement, stabilization/pressing, and final insertion/pushing.

| Test environment | Stage 1 failures | Stage 2 failures | Stage 3 failures | Stage 4 failures |
|---|---:|---:|---:|---:|
| Normal laboratory lighting | 0 | 0 | 2 | 1 |
| Natural lighting | 0 | 0 | 3 | 2 |

## Trial-level failure indices

- Normal laboratory lighting: trials 6 and 27 failed at Stage 3; trial 16 failed at Stage 4.
- Natural lighting: trial 6 failed at Stage 3; trial 10 failed at Stage 4; trial 16 failed at Stage 3; trial 18 failed at Stage 3; trial 27 failed at Stage 4.

## Deployment configuration

The real-robot inference implementation uses three-view RGB observations, a 40 Hz command-publishing loop, and inference-only feature-level Temporal Feature Cache with K_c = 2. Visual features are refreshed every two policy calls; current proprioception and ACT action decoding are processed at every policy query. The cache interval does not reduce the policy-query or action-publication frequency.

Both 30-trial sets were collected using the Feature-TFC real-robot implementation described above.
