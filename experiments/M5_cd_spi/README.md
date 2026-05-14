# M5: Cross-Domain Step Polysemy Index (CD-SPI)

## Objective
Measure whether step embeddings share universal substrate across domains.

## Hypothesis
Logical connector steps have low CD-SPI (universal), while domain-specific references have high CD-SPI (polysemous).

## Method
1. Train client-specific PRMs
2. Extract embeddings for shared step categories
3. Compute CD-SPI = 1 - mean cosine similarity

## Status
✅ M1 verification complete (measurement on initialized models)

### Mini Verification Results
CD-SPI computed on 4 anchor steps across 4 clients (initialized models):

| Step | CD-SPI | Interpretation |
|------|--------|---------------|
| "Let x be the variable..." | 0.0000 | Perfect alignment (universal) |
| "First, identify the key constraints..." | 1.0000 | Max polysemy |
| "Therefore, the answer must be positive." | 0.0000 | Perfect alignment (universal) |
| "We can verify this by substitution." | 0.0005 | Near-perfect alignment |
| **Average** | **0.2501** | — |

**Note**: Results from uninitialized models serve as baseline.
After M4 training, CD-SPI should differentiate step types more clearly.

⏳ Post-training measurement pending full M4 run.
