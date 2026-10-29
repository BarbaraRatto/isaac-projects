# Normalizzazione per gruppi — actor fisso a 300k

baseline_300k: tutta la normalizzazione a 300k.
goal_100k: solo media/varianza del goal XY da 100k.
other_state_100k: solo velocità lineari, angolari e potenza media da 100k.
gridmap_100k: solo feature visive PCA e geometriche da 100k.
all_100k: tutta la normalizzazione da 100k, controllo del precedente esperimento.
Mappa grezza, goal, altri input e pesi sono identici fra condizioni; cambiano solo le statistiche indicate.

| Replay | Goal ° | Condizione | vx m/s | wz rad/s | Sinistra | Δ wz vs baseline |
|---|---|---|---:|---:|---:|---:|
| 100k | original | baseline_300k | 0.677 | 1.253 | 93.6% | +0.000 |
| 100k | original | goal_100k | 0.785 | 0.776 | 77.5% | -0.477 |
| 100k | original | other_state_100k | 0.705 | 1.238 | 93.4% | -0.015 |
| 100k | original | gridmap_100k | 0.691 | 1.254 | 93.8% | +0.001 |
| 100k | original | all_100k | 0.826 | 0.729 | 76.0% | -0.524 |
| 100k | -90 | baseline_300k | 0.277 | 1.159 | 97.3% | +0.000 |
| 100k | -90 | goal_100k | 0.431 | 0.088 | 52.7% | -1.071 |
| 100k | -90 | other_state_100k | 0.283 | 1.133 | 97.1% | -0.026 |
| 100k | -90 | gridmap_100k | 0.286 | 1.158 | 97.5% | -0.001 |
| 100k | -90 | all_100k | 0.455 | -0.013 | 47.3% | -1.172 |
| 100k | -30 | baseline_300k | 0.668 | 0.479 | 68.6% | +0.000 |
| 100k | -30 | goal_100k | 0.756 | -0.408 | 33.0% | -0.887 |
| 100k | -30 | other_state_100k | 0.688 | 0.419 | 67.6% | -0.060 |
| 100k | -30 | gridmap_100k | 0.691 | 0.459 | 67.2% | -0.020 |
| 100k | -30 | all_100k | 0.798 | -0.518 | 27.9% | -0.997 |
| 100k | 0 | baseline_300k | 0.859 | 1.044 | 92.0% | +0.000 |
| 100k | 0 | goal_100k | 0.809 | -0.047 | 47.5% | -1.091 |
| 100k | 0 | other_state_100k | 0.892 | 1.006 | 91.0% | -0.039 |
| 100k | 0 | gridmap_100k | 0.881 | 1.039 | 92.0% | -0.005 |
| 100k | 0 | all_100k | 0.858 | -0.159 | 41.2% | -1.203 |
| 100k | 30 | baseline_300k | 0.916 | 1.440 | 100.0% | +0.000 |
| 100k | 30 | goal_100k | 0.843 | 0.834 | 83.2% | -0.606 |
| 100k | 30 | other_state_100k | 0.957 | 1.437 | 100.0% | -0.004 |
| 100k | 30 | gridmap_100k | 0.935 | 1.439 | 100.0% | -0.001 |
| 100k | 30 | all_100k | 0.895 | 0.767 | 80.1% | -0.673 |
| 100k | 90 | baseline_300k | 0.795 | 1.490 | 100.0% | +0.000 |
| 100k | 90 | goal_100k | 0.931 | 1.425 | 99.8% | -0.065 |
| 100k | 90 | other_state_100k | 0.831 | 1.490 | 100.0% | +0.000 |
| 100k | 90 | gridmap_100k | 0.817 | 1.490 | 100.0% | +0.000 |
| 100k | 90 | all_100k | 0.991 | 1.419 | 99.8% | -0.072 |
| 300k | original | baseline_300k | 0.718 | 1.229 | 95.7% | +0.000 |
| 300k | original | goal_100k | 0.871 | 0.499 | 69.7% | -0.730 |
| 300k | original | other_state_100k | 0.762 | 1.198 | 94.5% | -0.031 |
| 300k | original | gridmap_100k | 0.765 | 1.225 | 95.5% | -0.004 |
| 300k | original | all_100k | 0.950 | 0.402 | 65.6% | -0.827 |
| 300k | -90 | baseline_300k | 0.293 | 1.196 | 99.6% | +0.000 |
| 300k | -90 | goal_100k | 0.578 | -0.081 | 45.3% | -1.277 |
| 300k | -90 | other_state_100k | 0.309 | 1.158 | 99.4% | -0.038 |
| 300k | -90 | gridmap_100k | 0.325 | 1.193 | 99.6% | -0.003 |
| 300k | -90 | all_100k | 0.655 | -0.206 | 40.0% | -1.402 |
| 300k | -30 | baseline_300k | 0.864 | 0.486 | 68.2% | +0.000 |
| 300k | -30 | goal_100k | 0.996 | -0.540 | 30.3% | -1.026 |
| 300k | -30 | other_state_100k | 0.893 | 0.401 | 64.8% | -0.085 |
| 300k | -30 | gridmap_100k | 0.919 | 0.472 | 67.2% | -0.015 |
| 300k | -30 | all_100k | 1.064 | -0.645 | 26.4% | -1.131 |
| 300k | 0 | baseline_300k | 1.064 | 1.119 | 94.1% | +0.000 |
| 300k | 0 | goal_100k | 1.043 | -0.183 | 43.0% | -1.302 |
| 300k | 0 | other_state_100k | 1.105 | 1.056 | 93.2% | -0.063 |
| 300k | 0 | gridmap_100k | 1.112 | 1.119 | 93.8% | +0.001 |
| 300k | 0 | all_100k | 1.115 | -0.295 | 39.1% | -1.413 |
| 300k | 30 | baseline_300k | 1.063 | 1.455 | 100.0% | +0.000 |
| 300k | 30 | goal_100k | 1.058 | 0.733 | 80.7% | -0.722 |
| 300k | 30 | other_state_100k | 1.120 | 1.451 | 100.0% | -0.005 |
| 300k | 30 | gridmap_100k | 1.106 | 1.455 | 100.0% | -0.001 |
| 300k | 30 | all_100k | 1.144 | 0.634 | 74.4% | -0.821 |
| 300k | 90 | baseline_300k | 0.828 | 1.491 | 100.0% | +0.000 |
| 300k | 90 | goal_100k | 1.069 | 1.430 | 100.0% | -0.062 |
| 300k | 90 | other_state_100k | 0.890 | 1.491 | 100.0% | -0.000 |
| 300k | 90 | gridmap_100k | 0.877 | 1.492 | 100.0% | +0.000 |
| 300k | 90 | all_100k | 1.163 | 1.419 | 99.8% | -0.072 |

Tutte le direzioni e i singoli campioni sono nei CSV.

Limiti: combinazioni non addestrate, possibili input fuori distribuzione, campioni correlati.
Gli effetti dei gruppi possono interagire: non sono necessariamente additivi.
Il test localizza la sensibilità del comando; non dimostra una causa unica di apprendimento né successo in simulazione.
Nessun modello modificato, nessun limite alle rotazioni. Non usare lo scambio come correzione già validata.
