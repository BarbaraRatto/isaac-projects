# Pesi actor e normalizzazione: confronto incrociato

Stessi campioni e goal per tutte le combinazioni; nessuna modifica ai modelli.
Le combinazioni originali sono verificate equivalenti al normale percorso di inferenza.

## Statistiche salvate del goal (unità: metri)

| Normalizzazione | Input | Media | Deviazione standard |
|---|---|---:|---:|
| 100k | goal_x | -0.3421 | 2.3953 |
| 100k | goal_y | 0.3564 | 2.0388 |
| 300k | goal_x | 0.9399 | 2.2959 |
| 300k | goal_y | -0.0853 | 1.5146 |

## Risposte (selezione; tutte le direzioni nei CSV)

| Replay | Goal ° | Actor | Norm | vx m/s | wz rad/s | Sinistra | Δ wz vs norm propria |
|---|---|---|---|---:|---:|---:|---:|
| 100k | original | 100k | 100k | 0.370 | 0.885 | 83.8% | +0.000 |
| 100k | original | 300k | 100k | 0.826 | 0.729 | 76.0% | -0.524 |
| 100k | original | 100k | 300k | 0.283 | 1.129 | 92.2% | +0.244 |
| 100k | original | 300k | 300k | 0.677 | 1.253 | 93.6% | +0.000 |
| 100k | -90 | 100k | 100k | 0.328 | 0.429 | 63.9% | +0.000 |
| 100k | -90 | 300k | 100k | 0.455 | -0.013 | 47.3% | -1.172 |
| 100k | -90 | 100k | 300k | 0.246 | 0.686 | 76.2% | +0.256 |
| 100k | -90 | 300k | 300k | 0.277 | 1.159 | 97.3% | +0.000 |
| 100k | -30 | 100k | 100k | 0.389 | 0.514 | 67.8% | +0.000 |
| 100k | -30 | 300k | 100k | 0.798 | -0.518 | 27.9% | -0.997 |
| 100k | -30 | 100k | 300k | 0.301 | 0.808 | 82.4% | +0.295 |
| 100k | -30 | 300k | 300k | 0.668 | 0.479 | 68.6% | +0.000 |
| 100k | 0 | 100k | 100k | 0.415 | 0.725 | 79.1% | +0.000 |
| 100k | 0 | 300k | 100k | 0.858 | -0.159 | 41.2% | -1.203 |
| 100k | 0 | 100k | 300k | 0.323 | 1.041 | 90.8% | +0.316 |
| 100k | 0 | 300k | 300k | 0.859 | 1.044 | 92.0% | +0.000 |
| 100k | 30 | 100k | 100k | 0.420 | 0.954 | 87.5% | +0.000 |
| 100k | 30 | 300k | 100k | 0.895 | 0.767 | 80.1% | -0.673 |
| 100k | 30 | 100k | 300k | 0.323 | 1.235 | 97.1% | +0.281 |
| 100k | 30 | 300k | 300k | 0.916 | 1.440 | 100.0% | +0.000 |
| 100k | 90 | 100k | 100k | 0.370 | 1.200 | 95.7% | +0.000 |
| 100k | 90 | 300k | 100k | 0.991 | 1.419 | 99.8% | -0.072 |
| 100k | 90 | 100k | 300k | 0.277 | 1.370 | 99.4% | +0.170 |
| 100k | 90 | 300k | 300k | 0.795 | 1.490 | 100.0% | +0.000 |
| 300k | original | 100k | 100k | 0.233 | 0.831 | 88.9% | +0.000 |
| 300k | original | 300k | 100k | 0.950 | 0.402 | 65.6% | -0.827 |
| 300k | original | 100k | 300k | 0.167 | 1.149 | 96.5% | +0.318 |
| 300k | original | 300k | 300k | 0.718 | 1.229 | 95.7% | +0.000 |
| 300k | -90 | 100k | 100k | 0.197 | 0.386 | 61.1% | +0.000 |
| 300k | -90 | 300k | 100k | 0.655 | -0.206 | 40.0% | -1.402 |
| 300k | -90 | 100k | 300k | 0.138 | 0.703 | 78.3% | +0.317 |
| 300k | -90 | 300k | 300k | 0.293 | 1.196 | 99.6% | +0.000 |
| 300k | -30 | 100k | 100k | 0.240 | 0.465 | 65.0% | +0.000 |
| 300k | -30 | 300k | 100k | 1.064 | -0.645 | 26.4% | -1.131 |
| 300k | -30 | 100k | 300k | 0.170 | 0.824 | 85.9% | +0.360 |
| 300k | -30 | 300k | 300k | 0.864 | 0.486 | 68.2% | +0.000 |
| 300k | 0 | 100k | 100k | 0.257 | 0.688 | 79.1% | +0.000 |
| 300k | 0 | 300k | 100k | 1.115 | -0.295 | 39.1% | -1.413 |
| 300k | 0 | 100k | 300k | 0.183 | 1.066 | 94.7% | +0.378 |
| 300k | 0 | 300k | 300k | 1.064 | 1.119 | 94.1% | +0.000 |
| 300k | 30 | 100k | 100k | 0.260 | 0.936 | 91.6% | +0.000 |
| 300k | 30 | 300k | 100k | 1.144 | 0.634 | 74.4% | -0.821 |
| 300k | 30 | 100k | 300k | 0.184 | 1.261 | 99.6% | +0.325 |
| 300k | 30 | 300k | 300k | 1.063 | 1.455 | 100.0% | +0.000 |
| 300k | 90 | 100k | 100k | 0.227 | 1.199 | 99.6% | +0.000 |
| 300k | 90 | 300k | 100k | 1.163 | 1.419 | 99.8% | -0.072 |
| 300k | 90 | 100k | 300k | 0.158 | 1.385 | 100.0% | +0.186 |
| 300k | 90 | 300k | 300k | 0.828 | 1.491 | 100.0% | +0.000 |

## Clipping sugli stati originali

| Replay | Norm | Gruppo | Valori validi | % oltre ±10 prima del clipping |
|---|---|---|---:|---:|
| 100k | 100k | goal_x | 512 | 0.0000% |
| 100k | 100k | goal_y | 512 | 0.0000% |
| 100k | 100k | state_normalized | 4608 | 0.0217% |
| 100k | 100k | visual_valid | 967264 | 0.0000% |
| 100k | 100k | geometry_valid | 120272 | 0.0017% |
| 100k | 300k | goal_x | 512 | 0.0000% |
| 100k | 300k | goal_y | 512 | 0.0000% |
| 100k | 300k | state_normalized | 4608 | 0.0217% |
| 100k | 300k | visual_valid | 967264 | 0.0000% |
| 100k | 300k | geometry_valid | 120272 | 0.0000% |
| 300k | 100k | goal_x | 512 | 0.0000% |
| 300k | 100k | goal_y | 512 | 0.0000% |
| 300k | 100k | state_normalized | 4608 | 0.0000% |
| 300k | 100k | visual_valid | 1164352 | 0.0000% |
| 300k | 100k | geometry_valid | 145028 | 0.0021% |
| 300k | 300k | goal_x | 512 | 0.0000% |
| 300k | 300k | goal_y | 512 | 0.0000% |
| 300k | 300k | state_normalized | 4608 | 0.0217% |
| 300k | 300k | visual_valid | 1164352 | 0.0000% |
| 300k | 300k | geometry_valid | 145028 | 0.0021% |

## Limiti

Lo scambio riguarda l’intera normalizzazione dell’actor, non soltanto il goal.
Le combinazioni incrociate non sono policy addestrate: pesi e statistiche si sono adattati insieme.
Una grande variazione dimostra sensibilità allo scambio, non un bug o una causa unica del comportamento.
Assenza di clipping non implica assenza di effetti della normalizzazione.
Goal artificiali e combinazioni incrociate possono essere fuori distribuzione.
Campioni correlati da replay, non dai nove episodi visualizzati; nessuna misura di successo futuro.
Non usare le combinazioni incrociate come correzione pronta per il robot.
