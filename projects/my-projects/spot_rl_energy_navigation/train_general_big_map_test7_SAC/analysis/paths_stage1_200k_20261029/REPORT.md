# Test 7 SAC — 200.000 transizioni — fase 1

Fonte: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test7_SAC/results/evaluation_20261029_233158_931651`
Checkpoint: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test7_SAC/runs/general_map_test7_sac_20261029_163507_402204/policy_final.pt`

Successi: 0/9. Esiti: {'goal': 0, 'fallen': 0, 'out_of_bounds': 0, 'timeout': 9, 'alignment_timeout': 0}.
Allineamento iniziale riuscito: 9/9.

| Episodio | Esito | A–B diretto (m) | Percorso (m) | Distanza finale (m) | Tempo (s) | Energia (J) |
|---|---|---:|---:|---:|---:|---:|
| 1 | Timeout | 4.86 | 9.78 | 4.12 | 34.6 | 3547 |
| 2 | Timeout | 5.29 | 9.72 | 6.60 | 36.2 | 3592 |
| 3 | Timeout | 4.40 | 6.85 | 2.70 | 32.8 | 3244 |
| 4 | Timeout | 5.14 | 6.13 | 3.83 | 35.6 | 3361 |
| 5 | Timeout | 5.99 | 12.49 | 2.85 | 39.0 | 4787 |
| 6 | Timeout | 5.69 | 6.48 | 4.08 | 37.8 | 3365 |
| 7 | Timeout | 4.68 | 6.55 | 2.36 | 33.8 | 3702 |
| 8 | Timeout | 3.53 | 5.61 | 3.58 | 30.0 | 2915 |
| 9 | Timeout | 3.47 | 9.02 | 3.34 | 30.0 | 3473 |

Tempo, percorso ed energia comprendono l’allineamento iniziale. Raggio di successo: 0,55 m.
Nove timeout, nessuna caduta: il problema in questo test è raggiungere il goal entro il tempo disponibile.
Le distanze finali non dimostrano da sole la causa: consultare i percorsi e la distanza dal goal nel tempo.

## Distanza minima dal goal nei campioni registrati

| Episodio | Minima distanza (m) |
|---|---:|
| 1 | 2.454 |
| 2 | 3.870 |
| 3 | 2.697 |
| 4 | 3.122 |
| 5 | 1.009 |
| 6 | 3.736 |
| 7 | 2.102 |
| 8 | 2.999 |
| 9 | 2.252 |

## Artefatti

- percorsi_test7_SAC_fase1_200k.pdf / .png: panoramica con start, goal e percorsi.
- episodi_dettaglio.pdf: un episodio per pagina, percorso e distanza dal goal nel tempo.
- riepilogo_metriche.pdf: distanza iniziale, percorso ed energia.
- riepilogo_episodi.csv: tabella con coordinate e metriche.

Verificati numero dei campioni, fasi, goal, posizione terminale e diagnostica post-step.
Scale diverse tra episodi. Mappa dei materiali schematica; nessuna simulazione avviata.
