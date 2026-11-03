# Test 8 SAC — 600.000 transizioni — fase 1

Fonte: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test8_SAC/results/evaluation_20261101_011743_792676`
Checkpoint: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test8_SAC/runs/general_map_test8_sac_20261031_010201_266860/policy_final.pt`

Successi: 6/9. Esiti: {'goal': 6, 'fallen': 0, 'out_of_bounds': 0, 'timeout': 3, 'alignment_timeout': 0}.
Allineamento iniziale riuscito: 9/9.

| Episodio | Esito | A–B diretto (m) | Percorso (m) | Distanza finale (m) | Tempo (s) | Energia (J) |
|---|---|---:|---:|---:|---:|---:|
| 1 | Goal raggiunto | 4.86 | 13.32 | 0.54 | 21.0 | 2508 |
| 2 | Goal raggiunto | 5.29 | 15.08 | 0.49 | 21.6 | 2983 |
| 3 | Timeout | 4.40 | 16.86 | 3.23 | 32.8 | 3977 |
| 4 | Goal raggiunto | 5.14 | 5.14 | 0.46 | 7.6 | 1085 |
| 5 | Timeout | 5.99 | 14.34 | 6.52 | 39.0 | 4226 |
| 6 | Timeout | 5.69 | 14.22 | 5.89 | 37.8 | 3514 |
| 7 | Goal raggiunto | 4.68 | 5.51 | 0.49 | 7.4 | 1058 |
| 8 | Goal raggiunto | 3.53 | 3.88 | 0.43 | 5.4 | 842 |
| 9 | Goal raggiunto | 3.47 | 3.55 | 0.55 | 6.6 | 895 |

Tempo, percorso ed energia comprendono l’allineamento iniziale. Raggio di successo: 0,55 m.
Nei soli successi: percorso medio 7.75 m, tempo medio 11.60 s, energia media 1562 J.
Le distanze finali non dimostrano da sole la causa: consultare i percorsi e la distanza dal goal nel tempo.

## Distanza minima dal goal nei campioni registrati

| Episodio | Minima distanza (m) |
|---|---:|
| 1 | 0.541 |
| 2 | 0.487 |
| 3 | 0.680 |
| 4 | 0.462 |
| 5 | 2.998 |
| 6 | 2.229 |
| 7 | 0.487 |
| 8 | 0.432 |
| 9 | 0.548 |

## Artefatti

- percorsi_test8_SAC_fase1_600k.pdf / .png: panoramica con start, goal e percorsi.
- episodi_dettaglio.pdf: un episodio per pagina, percorso e distanza dal goal nel tempo.
- riepilogo_metriche.pdf: distanza iniziale, percorso ed energia.
- riepilogo_episodi.csv: tabella con coordinate e metriche.

Verificati numero dei campioni, fasi, goal, posizione terminale e diagnostica post-step.
Scale diverse tra episodi. Mappa dei materiali schematica; nessuna simulazione avviata.
