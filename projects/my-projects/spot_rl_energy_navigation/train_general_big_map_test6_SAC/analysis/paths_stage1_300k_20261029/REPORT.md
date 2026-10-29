# Test 6 SAC — fase 1 — 300.000 transizioni

Fonte: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test6_SAC/results/evaluation_20261029_110049_426900`
Checkpoint: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test6_SAC/runs/general_map_test6_sac_20261029_001023_953670/policy_final.pt`

Successi: 8/9 (88.9%). Un timeout, nessuna caduta o uscita dalla mappa. Allineamento riuscito 9/9. Fase 1: terreni misti 3–6 m senza scale/ostacoli.

| Episodio | Esito | A–B diretto (m) | Percorso (m) | Distanza finale (m) | Tempo (s) | Energia (J) |
|---|---|---:|---:|---:|---:|---:|
| 1 | Goal raggiunto | 4.86 | 8.03 | 0.55 | 12.2 | 1783 |
| 2 | Goal raggiunto | 5.29 | 12.40 | 0.47 | 16.2 | 2890 |
| 3 | Goal raggiunto | 4.40 | 15.51 | 0.47 | 20.8 | 3888 |
| 4 | Timeout | 5.14 | 22.22 | 0.79 | 35.6 | 5472 |
| 5 | Goal raggiunto | 5.99 | 12.94 | 0.47 | 16.0 | 2856 |
| 6 | Goal raggiunto | 5.69 | 12.05 | 0.51 | 20.8 | 3007 |
| 7 | Goal raggiunto | 4.68 | 11.86 | 0.45 | 15.2 | 2672 |
| 8 | Goal raggiunto | 3.53 | 12.57 | 0.47 | 18.6 | 3081 |
| 9 | Goal raggiunto | 3.47 | 13.46 | 0.52 | 19.0 | 3131 |

Distanze in metri, energia meccanica in joule. Tempo, percorso ed energia includono l’allineamento. Il raggio di successo è 0,55 m. A–B è la distanza euclidea, non una traiettoria ottima sul terreno.

## Interpretazione

Otto missioni su nove vengono completate. Resta un problema di percorsi lunghi: nei successi le traiettorie misurano da 8,03 a 15,51 m, a fronte di distanze iniziali di 3,47–5,99 m. Raggiungere il goal non dimostra di aver imparato una navigazione energeticamente efficiente.

Nei soli successi: percorso medio 12.35 m, tempo medio 17.35 s, energia media 2914 J.

L’episodio 4 termina per timeout dopo 35,6 s e 22,22 m percorsi, a 0,79 m dal centro del goal: circa 0,24 m fuori dal raggio di successo. È vicino al goal alla scadenza, ma la lunghezza del percorso segnala che aumentare il timeout da solo non risolverebbe l’inefficienza.

Questo test usa la fase 1; il precedente test SAC a 100.000 transizioni usava la fase 0. Le due percentuali non sono un confronto a difficoltà uguale. Le nove missioni non bastano a stimare con precisione il successo generale.

## Grafici e verifiche

- `percorsi_test6_SAC_fase1_300k.pdf` / `.png`: nove percorsi, start A nero, goal B verde, allineamento arancione, navigazione blu e posa finale a croce rossa.
- `episodi_dettaglio.pdf`: una pagina per episodio, con percorso e distanza dal goal nel tempo.
- `riepilogo_metriche.pdf`: distanza iniziale, percorso ed energia.
- `riepilogo_episodi.csv`: dati numerici e coordinate.

Scale diverse per episodio. La posa terminale è aggiunta ai campioni pre-step. Verificati numero e ordine dei campioni, fasi, coordinate goal, distanza finale e corrispondenza con la diagnostica post-step. Nessuna simulazione avviata.

Riproduzione offline:
```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 spot_rl_energy_navigation/train_general_big_map_test6_SAC/analysis/paths_stage1_300k_20261029/plot_paths.py
```
