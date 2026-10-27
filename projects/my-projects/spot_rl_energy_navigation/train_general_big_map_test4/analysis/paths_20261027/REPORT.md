# Percorsi test 4 — 300.000 transizioni

Fonte: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test4/results/evaluation_20261027_130913_760448`
Checkpoint: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test4/runs/general_map_test4_ppo_20261027_003916_993735/policy_final.pt`

Fase 1: terreni misti 3–6 m senza scale/ostacoli. Successi: 4/9 (44.4%); 3 cadute e 2 timeout. Allineamento completato in tutti i 9 episodi.

| Episodio | Esito | A–B diretto (m) | Percorso (m) | Distanza finale (m) | Tempo (s) | Energia (J) |
|---|---|---:|---:|---:|---:|---:|
| 1 | Goal raggiunto | 4.86 | 10.13 | 0.55 | 16.8 | 1923 |
| 2 | Timeout | 5.29 | 22.80 | 5.79 | 36.2 | 4071 |
| 3 | Caduta | 4.40 | 19.70 | 1.94 | 30.6 | 3701 |
| 4 | Goal raggiunto | 5.14 | 16.75 | 0.53 | 26.0 | 3005 |
| 5 | Caduta | 5.99 | 7.30 | 2.03 | 11.0 | 1762 |
| 6 | Caduta | 5.69 | 5.07 | 3.25 | 9.8 | 1133 |
| 7 | Goal raggiunto | 4.68 | 13.33 | 0.51 | 21.6 | 2616 |
| 8 | Goal raggiunto | 3.53 | 3.53 | 0.48 | 6.2 | 699 |
| 9 | Timeout | 3.47 | 19.57 | 4.37 | 30.0 | 3476 |

Tempo, percorso ed energia includono l’allineamento automatico. A–B diretto è la distanza euclidea iniziale, non un percorso ottimo calcolato sul terreno. Il raggio di raggiungimento è 0,55 m: non occorre arrivare al centro del goal.

## Lettura dei risultati

I successi sono negli episodi 1, 4, 7 e 8. L’episodio 8 completa una missione con distanza iniziale di 3,53 m percorrendo 3,53 m in 6,2 s e consumando circa 699 J. Negli altri tre successi il percorso è più lungo: 10,13 m, 16,75 m e 13,33 m, a fronte di distanze dirette di 4,86 m, 5,14 m e 4,68 m.

I timeout 2 e 9 terminano più lontano dal goal rispetto alla partenza, dopo percorsi rispettivamente di 22,80 m e 19,57 m. Non si tratta semplicemente di un robot lento che si arresta poco prima del goal. L’allineamento automatico riesce sempre; i problemi osservati avvengono durante la navigazione RL. Il risultato resta insufficiente per una navigazione affidabile. Nove episodi descrivono queste missioni e non bastano a stimare con precisione la prestazione generale o a isolare l’effetto della PCA rispetto alla reward.

Solo successi: energia media 2061 J; percorso medio 10.94 m; tempo medio 17.65 s.

## Grafici e verifiche

Start A nero, goal B verde con raggio 0,55 m, orientamento iniziale indicato dalla freccia nera, allineamento arancione, navigazione blu, inizio RL a rombo e posa terminale esatta a croce rossa. Il segmento tratteggiato A–B è un riferimento geometrico, non una traiettoria pianificata. Le scale cambiano tra episodi. I colori del terreno derivano dalla mappa congelata.

`percorsi_test4_300k.pdf` e `.png`: panoramica dei nove percorsi. `episodi_dettaglio.pdf`: una pagina per episodio, con percorso ingrandito e distanza dal goal nel tempo. `riepilogo_episodi.csv`: dati della tabella, coordinate, materiali, tempi/energia separati per allineamento e navigazione.

La posa terminale è aggiunta dai record degli episodi; i campioni di traiettoria sono pre-step. Controllati numero e ordine dei campioni, fasi allineamento/navigazione, goal registrati e distanza finale ricostruita. Nessuna simulazione avviata.

Riproduzione offline (Python di sistema, senza pacchetti utente, per mantenere compatibili NumPy e Matplotlib):

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 spot_rl_energy_navigation/train_general_big_map_test4/analysis/paths_20261027/plot_paths.py
```
