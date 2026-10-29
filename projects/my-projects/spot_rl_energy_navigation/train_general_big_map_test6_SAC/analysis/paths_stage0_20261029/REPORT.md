# Test 6 SAC — fase 0 — 100.000 transizioni

Fonte: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test6_SAC/results/evaluation_20261029_000206_503344`
Checkpoint: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test6_SAC/runs/general_map_test6_sac_20261028_201519_133017/policy_final.pt`

Fase 0: asfalto 1.5–3 m. Seed 20000, 1 robot, nove missioni. Successi 7/9 (77.8%), due timeout, zero cadute e zero uscite dalla mappa. Allineamento completato 9/9.

| Episodio | Esito | A–B diretto (m) | Percorso (m) | Distanza finale (m) | Tempo (s) | Energia (J) |
|---|---|---:|---:|---:|---:|---:|
| 1 | Goal raggiunto | 2.00 | 1.82 | 0.44 | 5.0 | 501 |
| 2 | Goal raggiunto | 2.04 | 2.00 | 0.41 | 5.0 | 572 |
| 3 | Goal raggiunto | 2.65 | 2.48 | 0.54 | 5.6 | 643 |
| 4 | Goal raggiunto | 1.75 | 1.57 | 0.39 | 3.4 | 388 |
| 5 | Goal raggiunto | 1.83 | 1.75 | 0.37 | 2.2 | 358 |
| 6 | Timeout | 2.82 | 5.00 | 2.87 | 30.0 | 2778 |
| 7 | Timeout | 1.74 | 7.29 | 1.82 | 30.0 | 2970 |
| 8 | Goal raggiunto | 2.67 | 2.84 | 0.45 | 6.2 | 721 |
| 9 | Goal raggiunto | 1.71 | 4.66 | 0.55 | 15.8 | 1556 |

Tempo, percorso ed energia includono l’allineamento. A–B diretto è la distanza euclidea iniziale. Il goal si considera raggiunto entrando nel raggio di 0,55 m: il percorso può essere inferiore alla distanza tra i due centri.

## Lettura dei risultati

Gli episodi 1–5 e 8 completano missioni brevi con percorsi di lunghezza vicina alla distanza iniziale. L’episodio 9 raggiunge il goal ma percorre 4,66 m a fronte di 1,71 m iniziali e impiega 15,8 s: è un successo con una deviazione significativa.

Gli episodi 6 e 7 terminano dopo 30 s, rispettivamente a 2,87 m e 1,82 m dal goal, leggermente più lontano che alla partenza, dopo aver percorso 5,00 m e 7,29 m. Il mancato completamento non è accompagnato da cadute; non si tratta soltanto di essere arrivati poco prima del goal alla scadenza.

Nei soli sette successi: energia media 677 J, percorso medio 2.44 m, tempo medio 6.17 s (allineamento incluso).

La policy completa una maggioranza di queste missioni semplici, ma non è ancora affidabile neppure nella fase 0. Il risultato non dimostra la superiorità di SAC rispetto ai test PPO precedenti sulla fase 1: qui start e goal sono nella stessa cella di asfalto e distanti 1,5–3 m. Per un confronto tra algoritmi servono missioni della stessa difficoltà.

Il curriculum era passato alla fase 1 solo alla valutazione finale delle 100.000 transizioni; il modello valutato non ha ancora ricevuto training di fase 1. Nove missioni non bastano a stimare con precisione il successo generale né a stabilire la causa dei due timeout.

## File e verifiche

- `percorsi_test6_SAC_fase0_100k.pdf` / `.png`: panoramica; A nero start, B verde goal, arancione allineamento, blu navigazione, croce rossa fine episodio. Freccia nera: orientamento iniziale; rombo arancione: inizio RL.
- `episodi_dettaglio.pdf`: percorso e distanza dal goal nel tempo, una pagina per episodio.
- `riepilogo_metriche.pdf`: confronto della distanza iniziale con il percorso, ed energia per episodio.
- `riepilogo_episodi.csv`: valori, coordinate, terreni e diagnostica terminale.

Scale diverse tra episodi; coordinate della mappa in metri. Il tratteggio A–B è solo un riferimento geometrico. Aggiunta la posizione terminale esatta ai campioni pre-step delle traiettorie. Controllati conteggi, ordine, fasi, coordinate goal, distanza finale e corrispondenza con il CSV diagnostico post-step. Nessuna simulazione avviata.

Riproduzione offline:
```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 spot_rl_energy_navigation/train_general_big_map_test6_SAC/analysis/paths_stage0_20261029/plot_paths.py
```
