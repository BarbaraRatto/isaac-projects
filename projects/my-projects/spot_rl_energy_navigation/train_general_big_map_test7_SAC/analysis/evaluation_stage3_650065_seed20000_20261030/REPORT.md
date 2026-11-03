# Test7 SAC · fase 3 · checkpoint 650065 transizioni

Checkpoint: `/root/spot-training-20261029/spot-cloud-training/spot_rl_energy_navigation/train_general_big_map_test7_SAC/runs/general_map_test7_sac_20261030_003829_628341/policy_best_stage_3.pt`

Esiti: **6/10 successi**. {'goal': 6, 'fallen': 3, 'out_of_bounds': 0, 'timeout': 1, 'alignment_timeout': 0}
Allineamento riuscito: 100%. Energia media sui soli successi: 1723.3 J.

| Episodio | Esito | Percorso (m) | Tempo (s) | Energia (J) | Distanza finale (m) | Cella finale |
|---|---|---:|---:|---:|---:|---|
| 1 | Successo | 6.76 | 7.2 | 1290 | 0.38 | Ostacoli |
| 2 | Timeout | 7.65 | 57.2 | 5372 | 4.51 | Scale |
| 3 | Successo | 7.30 | 6.8 | 1505 | 0.42 | Rocce |
| 4 | Successo | 11.75 | 12.8 | 2523 | 0.49 | Rocce |
| 5 | Successo | 8.13 | 9.4 | 1565 | 0.40 | Rampe |
| 6 | Caduta | 6.16 | 10.2 | 2483 | 5.12 | Scale |
| 7 | Caduta | 8.84 | 9.0 | 2369 | 4.45 | Ghiaccio |
| 8 | Successo | 6.97 | 9.2 | 1645 | 0.37 | Rampe |
| 9 | Caduta | 3.37 | 8.0 | 1955 | 7.88 | Scale |
| 10 | Successo | 8.53 | 10.2 | 1812 | 0.37 | Ghiaia fine |

## Lettura e limiti

- A = partenza; B = obiettivo; croce = posizione finale. Arancione = allineamento; blu = navigazione nei pannelli separati.
- Lo sfondo mostra le celle dei materiali, non le singole rocce/scale/mesh. La cella del punto finale non dimostra la causa della caduta.
- Il tratto finale è ricostruito da summary.json: trajectories.csv contiene campioni prima del passo; fall_diagnostics.csv dopo il passo, prima del reset.
- Verificati: conteggio e ordine dei campioni, corrispondenza pre/post-step, fasi, obiettivi, distanza finale, lunghezza del percorso e terreni di partenza/arrivo.
- Le curve di inclinazione sono campionate ogni 0,2 s; i massimi dei record includono i sottopassi fisici. Un picco isolato non equivale alla condizione di caduta persistente.
- Le durate sono tempo simulato, non tempo reale di esecuzione. Energia e percorso includono l’allineamento.
- Il rapporto percorso/distanza iniziale è solo un riferimento geometrico: la retta può attraversare ostacoli e il goal ha raggio 0,55 m.
- Dieci missioni e un solo seed non bastano per concludere che una policy sia migliore di un’altra. Confrontare stessi seed, fase e missioni; aumentare il numero di episodi.
- Le missioni non hanno una verifica preventiva di raggiungibilità; un fallimento non prova da solo un difetto della policy.

## Osservazioni sui fallimenti

- E2: timeout, distanza residua 4.51 m, ultimi 10 s percorsi 0.01 m, comando avanti medio negli ultimi 10 s 1.58 m/s, massimo tilt 84.7°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E6: causa registrata `tilt`, massimo tilt 167.4°; prima inclinazione campionata ≥90° a 9.200000000000001 s nella cella Scale; fine in Scale.
- E7: causa registrata `tilt`, massimo tilt 180.0°; prima inclinazione campionata ≥90° a 8.0 s nella cella Ghiaccio; fine in Ghiaccio.
- E9: causa registrata `tilt`, massimo tilt 170.3°; prima inclinazione campionata ≥90° a 7.0 s nella cella Scale; fine in Scale.
