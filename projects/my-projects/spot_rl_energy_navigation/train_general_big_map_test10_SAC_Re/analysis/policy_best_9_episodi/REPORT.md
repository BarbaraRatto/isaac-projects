# Test10 SAC Re · Policy best · suite 9 missioni
Checkpoint 10024 transizioni

Checkpoint: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test10_SAC_Re/runs/general_map_test10_sac_re_20261106_193839_739979/policy_best.pt`

Esiti: **2/9 successi**. {'goal': 2, 'fallen': 0, 'out_of_bounds': 6, 'timeout': 1, 'alignment_timeout': 0}
Allineamento riuscito: 100%. Energia media sui soli successi: 489.6 J.

| Episodio | Esito | Percorso (m) | Tempo (s) | Energia (J) | Distanza finale (m) | Cella finale |
|---|---|---:|---:|---:|---:|---|
| 1 | Successo | 1.57 | 4.8 | 409 | 0.42 | Grigio |
| 2 | Successo | 2.44 | 6.2 | 570 | 0.46 | Grigio |
| 3 | Fuori mappa | 13.36 | 22.2 | 2471 | 7.51 | Grigio |
| 4 | Fuori mappa | 10.77 | 17.2 | 2014 | 4.41 | Grigio |
| 5 | Fuori mappa | 7.36 | 15.0 | 1474 | 3.95 | Verde resistente |
| 6 | Fuori mappa | 15.54 | 26.6 | 2838 | 7.34 | Verde resistente |
| 7 | Fuori mappa | 6.38 | 15.2 | 1378 | 6.37 | Verde resistente |
| 8 | Fuori mappa | 14.98 | 23.2 | 2672 | 5.86 | Verde resistente |
| 9 | Timeout | 15.32 | 30.8 | 2990 | 10.45 | Grigio |

## Lettura e limiti

- A = partenza; B = obiettivo; croce = posizione finale. Arancione = allineamento; blu = navigazione nei pannelli separati.
- Lo sfondo mostra le celle dei materiali, con verde resistente e grigio senza carico. La cella del punto finale non dimostra la causa della caduta.
- Il tratto finale è ricostruito da summary.json: trajectories.csv contiene campioni prima del passo; fall_diagnostics.csv dopo il passo, prima del reset.
- Verificati: conteggio e ordine dei campioni, corrispondenza pre/post-step, fasi, obiettivi, distanza finale, lunghezza del percorso e terreni di partenza/arrivo.
- Le curve di inclinazione sono campionate ogni 0,2 s; i massimi dei record includono i sottopassi fisici. Un picco isolato non equivale alla condizione di caduta persistente.
- Le durate sono tempo simulato, non tempo reale di esecuzione. Energia e percorso includono l’allineamento.
- Il rapporto percorso/distanza iniziale è solo un riferimento geometrico: la retta può attraversare ostacoli e il goal ha raggio 0,55 m.
- Una singola suite e un solo seed non bastano per concludere che una policy sia migliore di un’altra. Confrontare gli stessi start, goal e orientamenti; aumentare il numero di episodi.
- Le missioni non hanno una verifica preventiva di raggiungibilità; un fallimento non prova da solo un difetto della policy.

## Osservazioni sui fallimenti

- E9: timeout, distanza residua 10.45 m, ultimi 10 s percorsi 5.61 m, comando avanti medio negli ultimi 10 s 0.80 m/s, massimo tilt 3.5°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
