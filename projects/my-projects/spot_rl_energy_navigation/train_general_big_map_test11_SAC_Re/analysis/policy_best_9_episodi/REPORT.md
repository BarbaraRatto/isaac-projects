# Test11 SAC Re · Policy best · suite 9 missioni
Checkpoint 20007 transizioni

Checkpoint: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test11_SAC_Re/runs/general_map_test11_sac_re_20261107_005045_107864/policy_best.pt`

Esiti: **2/9 successi**. {'goal': 2, 'fallen': 0, 'out_of_bounds': 7, 'timeout': 0, 'alignment_timeout': 0}
Allineamento riuscito: 100%. Energia media sui soli successi: 468.9 J.

| Episodio | Esito | Percorso (m) | Tempo (s) | Energia (J) | Distanza finale (m) | Cella finale |
|---|---|---:|---:|---:|---:|---|
| 1 | Successo | 1.50 | 4.4 | 371 | 0.45 | Grigio |
| 2 | Successo | 2.33 | 5.6 | 567 | 0.41 | Grigio |
| 3 | Fuori mappa | 6.18 | 10.8 | 1182 | 2.25 | Grigio |
| 4 | Fuori mappa | 14.44 | 23.2 | 2575 | 8.62 | Grigio |
| 5 | Fuori mappa | 11.76 | 26.0 | 2369 | 6.53 | Verde resistente |
| 6 | Fuori mappa | 12.36 | 20.6 | 2208 | 7.31 | Grigio |
| 7 | Fuori mappa | 12.09 | 16.0 | 1993 | 7.20 | Grigio |
| 8 | Fuori mappa | 16.70 | 30.4 | 3112 | 10.87 | Verde resistente |
| 9 | Fuori mappa | 17.50 | 26.8 | 3049 | 13.16 | Grigio |

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

