# Test14 SAC Re · Policy best — Test14 SAC Re · suite 9 missioni
Checkpoint 590705 transizioni

Checkpoint: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test14_SAC_Re/runs/general_map_test14_sac_re_20261109_121123_675209/policy_best.pt`

Esiti: **4/9 successi**. {'goal': 4, 'fallen': 5, 'out_of_bounds': 0, 'timeout': 0, 'alignment_timeout': 0}
Allineamento riuscito: 100%. Energia media sui soli successi: 873.3 J.

| Episodio | Esito | Percorso (m) | Tempo (s) | Energia (J) | Distanza finale (m) | Cella finale |
|---|---|---:|---:|---:|---:|---|
| 1 | Successo | 2.36 | 4.6 | 585 | 0.51 | Grigio |
| 2 | Successo | 2.96 | 4.4 | 609 | 0.55 | Grigio |
| 3 | Successo | 5.76 | 6.6 | 1122 | 0.54 | Grigio |
| 4 | Successo | 6.03 | 7.2 | 1178 | 0.51 | Grigio |
| 5 | Caduta | 10.70 | 13.2 | 2695 | 0.84 | Nero scivoloso |
| 6 | Caduta | 4.56 | 6.0 | 1784 | 6.16 | Nero scivoloso |
| 7 | Caduta | 8.40 | 7.2 | 2539 | 5.10 | Nero scivoloso |
| 8 | Caduta | 4.48 | 5.6 | 1600 | 8.13 | Nero scivoloso |
| 9 | Caduta | 3.29 | 6.8 | 1365 | 1.26 | Nero scivoloso |

## Lettura e limiti

- A = partenza; B = obiettivo; croce = posizione finale. Arancione = allineamento; blu = navigazione nei pannelli separati.
- Lo sfondo mostra le celle dei materiali, con nero scivoloso e grigio ad alto attrito. La cella del punto finale non dimostra la causa della caduta.
- Il tratto finale è ricostruito da summary.json: trajectories.csv contiene campioni prima del passo; fall_diagnostics.csv dopo il passo, prima del reset.
- Verificati: conteggio e ordine dei campioni, corrispondenza pre/post-step, fasi, obiettivi, distanza finale, lunghezza del percorso e terreni di partenza/arrivo.
- Le curve di inclinazione sono campionate ogni 0,2 s; i massimi dei record includono i sottopassi fisici. Un picco isolato non equivale alla condizione di caduta persistente.
- Le durate sono tempo simulato, non tempo reale di esecuzione. Energia e percorso includono l’allineamento.
- Il rapporto percorso/distanza iniziale è solo un riferimento geometrico: la retta può attraversare ostacoli e il goal ha raggio 0,55 m.
- Una singola suite e un solo seed non bastano per concludere che una policy sia migliore di un’altra. Confrontare gli stessi start, goal e orientamenti; aumentare il numero di episodi.
- Le missioni non hanno una verifica preventiva di raggiungibilità; un fallimento non prova da solo un difetto della policy.

## Osservazioni sui fallimenti

- E5: causa registrata `tilt`, massimo tilt 180.0°; prima inclinazione campionata ≥90° a 12.200000000000001 s nella cella Grigio; fine in Nero scivoloso.
- E6: causa registrata `tilt`, massimo tilt 177.4°; prima inclinazione campionata ≥90° a 5.0 s nella cella Nero scivoloso; fine in Nero scivoloso.
- E7: causa registrata `tilt`, massimo tilt 180.0°; prima inclinazione campionata ≥90° a 6.2 s nella cella Nero scivoloso; fine in Nero scivoloso.
- E8: causa registrata `tilt`, massimo tilt 180.0°; prima inclinazione campionata ≥90° a 4.6000000000000005 s nella cella Nero scivoloso; fine in Nero scivoloso.
- E9: causa registrata `tilt`, massimo tilt 180.0°; prima inclinazione campionata ≥90° a 5.800000000000001 s nella cella Nero scivoloso; fine in Nero scivoloso.
