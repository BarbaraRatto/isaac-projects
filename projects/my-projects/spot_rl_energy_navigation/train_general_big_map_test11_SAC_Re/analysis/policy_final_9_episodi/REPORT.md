# Test11 SAC Re · Policy final · suite 9 missioni
Checkpoint 280000 transizioni

Checkpoint: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test11_SAC_Re/runs/general_map_test11_sac_re_20261107_024047_872269/policy_final.pt`

Esiti: **1/9 successi**. {'goal': 1, 'fallen': 0, 'out_of_bounds': 0, 'timeout': 8, 'alignment_timeout': 0}
Allineamento riuscito: 100%. Energia media sui soli successi: 504.0 J.

| Episodio | Esito | Percorso (m) | Tempo (s) | Energia (J) | Distanza finale (m) | Cella finale |
|---|---|---:|---:|---:|---:|---|
| 1 | Successo | 1.87 | 4.0 | 504 | 0.49 | Grigio |
| 2 | Timeout | 19.28 | 30.0 | 4220 | 1.32 | Grigio |
| 3 | Timeout | 29.07 | 39.6 | 6008 | 2.64 | Grigio |
| 4 | Timeout | 29.07 | 38.2 | 6093 | 4.83 | Grigio |
| 5 | Timeout | 35.39 | 49.4 | 7292 | 6.39 | Grigio |
| 6 | Timeout | 37.95 | 49.4 | 7814 | 3.02 | Verde resistente |
| 7 | Timeout | 39.26 | 64.6 | 9328 | 4.68 | Grigio |
| 8 | Timeout | 35.32 | 61.6 | 8274 | 7.93 | Grigio |
| 9 | Timeout | 17.95 | 30.0 | 4330 | 1.47 | Verde resistente |

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

- E2: timeout, distanza residua 1.32 m, ultimi 10 s percorsi 5.04 m, comando avanti medio negli ultimi 10 s 0.52 m/s, massimo tilt 6.0°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E3: timeout, distanza residua 2.64 m, ultimi 10 s percorsi 6.14 m, comando avanti medio negli ultimi 10 s 0.58 m/s, massimo tilt 6.0°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E4: timeout, distanza residua 4.83 m, ultimi 10 s percorsi 5.69 m, comando avanti medio negli ultimi 10 s 0.53 m/s, massimo tilt 5.7°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E5: timeout, distanza residua 6.39 m, ultimi 10 s percorsi 4.06 m, comando avanti medio negli ultimi 10 s 0.35 m/s, massimo tilt 5.5°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E6: timeout, distanza residua 3.02 m, ultimi 10 s percorsi 3.94 m, comando avanti medio negli ultimi 10 s 0.50 m/s, massimo tilt 4.6°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E7: timeout, distanza residua 4.68 m, ultimi 10 s percorsi 3.02 m, comando avanti medio negli ultimi 10 s 0.37 m/s, massimo tilt 6.1°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E8: timeout, distanza residua 7.93 m, ultimi 10 s percorsi 3.90 m, comando avanti medio negli ultimi 10 s 0.51 m/s, massimo tilt 5.3°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E9: timeout, distanza residua 1.47 m, ultimi 10 s percorsi 4.01 m, comando avanti medio negli ultimi 10 s 0.78 m/s, massimo tilt 7.0°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
