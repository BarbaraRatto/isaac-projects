# Test10 SAC Re · Policy final · suite 15 missioni
Checkpoint 100000 transizioni

Checkpoint: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test10_SAC_Re/runs/general_map_test10_sac_re_20261106_193839_739979/policy_final.pt`

Esiti: **2/15 successi**. {'goal': 2, 'fallen': 0, 'out_of_bounds': 0, 'timeout': 13, 'alignment_timeout': 0}
Allineamento riuscito: 100%. Energia media sui soli successi: 432.2 J.

| Episodio | Esito | Percorso (m) | Tempo (s) | Energia (J) | Distanza finale (m) | Cella finale |
|---|---|---:|---:|---:|---:|---|
| 1 | Successo | 1.66 | 4.4 | 487 | 0.49 | Grigio |
| 2 | Timeout | 20.49 | 30.0 | 3997 | 2.23 | Grigio |
| 3 | Timeout | 19.41 | 30.0 | 3878 | 3.35 | Grigio |
| 4 | Successo | 1.44 | 2.4 | 377 | 0.45 | Grigio |
| 5 | Timeout | 27.17 | 40.8 | 5511 | 3.95 | Grigio |
| 6 | Timeout | 25.05 | 40.2 | 5277 | 5.09 | Grigio |
| 7 | Timeout | 17.88 | 41.0 | 4607 | 3.97 | Verde resistente |
| 8 | Timeout | 32.27 | 40.6 | 5943 | 5.06 | Grigio |
| 9 | Timeout | 36.77 | 51.2 | 7116 | 7.73 | Grigio |
| 10 | Timeout | 27.53 | 50.8 | 6121 | 7.39 | Grigio |
| 11 | Timeout | 22.55 | 51.8 | 5769 | 5.20 | Grigio |
| 12 | Timeout | 27.22 | 62.0 | 7329 | 8.31 | Grigio |
| 13 | Timeout | 34.41 | 61.4 | 7562 | 8.76 | Grigio |
| 14 | Timeout | 25.98 | 62.0 | 6798 | 7.72 | Verde resistente |
| 15 | Timeout | 13.88 | 30.8 | 3541 | 3.32 | Grigio |

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

- E2: timeout, distanza residua 2.23 m, ultimi 10 s percorsi 6.98 m, comando avanti medio negli ultimi 10 s 0.72 m/s, massimo tilt 4.8°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E3: timeout, distanza residua 3.35 m, ultimi 10 s percorsi 5.46 m, comando avanti medio negli ultimi 10 s 0.53 m/s, massimo tilt 6.1°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E5: timeout, distanza residua 3.95 m, ultimi 10 s percorsi 4.62 m, comando avanti medio negli ultimi 10 s 0.47 m/s, massimo tilt 5.3°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E6: timeout, distanza residua 5.09 m, ultimi 10 s percorsi 4.37 m, comando avanti medio negli ultimi 10 s 0.42 m/s, massimo tilt 5.2°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E7: timeout, distanza residua 3.97 m, ultimi 10 s percorsi 5.59 m, comando avanti medio negli ultimi 10 s 0.57 m/s, massimo tilt 4.3°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E8: timeout, distanza residua 5.06 m, ultimi 10 s percorsi 7.86 m, comando avanti medio negli ultimi 10 s 0.76 m/s, massimo tilt 5.1°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E9: timeout, distanza residua 7.73 m, ultimi 10 s percorsi 5.74 m, comando avanti medio negli ultimi 10 s 0.59 m/s, massimo tilt 6.0°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E10: timeout, distanza residua 7.39 m, ultimi 10 s percorsi 3.41 m, comando avanti medio negli ultimi 10 s 0.36 m/s, massimo tilt 5.4°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E11: timeout, distanza residua 5.20 m, ultimi 10 s percorsi 5.64 m, comando avanti medio negli ultimi 10 s 0.64 m/s, massimo tilt 6.7°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E12: timeout, distanza residua 8.31 m, ultimi 10 s percorsi 1.30 m, comando avanti medio negli ultimi 10 s 0.13 m/s, massimo tilt 5.0°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E13: timeout, distanza residua 8.76 m, ultimi 10 s percorsi 4.49 m, comando avanti medio negli ultimi 10 s 0.48 m/s, massimo tilt 5.5°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E14: timeout, distanza residua 7.72 m, ultimi 10 s percorsi 2.44 m, comando avanti medio negli ultimi 10 s 0.46 m/s, massimo tilt 5.3°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
- E15: timeout, distanza residua 3.32 m, ultimi 10 s percorsi 4.18 m, comando avanti medio negli ultimi 10 s 0.44 m/s, massimo tilt 7.9°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.
