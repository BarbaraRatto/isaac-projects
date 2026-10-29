# Test 5 — risultati a 200.000 transizioni

Fonte: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test5/results/evaluation_20261028_114806_451932`
Checkpoint: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test5/runs/general_map_test5_ppo_20261028_012808_511154/policy_final.pt`

Fase 1: terreni misti 3–6 m senza scale/ostacoli. Seed 20000; 1 robot. Successi 0/9; 8 timeout e 1 caduta. Allineamento riuscito in tutti i 9 episodi.

| Episodio | Esito | A–B diretto (m) | Percorso (m) | Distanza finale (m) | Tempo (s) | Energia (J) |
|---|---|---:|---:|---:|---:|---:|
| 1 | Timeout | 4.86 | 17.16 | 4.19 | 34.6 | 3738 |
| 2 | Timeout | 5.29 | 19.33 | 12.38 | 36.2 | 4126 |
| 3 | Timeout | 4.40 | 14.52 | 3.73 | 32.8 | 3521 |
| 4 | Timeout | 5.14 | 16.10 | 7.91 | 35.6 | 3901 |
| 5 | Timeout | 5.99 | 15.96 | 7.46 | 39.0 | 4706 |
| 6 | Timeout | 5.69 | 20.37 | 7.17 | 37.8 | 4184 |
| 7 | Timeout | 4.68 | 10.65 | 7.12 | 33.8 | 3608 |
| 8 | Caduta | 3.53 | 4.32 | 1.51 | 10.4 | 1748 |
| 9 | Timeout | 3.47 | 7.34 | 3.64 | 30.0 | 3171 |

Tempo, percorso ed energia sono totali, incluso l’allineamento iniziale. A–B diretto è la distanza euclidea, non un percorso ottimo sul terreno. Il goal è raggiunto entro 0,55 m.

## Lettura dei risultati

Gli episodi 2, 4, 5, 6, 7, 9 terminano più lontano dal goal rispetto alla partenza. Negli otto timeout Spot percorre da 7,34 a 20,37 m senza raggiungere goal inizialmente distanti 3,47–5,99 m. Queste missioni indicano un problema di navigazione; il timeout è l’esito, ma non dimostra che serva soltanto più tempo.

Nell’episodio 8 il criterio di inclinazione conferma il ribaltamento: massimo 165,12°, permanenza oltre 90° di 1,04 s. Il contatto continuo massimo del corpo dura 0,205 s; la causa registrata è quindi tilt, non body_contact. Negli altri episodi l’inclinazione massima resta sotto 14°.

Nell’episodio 6 è registrato un campione con misura dell’altezza non valida: il vecchio criterio avrebbe richiesto la terminazione in quel punto, mentre il nuovo ha consentito di continuare. Questo non permette di ricostruire cosa sarebbe successo con una policy allenata diversamente.

Questo test di nove missioni ha risultati insufficienti, ma da solo non identifica la causa né isola l’effetto delle singole modifiche al training.

## Grafici

- `percorsi_test5_200k.pdf` e `.png`: nove percorsi con start A nero, goal B verde e raggio di successo, allineamento arancione, navigazione blu, punto finale a croce rossa. Freccia nera: orientamento iniziale; rombo arancione: inizio RL. Scale diverse per episodio.
- `episodi_dettaglio.pdf`: nove pagine con percorso e distanza dal goal nel tempo.
- `diagnostica_cadute.pdf` e `caduta_episodio_8.png`: inclinazione, durata delle condizioni di caduta e altezza della base.
- `riepilogo_episodi.csv`: tabella con coordinate, terreni, energia e tempi separati, diagnostica terminale.

I terreni sono quelli della mappa congelata del test 5. Il segmento tratteggiato A–B è solo un riferimento geometrico. La posa terminale esatta è aggiunta al CSV delle traiettorie, che registra campioni prima del passo. Verificati conteggi, ordine, fasi, goal e distanze finali; i campioni diagnostici dopo ogni passo coincidono con le pose successive delle traiettorie. Nessuna simulazione avviata.

Riproduzione offline:

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 spot_rl_energy_navigation/train_general_big_map_test5/analysis/paths_20261028/plot_paths.py
```
