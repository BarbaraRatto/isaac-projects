# Test 5 — confronto 110.938 / 200.000 transizioni

Nuova valutazione: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test5/results/evaluation_20261028_124938_174249`
Checkpoint: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test5/runs/general_map_test5_ppo_20261028_012808_511154/policy_110938.pt`
Valutazione finale di confronto: `/home/students/work/barbara/isaac-projects/projects/my-projects/spot_rl_energy_navigation/train_general_big_map_test5/results/evaluation_20261028_114806_451932`

Stessa mappa, fase 1, seed 20000 e un robot. Verificata uguaglianza di start, goal, orientamenti iniziali, deadline e traiettorie di allineamento. Nessuna simulazione o modifica del training effettuata.

## Risultati del checkpoint a 110.938

| Episodio | Esito | A–B diretto (m) | Percorso (m) | Distanza finale (m) | Tempo (s) | Energia (J) |
|---|---|---:|---:|---:|---:|---:|
| 1 | Caduta | 4.86 | 16.28 | 7.34 | 25.8 | 3740 |
| 2 | Timeout | 5.29 | 24.68 | 8.92 | 36.2 | 4338 |
| 3 | Fuori mappa | 4.40 | 13.65 | 7.93 | 21.6 | 2558 |
| 4 | Timeout | 5.14 | 23.61 | 8.29 | 35.6 | 4136 |
| 5 | Goal raggiunto | 5.99 | 5.82 | 0.50 | 8.8 | 1059 |
| 6 | Caduta | 5.69 | 8.67 | 2.87 | 15.6 | 2681 |
| 7 | Timeout | 4.68 | 22.36 | 3.41 | 33.8 | 4005 |
| 8 | Goal raggiunto | 3.53 | 4.28 | 0.50 | 9.0 | 975 |
| 9 | Goal raggiunto | 3.47 | 5.03 | 0.52 | 11.4 | 1138 |

Totale: 3/9 successi (33,3%), 3 timeout, 2 cadute e 1 uscita dalla mappa. Allineamento riuscito 9/9. Tempi, percorso ed energia includono l’allineamento automatico. La distanza A–B è euclidea, non un percorso ottimo calcolato sul terreno. Raggio goal: 0,55 m.

Nei tre successi: percorso medio 5.04 m, energia media 1057 J, durata media 9.73 s.

## Confronto episodio per episodio

| Episodio | Esito a 110.938 | Esito a 200.000 | Distanza finale 110.938 (m) | Distanza finale 200.000 (m) |
|---|---|---|---:|---:|
| 1 | Caduta | Timeout | 7.34 | 4.19 |
| 2 | Timeout | Timeout | 8.92 | 12.38 |
| 3 | Fuori mappa | Timeout | 7.93 | 3.73 |
| 4 | Timeout | Timeout | 8.29 | 7.91 |
| 5 | Goal raggiunto | Timeout | 0.50 | 7.46 |
| 6 | Caduta | Timeout | 2.87 | 7.17 |
| 7 | Timeout | Timeout | 3.41 | 7.12 |
| 8 | Goal raggiunto | Caduta | 0.50 | 1.51 |
| 9 | Goal raggiunto | Timeout | 0.52 | 3.64 |

Le missioni 5, 8 e 9 riescono a 110.938 transizioni e falliscono a 200.000. La missione 5 raggiunge il goal sulle rocce con un percorso di 5,82 m, contro 5,99 m di distanza iniziale: una lunghezza inferiore è possibile perché basta entrare nel raggio del goal. L’episodio 8 raggiunge un goal su rampa in 9 s; il modello finale vi si ribalta. L’episodio 9 raggiunge il goal in 11,4 s, mentre quello finale resta vicino allo start compiendo giri.

La policy intermedia resta inaffidabile: negli episodi 2 e 4 termina più lontano dal goal dopo circa 24 m; nell’episodio 7 percorre 22,36 m senza arrivare. Nell’episodio 3 esce dalla mappa. Non aveva già risolto la navigazione generale.

## Cadute

- Episodio 1: causa `tilt`, inclinazione massima 180.00°, permanenza massima oltre 90° 1.06 s. Coordinate terminali (32.29, 10.45) m. Le coordinate della base non identificano da sole il terreno di contatto delle zampe.
- Episodio 6: causa `tilt`, inclinazione massima 180.00°, permanenza massima oltre 90° 1.16 s. Coordinate terminali (29.57, 17.38) m. Le coordinate della base non identificano da sole il terreno di contatto delle zampe.

## Conclusione e limiti

Su queste nove missioni il checkpoint intermedio è migliore per raggiungimento del goal, ma ha più cadute e uscite dalla mappa. Il peggioramento finale non si spiega soltanto con un numero insufficiente di transizioni: durante questo allenamento sono andati persi alcuni comportamenti utili. Anche le valutazioni interne della fase 1 erano passate da 12/32 successi a 110.938 a 7/32 a 200.000.

Queste osservazioni non identificano la causa del peggioramento: non dimostrano che dipenda dalla penalità di orientamento, dalla capacità della rete o dalle soglie di caduta. Una sola run e nove missioni non stimano con precisione la prestazione generale. L’energia di episodi falliti e riusciti non è un confronto di efficienza energetica a parità di task completata.

Conservare il checkpoint intermedio come riferimento. Prima di prolungare o modificare ancora i pesi, analizzare gli aggiornamenti PPO e la distribuzione delle missioni nel tratto 110.938–200.000; una successiva prova controllata dovrebbe modificare un solo fattore.

## File

- `percorsi_test5_110938.pdf` / `.png`: start A nero, goal B verde, allineamento arancione, navigazione blu, posizione finale rossa.
- `episodi_dettaglio.pdf`: percorso e distanza dal goal per ogni episodio.
- `confronto_percorsi.pdf` / `.png`: traiettorie sovrapposte; blu 110.938, viola tratteggiato 200.000.
- `confronto_distanze.pdf`: distanza dal goal nel tempo per entrambi i checkpoint.
- `diagnostica_cadute.pdf`: inclinazione e persistenza nelle due cadute.
- `riepilogo_episodi.csv` e `confronto_episodi.csv`: dati numerici.

Verificati numero e ordine dei campioni, fasi, goal, pose terminali e corrispondenza dei log diagnostici post-step.

Riproduzione:
```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 spot_rl_energy_navigation/train_general_big_map_test5/analysis/comparison_110938_200000/plot_paths.py
```
