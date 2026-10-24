# Confronto policy su missioni identiche

A = partenza; B = goal; cerchio verde = raggio di successo 0,55 m.
Le curve sulla mappa terminano all’ultimo campione pre-step: manca la posizione dopo l’ultimo passo (0,2 s). La distanza terminale e le metriche episodiche provengono invece dal riepilogo esatto.
Il minimo riportato è quello fra campioni registrati e distanza terminale, non un minimo continuo.
I colori di fondo indicano le celle/materiali, non la quota o gli ostacoli interni. La linea A–B è un riferimento geometrico, non un percorso certificato attraversabile.

| Policy | Successi | Cadute | Fuori mappa | Timeout |
|---|---:|---:|---:|---:|
| 53248 | 0/9 | 5 | 3 | 1 |
| 253952 | 0/9 | 5 | 3 | 1 |
| 450560 | 0/9 | 3 | 2 | 4 |

Le missioni 1, 4, 7 appartengono alla fascia 3–6 m allenata. Le altre due fasce non erano state raggiunte dal curriculum.
Non confrontare l’efficienza energetica dei fallimenti come se fossero missioni completate.

## Sorgenti
- 53248: spot_rl_energy_navigation/train_general_big_map_test1/results/evaluation_20261024_170952_352059
- 253952: spot_rl_energy_navigation/train_general_big_map_test1/results/evaluation_20261024_171747_921095
- 450560: spot_rl_energy_navigation/train_general_big_map_test1/results/evaluation_20261024_172414_678775
