# Valutazione full grid: 20 episodi

Fonte: `full_grid_eval_20261006_220553.csv`. Checkpoint: `policy_final.zip`. Policy deterministica, un solo Spot, DINO in tempo reale. Il test non aggiorna i pesi.

| Misura | Tutti i 20 episodi | Soli 19 arrivi |
|---|---:|---:|
| Goal raggiunti | 19/20 (95%) | 19/19 |
| Energia media | 967.48 J | 979.02 J |
| Deviazione standard campionaria energia | 103.59 J | 92.28 J |
| Percorso medio | 4.147 m | 4.257 m |
| Energia per metro media | 236.35 J/m | 229.67 J/m |
| Decisioni medie | 15.90 | 16.32 |
| Episodi con contatto su rocce | 20/20 | 19/19 |
| Episodi con contatto su asfalto | 9/20 | 8/19 |
| Episodi con contatto su ostacoli | 19/20 | 19/19 |

Un episodio si è fermato per `unsafe` dopo 2,060 m e 748,21 J. Per confrontare l'energia di percorsi completati, usa la colonna **soli arrivi**.

| Tra i soli arrivi | Contatto con asfalto (n=8) | Nessun contatto con asfalto (n=11) |
|---|---:|---:|
| Energia media | 1013.06 J | 954.27 J |
| Percorso medio | 4.362 m | 4.181 m |
| Energia per metro media | 231.92 J/m | 228.04 J/m |

**Interpretazione:** tutti gli episodi hanno toccato le rocce. Il contatto con l'asfalto non basta a stabilire che Spot abbia scelto una deviazione sull'asfalto. I due gruppi sono descrittivi: non dimostrano quale percorso consumi meno energia.

Training: 32 robot, 102.400 transizioni, 5577.2 s (93.0 min), 3427/4032 arrivi cumulativi durante il training. Quest'ultimo rapporto non è il risultato dei 20 episodi di valutazione.
