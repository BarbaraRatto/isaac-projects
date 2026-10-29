# Confronto goal: 100k e 300k

Entrambe le policy ricevono esattamente gli stessi input grezzi, da ciascuno dei due replay.
Ogni policy usa la propria normalizzazione salvata. PCA e firma esperimento verificate uguali.
Inferenza deterministica CPU; nessun aggiornamento, simulazione o limite alla rotazione.
Goal modificato a distanza costante; mappa, velocità e azione precedente rimangono invariati.

| Replay | Goal ° (+ sinistra) | vx 100k | vx 300k | wz 100k | wz 300k | Sinistra 100k | Sinistra 300k | Δ wz medio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 100k | original | 0.370 | 0.677 | 0.885 | 1.253 | 83.8% | 93.6% | +0.368 |
| 100k | -150 | 0.298 | 0.311 | 0.822 | 1.492 | 83.0% | 100.0% | +0.669 |
| 100k | -110 | 0.313 | 0.270 | 0.523 | 1.436 | 68.4% | 100.0% | +0.913 |
| 100k | -90 | 0.328 | 0.277 | 0.429 | 1.159 | 63.9% | 97.3% | +0.729 |
| 100k | -60 | 0.357 | 0.411 | 0.402 | 0.580 | 62.7% | 72.5% | +0.177 |
| 100k | -30 | 0.389 | 0.668 | 0.514 | 0.479 | 67.8% | 68.6% | -0.035 |
| 100k | -10 | 0.408 | 0.809 | 0.648 | 0.795 | 74.4% | 80.9% | +0.147 |
| 100k | 0 | 0.415 | 0.859 | 0.725 | 1.044 | 79.1% | 92.0% | +0.319 |
| 100k | 10 | 0.419 | 0.894 | 0.804 | 1.259 | 82.6% | 98.4% | +0.455 |
| 100k | 30 | 0.420 | 0.916 | 0.954 | 1.440 | 87.5% | 100.0% | +0.487 |
| 100k | 60 | 0.402 | 0.889 | 1.119 | 1.484 | 94.7% | 100.0% | +0.365 |
| 100k | 90 | 0.370 | 0.795 | 1.200 | 1.490 | 95.7% | 100.0% | +0.290 |
| 100k | 110 | 0.348 | 0.696 | 1.215 | 1.490 | 96.1% | 100.0% | +0.275 |
| 100k | 150 | 0.313 | 0.466 | 1.166 | 1.491 | 95.1% | 100.0% | +0.325 |
| 100k | 180 | 0.299 | 0.352 | 1.037 | 1.491 | 92.0% | 100.0% | +0.454 |
| 300k | original | 0.233 | 0.718 | 0.831 | 1.229 | 88.9% | 95.7% | +0.398 |
| 300k | -150 | 0.176 | 0.292 | 0.795 | 1.493 | 85.7% | 100.0% | +0.698 |
| 300k | -110 | 0.187 | 0.269 | 0.483 | 1.451 | 66.6% | 100.0% | +0.969 |
| 300k | -90 | 0.197 | 0.293 | 0.386 | 1.196 | 61.1% | 99.6% | +0.810 |
| 300k | -60 | 0.218 | 0.489 | 0.355 | 0.590 | 59.2% | 72.3% | +0.235 |
| 300k | -30 | 0.240 | 0.864 | 0.465 | 0.486 | 65.0% | 68.2% | +0.022 |
| 300k | -10 | 0.252 | 1.023 | 0.605 | 0.853 | 73.4% | 84.8% | +0.248 |
| 300k | 0 | 0.257 | 1.064 | 0.688 | 1.119 | 79.1% | 94.1% | +0.431 |
| 300k | 10 | 0.260 | 1.083 | 0.773 | 1.321 | 84.4% | 99.6% | +0.547 |
| 300k | 30 | 0.260 | 1.063 | 0.936 | 1.455 | 91.6% | 100.0% | +0.520 |
| 300k | 60 | 0.248 | 0.973 | 1.113 | 1.487 | 97.3% | 100.0% | +0.374 |
| 300k | 90 | 0.227 | 0.828 | 1.199 | 1.491 | 99.6% | 100.0% | +0.293 |
| 300k | 110 | 0.212 | 0.701 | 1.215 | 1.492 | 99.8% | 100.0% | +0.277 |
| 300k | 150 | 0.188 | 0.452 | 1.159 | 1.492 | 98.6% | 100.0% | +0.333 |
| 300k | 180 | 0.178 | 0.337 | 1.021 | 1.493 | 94.1% | 100.0% | +0.472 |

## Limiti

vx in m/s, wz in rad/s. Sinistra = wz > 0.1 rad/s; Δ = 300k meno 100k sullo stesso campione.
I replay contengono esperienza di training, non gli episodi visualizzati. Campioni temporalmente correlati.
Le due sorgenti non hanno necessariamente la stessa distribuzione di missioni, distanza, terreno o stato.
Una differenza fra policy sugli stessi input riguarda la policy completa, normalizzazione inclusa.
Non isola da sola reward, ottimizzatore, curriculum o normalizzazione come causa del cambiamento.
Goal artificiali e stati del replay tardivo possono essere fuori distribuzione, soprattutto per il modello 100k.
Girare verso il goal non è sempre corretto: possono esserci ostacoli o terreni costosi.
Questi comandi non misurano successo o efficienza futura; servirà una verifica in simulazione.
