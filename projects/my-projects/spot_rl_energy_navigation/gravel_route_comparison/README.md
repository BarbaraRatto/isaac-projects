# Confronto energetico: ghiaia della riga 4

Questa prova usa **l'intero** `terrain_generator/real_terrains.usd` e un solo
Spot. Non allena RL, non usa DINO e non richiede un checkpoint. Usa la stessa
simulazione e il controllore locomotorio IsaacRobotics del progetto
`full_grid_energy_choice`, per misurare i joule di tre percorsi guidati.
Non modifica il USD originale.

## Punti e tre vie

Partenza comune `(26, 15)` m, nell'asfalto `Cell_5_4_t1_asphalt`.
Goal comune `(28.5, 9)` m, nella rampa `Cell_3_5_t3_ramp`.
Lo yaw iniziale punta verso il goal; non c'è variazione casuale della
partenza. Il goal ha raggio 0,55 m. Le celle intermedie sono:

```text
                  x in [21,27]             x in [27,33]
riga 5, y 13.5–16.5   asfalto START            ...
riga 4, y 10.5–13.5   rampa                    ghiaia
riga 3, y  7.5–10.5   asfalto                  rampa GOAL
```

| Via | Waypoint dopo lo start | Lunghezza nominale | Ghiaia nominale |
| --- | --- | ---: | ---: |
| `direct` | nessuno | 6,50 m | circa 2,28 m |
| `avoid` | `(26.4, 10.3)` nell'asfalto della riga 3 | 7,19 m | 0 m |
| `partial` | `(26.4, 11.4)` sulla rampa, poi `(27.6, 10.9)` sulla ghiaia | 7,03 m | circa 1,09 m |

Queste sono lunghezze **geometriche** dei segmenti tra i punti. Spot può
percorrere distanze diverse; inoltre termina quando arriva entro il raggio
del goal. Il codice misura per ogni episodio energia meccanica totale,
percorso effettivo e metri percorsi dal **centro del corpo** sulla ghiaia,
sulla rampa della riga 4 e sull'asfalto della riga 3. Verifica che le celle
attese esistano nel USD e abbiano le coordinate previste.

## Comando

Dalla cartella `my-projects`:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/gravel_route_comparison/check_routes.py --headless --episodes 20 --speed 0.8
```

Togli `--headless` per vedere Spot e i percorsi. Il comando crea un CSV in
`results/gravel_routes_<data>.csv` e stampa una sintesi per via. Le tre vie
sono eseguite in serie, con le stesse condizioni iniziali e limite di
velocità. Il tempo di assestamento iniziale non entra nell'energia misurata.

Confronta `mean_energy` solo per gli episodi con `success=1` e
`route_followed=1`. Per `avoid`, il centro del corpo deve percorrere al
massimo 0,15 m sulla ghiaia; per `partial`, fra 0,2 e 1,6 m; per `direct`,
almeno 1 m. Il passaggio dei piedi sul bordo non viene classificato come
percorso del corpo sulla ghiaia. Le misure di distanza del corpo escludono
l'ultimo passo di un episodio, perché l'ambiente riposiziona subito Spot;
la `path_length_m` e l'`energy_j` dell'ambiente includono invece il passo
finale. I risultati diranno quale via costa meno **a 0,8 m/s comandati**;
non presumiamo che evitare la ghiaia faccia risparmiare energia.
