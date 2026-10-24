# Confronto energetico guidato sulle nuove mappe

Benchmark fisico senza training RL, camera o DINO. Usa le mappe correnti di
`terrain_generator/manual_asphalt_rocks_10cm`, copiandole nella directory del
risultato prima dell'avvio. Non usa le mappe congelate del test 11.

```bash
TERM=xterm /home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/energy_route_benchmark/compare.py \
  --headless --device cuda:0 --repeats 10 --speed 0.8
```

40 episodi: 10 ripetizioni × 2 mappe × 2 percorsi. Un solo Spot viene
riposizionato fra le due mappe. Controller locomotorio, PD e pesi sono quelli
del progetto (48 osservazioni, locomozione 20 Hz, fisica 200 Hz). Comandi
guidati di navigazione a 5 Hz; non si modifica una policy.

Start (0,0), goal (4,4), yaw iniziale 45°. Per ogni ripetizione il jitter XY
±5 cm è comune ai quattro casi; l'ordine dei quattro casi è randomizzato.
Assestamento di 2 s escluso dalla misura. Gli stati iniziali dei giunti sono
riportati al default a ogni prova. La velocità residua dopo l'assestamento
viene registrata. Il benchmark non replica il reset con stato assestato
congelato del training: le condizioni sono abbinate fra le prove del benchmark.

Percorso destro: (2,0), (4,0), (4,2), (4,4). Sinistro: (0,2), (0,4), (2,4),
(4,4). Waypoint intermedi entro 25 cm; goal entro 55 cm. Velocità comandata
limitata dal parametro `--speed`, modulata dal coseno dell'errore di heading;
il robot ruota sul posto per errori >0,6 rad. Wz limitata a ±1,5 rad/s.
La lunghezza effettiva non è imposta: viene misurata e confrontata nei risultati.
Queste traiettorie guidate non sono percorsi energeticamente ottimi.

Terminazioni: goal, caduta (clearance <25 cm o gravità proiettata z >−0,5),
uscita dal bordo di sicurezza, deadline 40 s. Clearance misurata via raycast
sulle sole mesh del terreno, come nel test11. Nessun reset automatico prima
di registrare energia e posizione finale.

Energia assoluta meccanica integrata su tutti gli intervalli fisici tramite
trapezi: `sum(abs(tau_physx * joint_velocity))`. Si salvano anche energia
scontata con gamma 0,999 e reward di navigazione con i coefficienti di test11.
Soste e svolte durante il tragitto sono incluse. Non è energia elettrica.

`route_followed` richiede tutti i waypoint, almeno 3 m sul ramo richiesto e
meno di 15 cm su ghiaccio o ramo opposto. È classificazione della base, non
dei contatti delle zampe. Solo episodi riusciti e aderenti al percorso
entrano nel confronto energetico; i fallimenti restano nel CSV.

Risultati: `episodes.csv`, `steps.csv`, `config.json` con hash delle mappe,
copie USD e codice; `complete.json` solo a fine benchmark.

```bash
python3 spot_rl_energy_navigation/energy_route_benchmark/analyze.py \
  spot_rl_energy_navigation/energy_route_benchmark/results/<run>
```

L'analisi salva `analysis.json`: differenze abbinate rocce−asfalto per mappa,
media bilanciata sulle due mappe, bootstrap per ripetizione e pesi teorici
per vantaggi di 10/25/100 punti. Il peso non va scelto dalla sola differenza:
valutare anche la penalità sull'energia totale e la preferenza per successo
rispetto a fallimento anticipato. Le misure valgono per la velocità,
il controller, le due realizzazioni del terreno e le traiettorie provate.
