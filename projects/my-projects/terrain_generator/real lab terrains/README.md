# Real lab terrains — grigio chiaro e verde erba chiaro

Due superfici rigide e piane, senza texture esterne, rilievi o penalità legate
al colore. Entrambi usano l'attrito dell'asfalto dei vecchi test. Sul verde
gli script applicano una resistenza orizzontale alla base, proporzionale alla
velocità reale: F_xy = -90 * v_xy. Il maggiore lavoro ai giunti va verificato
prima del training: non viene sommato alcun costo artificiale alla misura.

| Proprietà | Grigio | Verde |
|---|---:|---:|
| Attrito statico | 1,0 | 1,0 |
| Attrito dinamico | 0,8 | 0,8 |
| Restituzione | 0 | 0 |
| Rigidità del contatto | Rigido (0) | Rigido (0) |
| Smorzamento del contatto | 0 | 0 |
| Resistenza orizzontale aggiuntiva | 0 | 90 N·s/m |
| Quota della superficie | 0 m | 0 m |
| Irregolarità geometriche | Nessuna | Nessuna |

Friction/restitution combine mode: `average`, come i vecchi materiali. Il
coefficiente effettivo del contatto dipende anche dal materiale dei piedi e
dalla sua combine mode. Robot e controllore non vengono modificati.

## Asset già generati

- `maps/grey_tile.usda`, `maps/green_tile.usda`: celle elementari 3×3 m.
- `maps/comparison.usda`: piccola mappa 12×6 m, due fasce affiancate larghe 3 m.
  Grigio in Y∈[−3,0], verde in Y∈[0,3]; X∈[−2,10].
- `maps/grey_flat.usda`, `maps/green_flat.usda`: aree uniformi identiche 12×6 m,
  usate per misurare ogni materiale lontano dalla separazione tra i colori.
- `maps/manifest.json`: impostazioni, hash degli asset e dimensioni.

La mappa di training 15×15 m, con 25 celle di cui 8 verdi, non è ancora
generata. Resta prevista con un misto di verdi isolati e adiacenti. Questi
asset più piccoli servono prima a verificare locomozione e consumo.

La generazione usa solo USD, senza avviare Isaac Sim:

```bash
bash "terrain_generator/real lab terrains/generate.sh"
```

## Guida manuale

Dal progetto, per vedere entrambi i colori:

```bash
bash "terrain_generator/real lab terrains/manual.sh"
```

Spot parte sul grigio e usa il controllore di locomozione già impiegato nelle
prove manuali. Cliccare il viewport. W/S o frecce su/giù: avanti/indietro;
A/D o frecce sinistra/destra: rotazione; R: riposizionamento; ESC: uscita.
Rilasciando i tasti il comando di velocità diventa zero. La misura è mostrata
nel terminale: SPAZIO avvia un nuovo intervallo, SPAZIO lo termina. Include
anche il tempo fermo tra le due pressioni; questa modalità è esplorativa.

Per guidare su un solo materiale, in un'area più larga:

```bash
bash "terrain_generator/real lab terrains/manual.sh" grey --speed 0.9
bash "terrain_generator/real lab terrains/manual.sh" green --speed 0.9
```

Eseguire un comando alla volta. Lo stesso `--speed` imposta lo stesso comando
longitudinale, ma la velocità fisica può differire per effetto dello scivolamento.

## Confronto automatico del consumo

Prima prova: cinque attraversamenti per materiale a 0,9 m/s comandati:

```bash
bash "terrain_generator/real lab terrains/measure_energy.sh" --speeds 0.9 --repeats 5
```

Lo script si occupa della guida: non premere i tasti per comandare il robot.
Alterna l'ordine dei materiali, ripristina ogni volta robot e memoria del
controllore, aspetta 2 s di assestamento, lascia 1 m di rincorsa e misura i
successivi 6 m di avanzamento. Entrambe le aree sono geometricamente identiche.
Non utilizza una policy RL, DINO o una reward. Ogni passaggio ha un timeout;
inclinazione persistente, altezza persistentemente bassa o uscita dall'area
rendono il tentativo non valido per la media energetica, ma restano nei log.

Per provare più velocità, oppure senza finestra:

```bash
bash "terrain_generator/real lab terrains/measure_energy.sh" --speeds 0.6 0.9 1.2 --repeats 5
bash "terrain_generator/real lab terrains/measure_energy.sh" --speeds 0.9 --repeats 5 --headless
```

La potenza è `P = Σ |τ_i · q̇_i|`, usando gli sforzi articolari misurati dal
simulatore. L'energia è integrata ogni passo fisico (0,005 s), con trapezi e
interpolazione ai due traguardi. Sono esclusi assestamento e rincorsa, ma sono
incluse tutte le inefficienze durante il tratto misurato. È una misura di
lavoro meccanico assoluto dei giunti, non l'energia elettrica della batteria.

I file vengono salvati automaticamente in `results/energy_.../`:

- `REPORT.md`: risultati leggibili, con J/m, velocità effettive e completamenti.
- `comparison.json`: media, dispersione e differenze nelle coppie riuscite.
- `trials.csv`: energia, tempo, distanze e motivo di conclusione di ogni prova.
- `samples.csv`: tracce fisiche di posizione, velocità e potenza.
- `config.json` e copie degli script: impostazioni e identità degli asset.

La misura principale usa J per metro di avanzamento sul medesimo tratto;
è disponibile anche J per metro del percorso XY realmente effettuato.
Un delta `verde − grigio` positivo indica maggiore costo meccanico nei tratti
riusciti. Verificare anche stabilità e velocità reale: se il verde fa cadere
Spot, o lo rallenta molto, la sola media dei successi non descrive tutto.
Le ripetizioni controllate possono essere quasi deterministiche e non vanno
considerate automaticamente campioni statisticamente indipendenti.

## Regolare la resistenza del verde

Il parametro `horizontal_drag_ns_per_m` in `materials.json` vale 90 sul verde
ed è assente (zero) sul grigio. Dopo ogni modifica rigenerare con `generate.sh`.
I contatti sono tornati rigidi e l'attrito verde è 1,0/0,8 come il grigio.
A 0,9 m/s effettivi la forza è 81 N e la potenza resistente è 72,9 W;
questi valori non sono sommati all'energia misurata dai giunti.

`surface_drag.py` applica F_xy = -c*v_xy in coordinate mondo, alla base
(body index 0) al centro di massa, senza forza verticale o coppia aggiunta.
L'attivazione dipende dalla proiezione XY della base dentro una cella verde,
non dal materiale sotto ogni singola zampa. A cavallo del bordo è quindi
un'approssimazione con cambio netto di coefficiente. La forza viene ricalcolata
ogni passo fisico e azzerata fuori dalle regioni verdi o a velocità nulla.

IMPORTANTE: questa resistenza è un modello implementato dagli script,
non una proprietà del solo USD. Aprire l'asset in un altro programma o usare
uno script senza `SurfaceDrag` non applica il carico. `manual.sh` passa
`--surface-drag-manifest` allo script manuale; il confronto automatico usa
la stessa implementazione, con le traslazioni delle due aree di test.
Per un futuro training occorrerà integrare lo stesso modello nell'ambiente.
Gli altri test manuali, senza questa opzione, non cambiano comportamento.

`samples.csv` include `drag_fx_n`, `drag_fy_n` e
`drag_dissipation_w_before_step` per verificare il carico pre-step, separato
dalla potenza ai giunti campionata dopo lo step. Non si deforma la superficie.
Il maggiore consumo non è garantito in proporzione al carico: verificarne
l'entità insieme a velocità e stabilità. È una resistenza sintetica, non un
modello calibrato di tappeto reale o terreno deformabile.

I risultati archiviati conservano i parametri usati: non mischiare misure
ottenute con coefficienti diversi.

Le proprietà e la geometria degli USD e l'integrazione energetica sono
verificabili offline. La locomozione e la differenza reale di Joule richiedono
l'esecuzione del comando di misura sulla macchina con Isaac Sim.
