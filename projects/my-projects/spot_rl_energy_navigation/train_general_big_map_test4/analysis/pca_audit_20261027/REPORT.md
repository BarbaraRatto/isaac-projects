# Provenienza e controllo della PCA — 27 ottobre 2026

> Nota successiva: questo report descrive la PCA storica `artifacts/pca32.npz`. Su richiesta dell’utente, il test 4 ora calibra automaticamente una nuova PCA `artifacts/pca32_big_map.npz` sui sette terreni prima di allenare la policy. Le conclusioni e misure sotto restano riferite al vecchio file.

## Risultato

La PCA del test 3, copiata senza modifiche nel test 4, è **identica byte per byte a quella del test 10**. È stata calibrata sulle due piccole mappe speculari rocce/ghiaia, non sulla mappa grande attuale. Il dataset di fit include quattro dei sette tipi di terreno.

SHA256: `bd2a8622ea2266604f1b9088e9b939f065b42b9204b0f6f252db0a107aadcf5e`.

## Raccolta originale verificata

Fonti: `multi_terrain_test10/prepare_pca.py`, metadata interni a `multi_terrain_test10/artifacts/pca32.npz`, manifest degli esperimenti e confronti degli hash.

- Raccolta scriptata prima dell'allenamento RL, sulle mappe `rocks_gravel__right_rocks` e `rocks_gravel__right_gravel`.
- Seed 123, 1500 passi di raccolta; campionamento bilanciato mediante reservoir.
- 40.000 vettori patch DINO da 384 elementi: 10.000 per materiale. Non sono 40.000 immagini indipendenti.
- Codici osservati nel fit: 0 asfalto, 1 ghiaccio, 3 ghiaia, 4 rocce. Assenti: 2 rampe, 5 ostacoli, 6 scale.
- DINOv2-small congelato, RGB ridimensionata a 518×518 con interpolazione lineare, nessun crop.
- PCA centrata 384→32, senza whitening. Varianza spiegata sul dataset di calibrazione: **81,93%**.
- I metadata identificano esperimento, raccolta e seed, ma non una data certa della raccolta: non ricaviamo il giorno di acquisizione dalla sola data di modifica del file.

Il manifest della mappa corrente conserva correttamente questa provenienza separata. Il controllo di compatibilità consente il trasferimento della PCA a un'altra mappa se DINO e preprocessing coincidono; questo verifica compatibilità tecnica, non rappresentatività dei dati.

## Controllo offline aggiuntivo

Senza avviare Isaac Sim o rifare il fit, abbiamo applicato la PCA esistente alle feature già salvate in `extracted_features_comparison_v2/data/features/pilot_01`: 70 immagini, 149 record per terreno/istanza, **20.550 patch**. I tre hash DINO nei metadata coincidono; la dimensione di input e il preprocessing nel codice di estrazione coincidono con quelli usati dalla pipeline RL.

Questo pilot usa celle donatrici della vecchia `real_terrains.usd`, non una nuova acquisizione sulla `real_terrains_v2.usd` del training. Sono riuniti gli split esistenti per una misura descrittiva, senza fitting, selezione di iperparametri o valutazione di classificatori. Le patch di una stessa immagine sono correlate.

Misura: `sum(||(X - media(X)) @ componenti.T||²) / sum(||X - media(X)||²)`. La media è calcolata sui dati della riga: all'interno del singolo materiale per le prime sette righe, sull'intero insieme per l'ultima. Quindi le righe per materiale non sono percentuali di riconoscimento né direttamente confrontabili con l'81,93% del fit originale misto.

| Terreno | Patch | Varianza conservata dalla PCA32 |
|---|---:|---:|
| Asfalto | 2.731 | 63,0% |
| Ghiaccio | 2.952 | 72,5% |
| Rampe | 2.847 | 57,0% |
| Ghiaia | 2.826 | 68,4% |
| Rocce | 2.980 | 66,5% |
| Ostacoli | 2.807 | 52,9% |
| Scale | 3.407 | 42,2% |
| Insieme dei sette terreni | 20.550 | 58,9% |

Il file `metrics.json` riporta anche l'energia dei vettori conservata rispetto al centro della calibrazione originale (57,8% aggregata): è una misura diversa dalla varianza centrata sul nuovo dataset.

Riproduzione dalla radice del progetto, senza simulazione:

```bash
OPENBLAS_NUM_THREADS=2 python3 spot_rl_energy_navigation/train_general_big_map_test4/analysis/pca_audit_20261027/audit.py
```

## Interpretazione e azione proposta

Il dubbio sulla rappresentatività è fondato. La PCA privilegia le direzioni di variabilità presenti nei dati su cui viene calibrata, come descritto nella [documentazione PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca). Materiali e geometrie nuovi possono perdere informazioni utili nella compressione.

La PCA non è però un classificatore: un terreno assente dal fit non diventa automaticamente invisibile o non riconoscibile. DINO rimane un modello preaddestrato; le misure geometriche da depth non passano attraverso la PCA. Inoltre alta varianza non equivale necessariamente a informazione utile al costo energetico. Questi numeri non dimostrano che la PCA causi giri a vuoto o mancate convergenze verso il goal.

Per il test 4 abbiamo mantenuto **esattamente la PCA del test 3**, rispettando il confronto richiesto sulla sola reward. Prima di un esperimento generalistico più ampio, è consigliabile raccogliere un dataset rappresentativo dei sette terreni e delle geometrie realmente usate, con diverse distanze, direzioni e pose; includere ghiaccio e scale anche se esclusi da alcuni endpoint, perché restano osservabili lungo i percorsi. Fit bilanciato sui soli dati di calibrazione/training, controllo su acquisizioni separate, poi congelamento definitivo prima dell'RL. La dimensione 32 va valutata su questi dati senza scegliere solo in base alla varianza spiegata.

Una PCA nuova cambia il significato delle coordinate ricevute dalla rete: non va sostituita a quella di una policy già allenata. Per questo confronto si allena una nuova policy e si mantiene sempre l'abbinamento policy/PCA, come già avviene nei checkpoint.

## Verifiche del test 4

31 test CPU superati, compresi soglia e simmetria della penalità di orientamento, scala temporale, esclusione dell'allineamento e integrazione nella testa navigazione con costo energetico invariato. Verificati launcher Bash, riferimenti agli asset e manifest. Mappa, banca spawn, metadata e PCA sono identici al test 3. L'unica differenza nella configurazione, oltre al nome esperimento, sono i quattro termini reward richiesti. Nessun training o simulazione avviato.
