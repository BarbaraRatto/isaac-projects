# Confronto delle feature DINOv2: ghiaccio, ghiaia, rocce

Cartella indipendente dagli esperimenti RL. Contiene codice, terreni, texture,
configurazione della camera e pesi DINO locali. Nessun import da test6/7/8,
nessun robot o checkpoint RL richiesto, nessun download durante l'esecuzione.
Isaac Sim/Isaac Lab e le librerie Python restano dipendenze di sistema.
Il materiale standard `OmniPBR.mdl` è fornito dal renderer Isaac Sim installato.

## Stato iniziale

Sono stati preparati soltanto file e copie degli asset. Non sono stati avviati
il simulatore, l'acquisizione, DINO o il classificatore sui dati reali.
Non ci sono risultati di separabilità già misurati.

## Cosa viene confrontato

- **Ghiaccio, ghiaia e rocce**, tutte e tre le coppie, con matrici 3×3.
- Terreni originali del board usato in test6, duplicati localmente:
  un tassello di ghiaccio, tre di ghiaia e tre di rocce. Ogni tassello misura
  2×2 m. Mesh e materiali sono conservati; si applica soltanto una traslazione
  per portare il centro nell'origine. Texture identiche agli originali.
- Un solo tassello visibile per immagine; gli altri sono nascosti. Le etichette
  derivano dal tassello selezionato, mai da predizioni DINO.
- Camera RGB-D 640×360 con ottiche ZED X copiate da quelle di test6.
  Altezza configurabile 0,70 m, inclinazione in basso 15°. L'altezza è una
  scelta per questa acquisizione, non una misura dell'assetto dinamico di Spot.
- `facebook/dinov2-small`, input 518×518, patch 14×14 pixel, 37×37 token
  da 384 componenti. Stesso preprocessing di test6, senza PCA, pesi congelati.

Tre rappresentazioni sono analizzate separatamente:

| Nome | Dati |
| --- | --- |
| `patches` | Feature dei token con centro e nove punti interni proiettati sul tassello, lontani dai bordi. |
| `cells` | Medie raw delle feature nelle celle da 0,5 m, tramite proiezione dei centri compatibile con il training. |
| `front_cells` | Celle valide intercettate dai 20 campioni frontali del training, con maschera angolare di 50°. |

Per `front_cells` una posa virtuale del robot si ricava da camera, yaw e
offset anteriore di 0,42 m. Non viene simulato Spot. Celle intercettate più
volte contano una volta sola nel confronto. Non sono incluse celle vuote.
Il vettore completo RL include anche stato e validità: qui si analizzano
soltanto i vettori visivi delle celle valide.

## Dimensioni della raccolta

`config.json` definisce tre distanze orizzontali dal centro del tassello:
0,8 / 1,2 / 1,8 m. Non sono distanze da ogni singola patch.

| Modalità | Tasselli | Angoli | Luci | Immagini |
| --- | ---: | ---: | ---: | ---: |
| `pilot` predefinita | Uno per terreno | 4 | Solo luce del training | **36** |
| `full` | Tutti e 7 | 8 | Tre condizioni | **504** |

La modalità pilota serve a verificare visibilità, proiezione ed etichette e
a ottenere un primo confronto. Prima di ampliare la raccolta occorre guardare
alcune immagini: una vista con pochi campioni validi non è utile.
Luci complete: intensità 3000 identica al training, 1500, e luce obliqua.
Le geometrie sono statiche: si aggiornano rendering e camere senza far
avanzare la fisica. Non c'è apprendimento RL.

## File

| File | Funzione |
| --- | --- |
| `config.json`, `common.py` | Configurazione, percorsi locali, integrità degli asset. |
| `assets/manifest.json` | Origine e hash dei terreni, texture e pesi; elenco dei tasselli. |
| `assets/terrains/`, `assets/textures/` | USD e texture locali, con riferimenti relativi. |
| `models/dinov2/` | Configurazione, preprocessore e pesi locali, copie reali senza symlink. |
| `scene.py`, `acquire.py` | Scena e acquisizione Isaac RGB-D. |
| `dino_features.py`, `projection.py`, `extract.py` | Estrazione offline, proiezione e aggregazione. |
| `check_dataset.py` | Verifica degli asset oppure di una raccolta registrata. |
| `compare.py` | Similarità, distribuzioni e classificatore lineare. |
| `report.py` | Figure PNG e report Markdown. |
| `requirements.txt` | Dipendenze offline; non installa Isaac Sim. |
| `requirements-analysis.txt`, `setup_analysis.sh`, `analysis_python.sh` | Ambiente locale isolato per confronto e report, senza modificare Python di sistema o Isaac. |
| `test_offline.py` | Controlli numerici con fixture sintetiche, senza simulatore o DINO. |

## Comandi per quando la macchina sarà libera

Eseguire da `my-projects`. I comandi qui sotto sono istruzioni: la creazione
della cartella non li esegue automaticamente. Soltanto `acquire.py` avvia
Isaac Sim; non lanciarlo mentre il training occupa la GPU.

Acquisizione pilota con finestra per ispezionare le inquadrature:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/extracted_features_comparison/acquire.py --mode pilot --output data/raw/pilot_01
```

Lo stesso comando con `--headless` non apre la finestra e mantiene il rendering.
Per la raccolta completa usare `--mode full` e un nuovo nome di output.
Nessuno script sovrascrive raccolte, feature o cartelle dei risultati esistenti.
Gli output devono restare dentro questa cartella.

Verifica della raccolta:

```bash
python3 spot_rl_energy_navigation/extracted_features_comparison/check_dataset.py --dataset data/raw/pilot_01
```

Estrazione DINO offline. Il Python fornito con Isaac Sim include Transformers;
questo comando avvia soltanto Python, senza `SimulationApp` né renderer.
`--device cpu` evita l'uso della GPU; per elaborare su GPU in seguito usare
`--device cuda:0` quando disponibile:

```bash
/home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh spot_rl_energy_navigation/extracted_features_comparison/extract.py --dataset data/raw/pilot_01 --device cpu
```

Per analisi e report usare l'ambiente locale isolato: evita di mescolare
NumPy del profilo utente e vecchie librerie SciPy/Matplotlib di sistema.
Preparazione una sola volta (non avvia Isaac né installa PyTorch):

```bash
bash spot_rl_energy_navigation/extracted_features_comparison/setup_analysis.sh
```

Poi usare sempre il launcher locale per confronto e report:

```bash
bash spot_rl_energy_navigation/extracted_features_comparison/analysis_python.sh spot_rl_energy_navigation/extracted_features_comparison/compare.py --features data/features/pilot_01
bash spot_rl_energy_navigation/extracted_features_comparison/analysis_python.sh spot_rl_energy_navigation/extracted_features_comparison/report.py --results results/pilot_01
```

La cartella `results/pilot_01/` conterrà `report.md`, matrici CSV, metriche JSON,
distribuzioni campionate e figure per tutte e tre le rappresentazioni.
Le immagini originali, profondità, intrinseci e pose restano in `data/raw/`.
I vettori e i riferimenti ai frame restano in `data/features/`.
`complete.json` viene scritto soltanto al termine: i passaggi successivi
rifiutano raccolte o estrazioni interrotte.

Se l'acquisizione fallisce, il traceback Python viene stampato con prefisso
`[CAPTURE-ERROR]` prima della chiusura di Isaac e salvato in `logs/` locale.
L'assenza di `complete.json` indica che la raccolta non è terminata.

## Come leggere i risultati

La matrice delle similarità ha una diagonale calcolata fra **viste diverse
dello stesso terreno**, non impostata a 1. Valori interni superiori a quelli
tra terreni sono un segnale favorevole; non c'è una soglia assoluta universale.
Vengono salvati media, deviazione standard e quantili, oltre alla differenza
tra similarità interna e tra terreni e alla ROC AUC per coppia
(0,5 = nessuna separazione mediante questa misura).

La regressione logistica è un probe lineare, con DINO congelato, L2 dei vettori,
nessuna PCA e nessuna ottimizzazione sul test set. Tutte le immagini della
stessa combinazione distanza/angolo, incluse luci e tasselli diversi, rimangono
nello stesso gruppo. I gruppi sono separati fra training e test (70/30).
Si misura balanced accuracy: riferimento casuale **33,3%** per tre classi.
Gli intervalli indicativi sono ottenuti ricampionando intere pose di test.
Il numero di patch non è il numero di osservazioni indipendenti.

Il confronto campiona al massimo 32 vettori per frame; le similarità campionano
uniformemente coppie di frame di pose diverse. La matrice di confusione mostra
quali terreni il probe confonde. Le norme originali dei vettori sono riportate
separatamente: la similarità coseno ne trascura l'ampiezza. La PCA a due dimensioni
è soltanto una figura illustrativa.

## Limiti

- È un controllo sui materiali e sulle geometrie di questi asset simulati,
  non una prova di generalizzazione ad altri ghiacci, ghiaie o rocce reali.
- I token DINO hanno contesto globale: nove punti proiettati non garantiscono
  che tutta l'informazione di un token provenga solo dal suo quadratino.
- Si usa un tassello isolato, senza robot né contesto misto della mappa RL.
  Questo controlla la distinguibilità di base; scene miste possono essere
  una raccolta successiva. La fedeltà delle feature aggregate è numerica,
  non una riproduzione completa di una navigazione.
- Poche pose danno stime incerte. Separare visivamente i terreni non dimostra
  che la policy usi l'informazione né che abbia imparato il costo energetico.
- La verifica del rendering/inquadrature resta da fare con la raccolta pilota:
  il simulatore non è stato avviato durante questa preparazione.

## Controlli senza GPU

```bash
python3 spot_rl_energy_navigation/extracted_features_comparison/check_dataset.py
PYTHONDONTWRITEBYTECODE=1 python3 spot_rl_energy_navigation/extracted_features_comparison/test_offline.py
```

Il primo verifica solo hash e presenza dei file; il secondo usa numeri
sintetici. Nessuno dei due avvia Isaac o carica i pesi DINO.
