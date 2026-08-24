# spot_terrain_gridmap

Pipeline immagine → gridmap del terreno (Sez. 4.2 del documento di contesto
del progetto di tesi). Estrae feature visive continue tramite **DINOv2
frozen** dall'immagine camera e le proietta in una **gridmap BEV locale**,
pubblicata su `/terrain_gridmap` come input per la policy di Reinforcement
Learning.

---

## Struttura del package

```
spot_terrain_gridmap/
├── package.xml                  # dipendenze ROS (ament_python)
├── setup.py / setup.cfg          # build standard ament_python
├── resource/spot_terrain_gridmap # marker file richiesto da ament
├── config/
│   └── terrain_gridmap_params.yaml   # TUTTI i parametri, inclusi i placeholder
├── launch/
│   └── terrain_gridmap.launch.py
├── spot_terrain_gridmap/
│   ├── __init__.py
│   ├── terrain_feature_node.py   # nodo ROS: orchestratore (I/O, callback)
│   ├── bev_projection.py         # logica PURA (no rclpy): geometria omografia
│   └── gridmap_builder.py        # costruzione messaggio grid_map_msgs/GridMap
└── test/
    └── test_bev_projection.py    # test standalone, NO ROS richiesto
```

**Perché la logica è divisa in 3 file invece di un unico nodo monolitico?**
`bev_projection.py` non importa `rclpy`: è pura matematica (numpy). Questo
significa che puoi testarla, debuggarla e tararla con uno script Python
qualsiasi, **senza dover avere ROS 2, Isaac Sim o la camera reale
accesi**. Utile ora, dato che la camera non è ancora integrata in
Isaac Sim.

---

## 1. Setup dell'ambiente Python (venv)

Il nodo ha bisogno di **PyTorch** e **transformers** (per DINOv2), librerie
pesanti che è meglio NON installare nel Python di sistema usato da ROS 2 —
per non rischiare di rompere versioni di `numpy`/altre librerie da cui
dipendono anche gli altri nodi del progetto (es. `spot_energy_estimation`).

Si usa quindi un **virtual environment (venv)** dedicato, con accesso anche
ai pacchetti di sistema (per vedere `rclpy`):

```bash
# Crea il venv (una tantum)
python3 -m venv ~/venvs/dino_env --system-site-packages

# Attiva il venv (da fare in OGNI terminale in cui vuoi lanciare il nodo)
source ~/venvs/dino_env/bin/activate

# Installa le dipendenze Python del nodo (una tantum, dentro il venv attivo)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow
```

**Nota sulla versione CUDA**: `cu121` nell'URL sopra assume CUDA 12.1.
Verifica la tua versione con `nvidia-smi` (in alto a destra nell'output) e
sostituisci se necessario (es. `cu118` per CUDA 11.8). Se non sei sicuro,
dimmi l'output di `nvidia-smi` e ti dico quale usare.

**Perché `--system-site-packages`?** Permette al venv di vedere comunque i
pacchetti installati dal sistema (incluso `rclpy`, `sensor_msgs`, ecc.,
installati con ROS 2), oltre ai propri pacchetti isolati (`torch`,
`transformers`). Senza questa flag dovresti reinstallare anche `rclpy` nel
venv, cosa sconsigliata (rclpy è legato alla build di ROS 2 del sistema).

**Verifica che tutto sia a posto:**

```bash
source ~/venvs/dino_env/bin/activate
python3 -c "import rclpy; import torch; print('rclpy OK, torch OK, CUDA disponibile:', torch.cuda.is_available())"
```

Se `CUDA disponibile: False` ma hai una GPU NVIDIA, quasi certamente hai
installato la versione sbagliata di PyTorch per la tua versione CUDA — fammi
sapere e ti aiuto a correggere.

---

## 2. Testare la proiezione BEV SENZA ROS (subito, ora)

Prima ancora di costruire il workspace ROS, puoi verificare che la
geometria di proiezione funzioni, con dati sintetici:

```bash
cd spot_terrain_gridmap
python3 test/test_bev_projection.py
```

Questo NON richiede il venv con PyTorch (usa solo numpy), né ROS, né Isaac
Sim: è pensato per validare/tarare `camera_pose` (altezza, tilt) e i
parametri della gridmap, sperimentando rapidamente. Se hai `matplotlib`
installato (`pip install matplotlib`), genera anche un'immagine di anteprima
della copertura in `test/bev_coverage_preview.png`.

**Come interpretare l'output:**
- `Celle valide: X/Y (Z%)` — quante celle della gridmap 6×6m vengono
  effettivamente "viste" dalla camera con i parametri correnti. Se è molto
  basso (<5%) o molto alto (>80%), probabilmente `camera_pose` o
  `gridmap.length_x/y` vanno rivisti.
- L'immagine di anteprima dovrebbe mostrare una forma "a ventaglio" che si
  allarga allontanandosi dal robot — è la forma tipica del campo visivo di
  una camera proiettato a terra. Se vedi rumore sparso senza struttura,
  c'è probabilmente un bug nei parametri o nella geometria: fermati e
  chiedi aiuto prima di procedere.

---

## 3. Integrare il package nel workspace ROS 2

```bash
# Copia (o crea link simbolico) il package nel tuo workspace esistente
cp -r spot_terrain_gridmap ~/ros2_ws/src/

cd ~/ros2_ws
colcon build --packages-select spot_terrain_gridmap
source install/setup.bash
```

**Dipendenze ROS di sistema** (da installare UNA TANTUM col Python di
sistema, non nel venv — sono binding C++ legati alla build ROS):

```bash
sudo apt install ros-humble-grid-map-msgs ros-humble-cv-bridge
```

---

## 4. Lanciare il nodo

Il nodo richiede il venv attivo (per PyTorch/transformers) MA anche
l'ambiente ROS 2 sorgente (per `ros2 launch`). Vanno combinati:

```bash
source /opt/ros/humble/setup.bash        # ambiente ROS 2
source ~/ros2_ws/install/setup.bash       # workspace del progetto
source ~/venvs/dino_env/bin/activate      # venv con PyTorch (DOPO i source ROS)

ros2 launch spot_terrain_gridmap terrain_gridmap.launch.py
```

**Nota sull'ordine**: attiva il venv PyTorch per ultimo. Se lo attivi prima
dei `source` ROS, alcuni tool ROS potrebbero usare l'interprete Python del
venv invece di quello di sistema per operazioni di build/indicizzazione,
causando errori poco chiari.

Per lanciare su CPU (debug, molto più lento, utile se non hai ancora
accesso alla GPU o vuoi solo verificare che non ci siano errori):

```bash
ros2 launch spot_terrain_gridmap terrain_gridmap.launch.py device:=cpu
```

---

## 5. Cosa aggiornare quando la camera sarà integrata in Isaac Sim

Quando aggiungerai la camera al robot in Isaac Sim (vedi la nota nel
documento di contesto, Sez. 1.2: "la telecamera non è ancora presente nel
simulatore"), dovrai aggiornare in `config/terrain_gridmap_params.yaml`:

1. **`camera_pose`** (x, y, z, tilt_deg): con i valori REALI di dove hai
   posizionato il prim Camera nel simulatore, rispetto a `base_link`.
2. Verifica che `/camera/color/camera_info` sia effettivamente pubblicato
   da Isaac Sim con parametri intrinseci sensati: il nodo lo userà
   automaticamente al posto del placeholder (vedi
   `TerrainFeatureNode._get_intrinsics()` in `terrain_feature_node.py`).
3. Ri-esegui `test/test_bev_projection.py` con i nuovi valori di
   `camera_pose` per verificare che la copertura/geometria abbia ancora
   senso, PRIMA di lanciare il nodo completo su Isaac Sim.

---

## 6. Punti aperti / limiti noti (da tenere a mente)

- **Ipotesi di terreno piano**: la proiezione BEV assume che il terreno
  davanti al robot sia localmente piano. Su terreni molto irregolari
  (sassi, buche pronunciate) introduce un errore geometrico crescente con
  la distanza. Per ora coerente con l'assenza di un layer elevation
  obbligatorio (vedi Sez. 4.2.5 del documento).
- **Dimensione/risoluzione gridmap** (TODO 1 del documento): i valori in
  `config/terrain_gridmap_params.yaml` (6×6m, celle 10cm) sono un punto di
  partenza, non tarati su dati reali. Da rivedere una volta nota la
  risoluzione reale della camera e il comportamento della policy RL.
- **Layer `terrain_features_XXX`**: con ViT-S (384-dim) il messaggio
  GridMap avrà 384 layer scalari — è il modo idiomatico di rappresentare un
  embedding vettoriale in `grid_map_msgs`, ma è verboso e comporta un
  overhead di serializzazione. Se in futuro risultasse un collo di
  bottiglia, si può valutare di ridurre la dimensionalità con un linear
  probe (già previsto come opzione nel documento, Sez. 4.2.3) prima della
  pubblicazione.
- **Layer elevation**: non implementato (placeholder in
  `gridmap_builder.build_elevation_layer_placeholder`), richiede la depth
  camera non ancora disponibile.
