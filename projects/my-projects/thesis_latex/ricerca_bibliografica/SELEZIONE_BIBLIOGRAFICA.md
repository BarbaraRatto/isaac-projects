# Selezione bibliografica per la tesi

File collegati: [valutazione di tutti i PDF locali](VALUTAZIONE_PDF_LOCALI.md), [appunti sul progetto e piano concordato](NOTE_PROGETTO.md), [matrice in CSV](matrice_riferimenti.csv), [metadati e provenienza delle verifiche](metadati_verificati.json), [bibliografia LaTeX](../template_tesi_RoboticsLaTeX2/RoboticsLaTeX2/references.bib).


Prima selezione ragionata, verificata il **6 novembre 2026**. **40 lavori** proposti; la voce PID del template non fa parte della selezione. La presenza nel `.bib` non obbliga a citare il lavoro: ogni citazione dovrà sostenere un passaggio preciso.

## Perimetro e metodo


Il tema guida è la **navigazione verso un obiettivo con costo energetico, a locomozione fissata**, usando rappresentazioni visive e feedback propriocettivo. La traversabilità è pertinente perché collega percezione, capacità del robot e costo di attraversamento, ma percorribilità, stabilità e consumo sono grandezze distinte.

Sono stati esaminati titolo, abstract e parti pertinenti dei 29 PDF locali (28 lavori distinti, con un duplicato), inclusi i riferimenti del survey non pubblicato, usato solo per trovare fonti primarie. La ricerca web ha aggiunto lavori mirati; non è una systematic review e non dimostra esaustività del campo.

Ricerche principali: `quadruped energy aware navigation visual proprioceptive learning`; `Energy-Based Legged Robots Terrain Traversability Modeling`; `Where Should I Walk self supervised terrain`; `TOP-Nav`; `AMCO`; `ProNav`; `STERLING`; `Identifying Terrain Physical Parameters from Vision`; `online incremental terrain traversal cost`. Sono stati poi cercati i lavori originali di PPO, SAC, GAE, actor-critic asimmetrico, DINOv2, mapping e simulatori.

Criteri di inclusione: vicinanza a scelta del tragitto/costo, uso dell’interazione robot-terreno, fusione sensoriale, assenza o ruolo delle classi semantiche, oppure necessità metodologica per spiegare il codice. Priorità ridotta per locomozione pura, ostacoli dinamici, survey generici e varianti di planner senza un nesso energetico specifico. Nessun PDF esistente è stato eliminato.

**Verifica e limiti della lettura.** Per i PDF locali e per Gan, AMCO, ProNav, STERLING, Chen, Roth, Prágr e Faigl sono stati consultati anche passaggi del testo integrale; non si tratta di una lettura critica integrale di tutti i 40 articoli. Per gli altri, lo screening si fonda su abstract/pagina ufficiale e metadati. Autori, titolo, sede e anno sono stati verificati tramite DOI/Crossref, proceedings PMLR/RSS e arXiv. Le risposte automatiche per titolo sono state controllate: corrispondenze errate non sono entrate nella bibliografia.

## Ordine consigliato per la lettura approfondita


1. Gan: precedente diretto sulla traversabilità energetica.
2. Zhang: navigazione RL RGB-D appresa sul robot reale.
3. ProNav: Spot, esperienza propriocettiva ed energia.
4. WVN: feature preaddestrate e apprendimento online.
5. STEPP: DINOv2 e MLP, senza costo energetico diretto.
6. TOP-Nav e AMCO: feedback, storia e architetture di navigazione.
7. Prágr e Faigl: costi di attraversamento appresi dall’esperienza.
8. STERLING e Chen: rappresentazioni multimodali e proprietà fisiche.

## Catalogo annotato


Le indicazioni di capitolo sono orientative: 1 motivazione; 2 stato dell’arte/fondamenti; 3 metodo; 4 valutazione. I riferimenti metodologici non vanno forzati nell’introduzione per raggiungere un numero prestabilito.


### A — Navigazione, traversabilità e apprendimento dall’esperienza


**`Gan2022Energy` — [Energy-Based Legged Robots Terrain Traversability Modeling via Deep Inverse Reinforcement Learning](https://doi.org/10.1109/lra.2022.3188100) (2022).** IEEE Robotics and Automation Letters. Priorità **alta**; capitoli 1,2.

Traversabilità energetica da esterocettività e propriocezione; precedente diretto del problema. **Distinzione:** Apprende una ricompensa mediante inverse RL e ranking di dimostrazioni; non coincide con la policy di navigazione proposta.


**`Zhang2026Traversability` — [Traversability-Aware Legged Navigation by Learning From Real-World Visual Data](https://doi.org/10.1109/tro.2025.3645885) (2026).** IEEE Transactions on Robotics. Priorità **alta**; capitoli 1,2.

Navigazione RL da RGB-D reale e costo di traversabilità legato al controllore. **Distinzione:** Non equiparare il costo derivato dal controllore a una misura elettrica o alla stessa funzione obiettivo della tesi.


**`Mattamala2025Wild` — [Wild visual navigation: fast traversability learning via pre-trained models and online self-supervision](https://doi.org/10.1007/s10514-025-10202-x) (2025).** Autonomous Robots. Priorità **alta**; capitoli 1,2.

Feature visive preaddestrate e supervisione dall'esperienza per apprendere traversabilità online. **Distinzione:** Predire traversabilità e ottimizzare energia con RL sono obiettivi diversi; qui l'adattamento online è esplicito.


**`Aegidius2025STEPP` — [Watch Your STEPP: Semantic Traversability Estimation Using Pose Projected Features](https://doi.org/10.1109/icra55743.2025.11127781) (2025).** 2025 IEEE International Conference on Robotics and Automation (ICRA). Priorità **alta**; capitoli 2,3.

DINOv2, proiezione delle osservazioni e MLP senza tassonomia manuale dei terreni. **Distinzione:** L'errore di ricostruzione rileva familiarità/anomalie; non stima direttamente Joule o percorso energetico ottimo.


**`Wellhausen2019Where` — [Where Should I Walk? Predicting Terrain Properties From Images Via Self-Supervised Learning](https://doi.org/10.1109/lra.2019.2895390) (2019).** IEEE Robotics and Automation Letters. Priorità **alta**; capitoli 1,2.

Collega immagini e interazioni piede-terreno tramite proiezione dei contatti. **Distinzione:** Il ground reaction score non coincide con l'energia ai giunti.


**`Fu2022Coupling` — [Coupling Vision and Proprioception for Navigation of Legged Robots](https://doi.org/10.1109/cvpr52688.2022.01676) (2022).** 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Priorità **alta**; capitoli 1,2.

VP-Nav: integrazione tra navigazione visiva e feedback propriocettivo. **Distinzione:** Il feedback di sicurezza non è una memoria esplicita di consumi; non trattarlo come ottimizzatore energetico.


**`Ren2025TOPNav` — [TOP-Nav: Legged Navigation Integrating Terrain, Obstacle and Proprioception Estimation](https://proceedings.mlr.press/v270/ren25a.html) (2025).** Conference on Robot Learning. Priorità **alta**; capitoli 2.

Chiude il ciclo tra pianificatore, stima del terreno e valutazione propriocettiva. **Distinzione:** L'IMU energy density è un indicatore di oscillazione e non un consumo meccanico/elettrico in Joule.


**`Elnoor2024AMCO` — [AMCO: Adaptive Multimodal Coupling of Vision and Proprioception for Quadruped Robot Navigation in Outdoor Environments](https://doi.org/10.1109/iros58592.2024.10801962) (2024).** 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Priorità **alta**; capitoli 2.

Fonde mappa semantica, storia di attraversamento e propriocezione corrente. **Distinzione:** Usa segmentazione semantica e selezione delle andature; differisce dalle feature senza classi e dal controllore congelato.


**`Elnoor2024ProNav` — [ProNav: Proprioceptive Traversability Estimation for Legged Robot Navigation in Outdoor Environments](https://doi.org/10.1109/lra.2024.3418270) (2024).** IEEE Robotics and Automation Letters. Priorità **alta**; capitoli 1,2.

Navigazione su Spot, stima propriocettiva di traversabilità e scelta dell'andatura. **Distinzione:** Il risparmio include cambi di andatura; la tesi interviene invece sui comandi di navigazione.


**`Karnan2023STERLING` — [STERLING: Self-Supervised Terrain Representation Learning from Unconstrained Robot Experience](https://proceedings.mlr.press/v229/karnan23a.html) (2023).** Conference on Robot Learning. Priorità **alta**; capitoli 2.

Apprende rappresentazioni multimodali del terreno da esperienza non annotata. **Distinzione:** Piattaforma e navigazione per preferenze differiscono; non dimostra l'ottimalità energetica della policy della tesi.


**`Chen2024Physical` — [Identifying Terrain Physical Parameters From Vision - Towards Physical-Parameter-Aware Locomotion and Navigation](https://doi.org/10.1109/lra.2024.3455788) (2024).** IEEE Robotics and Automation Letters. Priorità **alta**; capitoli 2.

Associa interazione fisica e feature visive per stimare attrito e rigidezza. **Distinzione:** Stima parametri fisici espliciti; non apprende direttamente il percorso di minimo consumo.


**`Frey2022PolicyGuided` — [Locomotion Policy Guided Traversability Learning using Volumetric Representations of Complex Environments](https://doi.org/10.1109/iros47612.2022.9982190) (2022).** 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Priorità **alta**; capitoli 2.

Traversabilità dipendente dalle capacità di una policy di locomozione. **Distinzione:** Rappresentazione volumetrica e stima di attraversabilità, non identiche alla gridmap RGB-D della tesi.


**`Muhamad2024MultipleCosts` — [Robust Traversability Prediction Using Multiple Costs for Quadruped Robot in Random Terrains](https://doi.org/10.1109/access.2024.3371579) (2024).** IEEE Access. Priorità **media**; capitoli 2.

Predizione di molteplici costi di traversabilità con dati da simulatore. **Distinzione:** Verificare i singoli costi prima di chiamarli energetici; non tutti rappresentano lavoro ai giunti.


**`Belter2016Adaptive` — [Adaptive Motion Planning for Autonomous Rough Terrain Traversal with a Walking Robot](https://doi.org/10.1002/rob.21610) (2016).** Journal of Field Robotics. Priorità **media**; capitoli 2.

Precedente di pianificazione gerarchica e regressione della traversabilità su terreno accidentato. **Distinzione:** Piattaforma camminante e pianificatori espliciti; il percorso non è deciso dalla stessa policy RL.


**`SanchezDelgado2025Mapping` — [Towards Proprioceptive Terrain Mapping with Quadruped Robots for Exploration in Planetary Permanently Shadowed Regions](https://doi.org/10.1109/isparo66239.2025.11436327) (2025).** 2025 International Conference on Space Robotics (iSpaRo). Priorità **media**; capitoli 2.

Mappatura propriocettiva delle proprietà di attraversamento su quadrupede. **Distinzione:** Scenario planetario e mappa fisica; il PDF locale è il preprint di un lavoro poi pubblicato a iSpaRo.


**`Roth2025Forward` — [Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation](https://doi.org/10.15607/rss.2025.xxi.001) (2025).** Robotics: Science and Systems XXI. Priorità **alta**; capitoli 2.

Predice dinamica e rischio in funzione di geometria e storia propriocettiva per pianificazione MPPI. **Distinzione:** Apprende un modello di dinamica usato da un planner; non una policy con costo energetico diretto.


**`Sevastopoulos2022Survey` — [A Survey of Traversability Estimation for Mobile Robots](https://doi.org/10.1109/access.2022.3202545) (2022).** IEEE Access. Priorità **media**; capitoli 1,2.

Survey pubblicato per definizioni di traversabilità e tassonomia dei metodi. **Distinzione:** Inquadra il campo ma non sostituisce i lavori primari né dimostra un gap specifico della tesi.


**`Pragr2019Online` — [Online Incremental Learning of the Terrain Traversal Cost in Autonomous Exploration](https://doi.org/10.15607/rss.2019.xv.040) (2019).** Robotics: Science and Systems XV. Priorità **alta**; capitoli 2.

Apprendimento incrementale online di costi di attraversamento e uso nella scelta delle mete esplorative. **Distinzione:** Esplorazione e regressione con Gaussian Processes su robot multigamba; differiscono da goal assegnato e policy RL.


**`Faigl2019Unsupervised` — [On Unsupervised Learning of Traversal Cost and Terrain Types Identification Using Self-organizing Maps](https://doi.org/10.1007/978-3-030-30487-4_50) (2019).** Artificial Neural Networks and Machine Learning – ICANN 2019: Theoretical Neural Computation. Priorità **alta**; capitoli 2.

Apprendimento non supervisionato del costo di attraversamento e identificazione del terreno. **Distinzione:** Usa self-organizing maps su esapode; distingue l'apprendimento del costo dalla sua ottimizzazione via RL.



### B — Efficienza energetica della locomozione


**`Liang2025AdaptiveEnergy` — [Adaptive Energy Regularization for Autonomous Gait Transition and Energy-Efficient Quadruped Locomotion](https://doi.org/10.1109/icra55743.2025.11128812) (2025).** 2025 IEEE International Conference on Robotics and Automation (ICRA). Priorità **media**; capitoli 2.

Mostra l'effetto di una regolarizzazione energetica sull'apprendimento della locomozione. **Distinzione:** Ottimizza andature e controllo motorio; usarlo per delimitare la scala del contributo.


**`Fu2022Gaits` — [Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged Robots](https://proceedings.mlr.press/v164/fu22a.html) (2022).** Conference on Robot Learning. Priorità **alta**; capitoli 1,2.

Mostra come la minimizzazione energetica possa far emergere andature. **Distinzione:** L'oggetto appreso è la locomozione; non è un confronto diretto sulla scelta del tragitto.


**`Yan2024Energy` — [Energy Consumption Minimization of Quadruped Robot Based on Reinforcement Learning of DDPG Algorithm](https://doi.org/10.3390/act13010018) (2024).** Actuators. Priorità **media**; capitoli 2.

Esempio di minimizzazione energetica con DDPG su quadrupede. **Distinzione:** Riguarda la locomozione; non giustifica l'uso di DDPG nella tesi e non prova selezione energetica del percorso.



### C — Locomozione, gerarchia e trasferimento


**`Rudin2022Advanced` — [Advanced Skills by Learning Locomotion and Local Navigation End-to-End](https://doi.org/10.1109/iros47612.2022.9981198) (2022).** 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Priorità **alta**; capitoli 2.

Confronto tra decomposizione navigazione/locomozione e controllo appreso end-to-end. **Distinzione:** Ottimizza congiuntamente comportamenti motori; la policy della tesi opera sopra un controllore congelato.


**`Miki2022Perceptive` — [Learning robust perceptive locomotion for quadrupedal robots in the wild](https://doi.org/10.1126/scirobotics.abk2822) (2022).** Science Robotics. Priorità **media**; capitoli 1,2.

Riferimento forte sull'unione di percezione e propriocezione nella locomozione robusta. **Distinzione:** Il successo su terreno complesso non equivale a minimizzare l'energia del percorso.


**`Kumar2021RMA` — [RMA: Rapid Motor Adaptation for Legged Robots](https://doi.org/10.15607/rss.2021.xvii.011) (2021).** Robotics: Science and Systems XVII. Priorità **media**; capitoli 2.

Distingue adattamento online da policy statica e uso di informazione privilegiata in training. **Distinzione:** Adattamento motorio, non navigazione energetica; non attribuire alla tesi un adaptation module non implementato.


**`Rudin2022Massively` — [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://proceedings.mlr.press/v164/rudin22a.html) (2022).** Conference on Robot Learning. Priorità **media**; capitoli 2,3.

Giustifica simulazione parallela e curriculum per RL su robot a zampe. **Distinzione:** Il risultato di training rapido sulla locomozione non si trasferisce automaticamente alla navigazione con RGB-D.



### D — Fondamenti degli algoritmi di apprendimento


**`Schulman2017PPO` — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017).** arXiv preprint. Priorità **alta**; capitoli 2,3.

Fonte primaria dell'algoritmo PPO usato nello scenario speculare. **Distinzione:** La separazione delle teste di valore nel codice è una scelta implementativa aggiuntiva da descrivere.


**`Haarnoja2018SAC` — [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://proceedings.mlr.press/v80/haarnoja18b.html) (2018).** International Conference on Machine Learning. Priorità **alta**; capitoli 2,3.

Fonte primaria di SAC per controllo continuo off-policy a massima entropia. **Distinzione:** Il paper ICML descrive la formulazione iniziale; la variante concreta del codice va verificata separatamente.


**`Schulman2016GAE` — [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) (2016).** International Conference on Learning Representations (ICLR). Priorità **alta**; capitoli 2,3.

Fonte della stima generalizzata del vantaggio utilizzata in PPO. **Distinzione:** La combinazione di vantaggi di navigazione ed energia non è automaticamente un nuovo algoritmo multiobiettivo.


**`Pinto2018Asymmetric` — [Asymmetric Actor Critic for Image-Based Robot Learning](https://doi.org/10.15607/rss.2018.xiv.008) (2018).** Robotics: Science and Systems XIV. Priorità **alta**; capitoli 2,3.

Fondamento dell'actor-critic asimmetrico con stato privilegiato durante il training. **Distinzione:** Documentare precisamente quali variabili sono solo nel critic e quali arrivano all'actor.


**`Ng1999Shaping` — [Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) (1999).** Proceedings of the Sixteenth International Conference on Machine Learning (ICML). Priorità **media**; capitoli 2,3.

Serve a discutere il termine di progresso verso il goal e il reward shaping. **Distinzione:** Una differenza di distanze non garantisce invarianza della policy per gamma diverso da 1 e terminazioni arbitrarie.



### E — Rappresentazioni visive e mappatura


**`Oquab2023DINOv2` — [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) (2023).** arXiv preprint. Priorità **alta**; capitoli 2,3.

Fonte del backbone congelato che produce le feature visive del progetto. **Distinzione:** Citata qui la versione arXiv verificata dal repository ufficiale; non è un classificatore dei sette materiali.


**`Caron2021DINO` — [Emerging Properties in Self-Supervised Vision Transformers](https://doi.org/10.1109/iccv48922.2021.00951) (2021).** 2021 IEEE/CVF International Conference on Computer Vision (ICCV). Priorità **media**; capitoli 2.

Antecedente del paradigma di apprendimento auto-supervisionato delle feature visive. **Distinzione:** Una breve contestualizzazione è sufficiente; evitare una rassegna di visione scollegata dal progetto.


**`Dosovitskiy2021ViT` — [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (2021).** International Conference on Learning Representations (ICLR). Priorità **media**; capitoli 2,3.

Definisce patch e Vision Transformer usati per spiegare la rappresentazione visiva. **Distinzione:** Architettura di base; non dimostra separabilità dei terreni o qualità delle decisioni energetiche.


**`Erni2023MEM` — [MEM: Multi-Modal Elevation Mapping for Robotics and Learning](https://doi.org/10.1109/iros55552.2023.10342108) (2023).** 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Priorità **alta**; capitoli 2,3.

Riferimento per mappe di elevazione con più modalità, feature semantiche e uso nell'apprendimento. **Distinzione:** La mappa personalizzata del progetto va descritta come implementata; non attribuirle automaticamente tutte le proprietà di MEM.


**`Fankhauser2018Mapping` — [Probabilistic Terrain Mapping for Mobile Robots With Uncertain Localization](https://doi.org/10.1109/lra.2018.2849506) (2018).** IEEE Robotics and Automation Letters. Priorità **media**; capitoli 2,3.

Mappatura del terreno con incertezza di localizzazione; utile per collocare la gridmap. **Distinzione:** La posa esatta del simulatore nel progetto non riproduce la deriva e l'incertezza reali.


**`Jolliffe2016PCA` — [Principal component analysis: a review and recent developments](https://doi.org/10.1098/rsta.2015.0202) (2016).** Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences. Priorità **media**; capitoli 2,3.

Fonte metodologica per la riduzione dimensionale PCA delle feature. **Distinzione:** La varianza preservata non prova di conservare l'informazione più utile al costo energetico.



### F — Simulazione e valutazione


**`Makoviychuk2021IsaacGym` — [Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning](https://arxiv.org/abs/2108.10470) (2021).** arXiv preprint. Priorità **media**; capitoli 2,3.

Antecedente tecnico della simulazione fisica GPU per RL parallelo. **Distinzione:** Isaac Gym e Isaac Lab sono sistemi distinti: citare il secondo per il framework effettivamente impiegato.


**`Mittal2025IsaacLab` — [Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning](https://arxiv.org/abs/2511.04831) (2025).** arXiv preprint. Priorità **alta**; capitoli 3.

Fonte scientifica per Isaac Lab e simulazione multimodale accelerata su GPU. **Distinzione:** È un preprint; la versione software effettiva va indicata dai file di configurazione finali.


**`Anderson2018Evaluation` — [On Evaluation of Embodied Navigation Agents](https://arxiv.org/abs/1807.06757) (2018).** arXiv preprint. Priorità **media**; capitoli 2,4.

Riferimento per successo e SPL nella valutazione della navigazione. **Distinzione:** SPL misura efficienza geometrica, non energetica: occorrono anche Joule e confronti a parità di successo.



## Confronti da sviluppare nel capitolo 2

Questa matrice è una guida di lettura; indica il contributo utile al confronto senza riportare classifiche numeriche fra esperimenti non omogenei.

| Lavoro | Collegamento fra esperienza e percezione | Cosa determina le decisioni | Differenza da verificare rispetto alla tesi |
|---|---|---|---|
| Gan 2022 | Input esterocettivi e propriocettivi; ranking energetico di traiettorie | Ricompensa appresa tramite inverse RL | Navigazione RL con penalità sul lavoro misurato, senza dimostrazioni da ordinare |
| Zhang 2026 | RGB-D e costo dipendente dal controllore | Policy di navigazione appresa nel mondo reale | Addestramento in simulazione e definizione operativa del costo energetico |
| WVN 2025 | Feature preaddestrate e supervisione generata dall’esperienza | Stima di traversabilità utilizzabile nella navigazione | Distinguere aggiornamento online del modello e policy congelata |
| STEPP 2025 | Feature DINOv2 e traiettorie dimostrate | Ricostruzione delle feature e rilevamento di anomalie | Familiarità del terreno diversa da costo energetico |
| VP-Nav 2022 | Visione e feedback propriocettivo di sicurezza | Navigazione accoppiata alla valutazione del moto | Feedback energetico e rappresentazione locale specifici del progetto |
| ProNav 2024 | Segnali articolari, forze e correnti su Spot | Traversabilità propriocettiva e selezione dell’andatura | Controllore locomotorio fissato e scelta di tragitto/velocità |
| AMCO 2024 | Semantica, storia di attraversamento e feedback corrente | Fusione adattiva di mappe e pianificazione | Rappresentazione visiva senza classi nell’actor |
| Prágr 2019 | Esperienza di attraversamento e modello del terreno | Regressione incrementale del costo ed esplorazione | Goal assegnato e decisioni apprese con RL |
| Faigl 2019 | Forma/aspetto e costi propriocettivi | Modello non supervisionato del costo, poi ricerca del percorso | Regressione esplicita del costo diversa da associazione appresa nella policy |
| Roth 2025 | Geometria e storia propriocettiva | Modello di dinamica e planner MPPI | Policy di navigazione senza lo stesso modello predittivo esplicito |

Nel testo finale le colonne dovranno usare definizioni uniformi: piattaforma, sensori, input della policy, uso di classi, apprendimento online/offline, livello di controllo, quantità energetica e tipo di validazione. Una cella non verificata va lasciata come “non verificato”, senza dedurla dalla sola assenza di informazioni nell’abstract.

## Scelte bibliografiche e controlli

- La distribuzione proposta è: **19** lavori su navigazione/traversabilità, **3** sull’efficienza della locomozione, **4** su locomozione/gerarchia/trasferimento, **5** sui fondamenti RL, **6** su feature e mappe, **3** su simulazione e valutazione.
- Per WVN è stata scelta l’estensione di rivista del 2025; il precedente RSS 2023 è consultabile attraverso il repository degli autori, senza contarlo come ulteriore lavoro della selezione.
- Per Zhang si cita T-RO 2026; per Belter il fascicolo 2016; per TOP-Nav i proceedings PMLR 2025 della conferenza CoRL 2024. L’anno del DOI, della conferenza o del preprint può essere diverso da quello del riferimento scelto.
- Le voci PPO, DINOv2, Isaac Gym, Isaac Lab e il rapporto sulla valutazione sono qui registrate come preprint/rapporti arXiv. Per DINOv2 è stata conservata la citazione arXiv indicata dal repository ufficiale; in fase di stesura si potrà uniformarla alla versione TMLR. GAE e ViT sono registrati come lavori ICLR, con collegamento al testo su arXiv.
- Per descrivere dettagli di SAC come temperatura appresa e formulazione senza rete di valore separata, consultare anche [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905), da aggiungere se quei dettagli saranno discussi. La bibliografia attuale include la fonte ICML dell’algoritmo.
- Il survey locale non pubblicato non compare nel `.bib`; il duplicato di Liang compare una volta sola. L’esempio `Aarzen1999` è rimasto nel file originale del template ed è estraneo ai 40 lavori proposti.
- Verificata la compilazione di tutte le **41 voci del file** (40 selezionate più l’esempio) in un documento di controllo separato, con `natbib` e lo stile `Classes/RoboticsBiblio.bst`: nessun errore o avviso BibTeX e nessun avviso LaTeX finale. Questo controllo non equivale alla compilazione o revisione dei capitoli della tesi.

Il prossimo passaggio concordato è discutere questa selezione e rivedere la struttura dei capitoli 1 e 2. Le note di implementazione sono in `NOTE_PROGETTO.md`; non costituiscono ancora una descrizione definitiva del metodo o una validazione delle policy.
