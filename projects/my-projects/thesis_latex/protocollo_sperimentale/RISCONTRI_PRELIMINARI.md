# Riscontri per preparare il protocollo

Data: 11 novembre 2026. Questa nota mantiene la provenienza dei dati consultati; il documento da condividere con il relatore è proposta_protocollo_sperimentale.pdf.

## Riscrittura breve richiesta dall'autrice

La proposta di sette pagine è stata sostituita da un testo breve in prima persona, senza formule o dettagli dei computer. I numeri S0–R1 della prima proposta non sono più il piano attivo. Restano utili i riscontri tecnici sotto riportati.

Il nuovo piano propone 30 missioni sulla mappa 7×7, 30 grigio–verde simulate e 30 grigio–nero simulate. Per ciascun caso indoor reale sono previste cinque missioni ripetute due volte. I cinque confronti diretti simulati e i due reali sono aggiuntivi. Le dieci esecuzioni PPO speculari sono già disponibili; il primo esempio SAC da nove/dieci prove non è ancora stato identificato con certezza.

L'autrice indica la camera ZED X in simulazione e ZED 2 per il reale e definisce il costo energetico come lavoro meccanico assoluto ai giunti in entrambi i casi. La disponibilità e la lettura delle coppie e velocità reali restano da definire.

La selezione delle nuove missioni di valutazione 7×7 esclude sia ghiaccio sia scale dagli estremi. Non attribuire retroattivamente la stessa esclusione al training di riferimento: il suo codice esclude il ghiaccio dalle partenze e dagli arrivi e le scale solo dagli arrivi.

Il documento descrive SAC per l'incentivo all'esplorazione, senza dedurre dal cambio di algoritmo la causa dei diversi risultati. Descrive geometrie, attriti e texture dei sette terreni come rappresentativi, senza attribuire loro una calibrazione fisica reale non documentata.

Per il telo verde, il maggiore costo fisico reale rimane da verificare. Cartelloni verdi o neri non implicano da soli resistenza o scivolosità: il caso nero con cartelloni è presentato come trasferimento dell'evitamento visivo, non come validazione sul ghiaccio reale.

## Prima proposta, superata: stato e conteggi

Il protocollo è una proposta, non un resoconto di nuove prove. Non sono stati modificati il capitolo 4, i capitoli già scritti, i checkpoint o i programmi di simulazione.

Piano di navigazione: S1 20 esecuzioni, S2 36, S3 18, R1 30. Sono 74 esecuzioni simulate e 30 reali. R0 aggiunge sei attraversamenti reali; S0 può riusare dati compatibili oppure richiede 63 attraversamenti simulati. La caratterizzazione delle due superfici S3, quando mancante, aggiunge sei esecuzioni simulate. Il secondo ambiente Re e le prove aggiuntive sui limiti non sono inclusi in questo budget.

L'autrice ha confermato che la pipeline reale è ancora da definire. Hardware, camera, localizzazione, segnali energetici, controllore e condizioni del luogo rimangono da completare.

## Caratterizzazione dei terreni

Consultati compare_terrain_speeds.py in spot_rl_energy_navigation/terrain_speed_study/ e i CSV e la sintesi terrain_speeds_20261007_005839_combined.md in spot_rl_energy_navigation/results/.

Sono disponibili sette terreni, sei velocità comandate (0,5, 0,7, 0,9, 1,1, 1,3, 1,6 m/s), tre posizioni laterali e tratte di 2 m. Lo script distingue coppia stimata e coppia PhysX e integra i prodotti con le velocità mediante trapezi. Per la tesi usare il segnale PhysX coerente con il capitolo 3.

La sintesi locale riporta sul ghiaccio 0/3 completamenti per ogni velocità: manca un costo J/m su attraversamenti completati. Gli altri materiali hanno prevalentemente 3/3 completamenti, con eccezioni documentate a 0,5 m/s. Il protocollo riduce l'eventuale nuova campagna a tre velocità; il riuso richiede controllo di asset, locomozione, reset e confini temporali.

## Benchmark guidato asfalto–rocce

Consultati README.md e compare.py in spot_rl_energy_navigation/energy_route_benchmark/, e results/20261023_024748_404195/REPORT.md.

Il benchmark conserva 40 attraversamenti guidati: dieci ripetizioni per ramo e per mappa, tutti completati nel report. Il costo medio bilanciato è circa 1361 J sull'asfalto e 1533 J sulle rocce. Questi numeri motivano un confronto guidato, ma **non costituiscono un confronto diretto con la policy speculare finale**.

Le mappe provengono da terrain_generator/manual_asphalt_rocks_10cm, mentre la policy di riferimento usa gli asset congelati di multi_terrain_test13/maps. Il README segnala anche reset e preparazione della postura diversi. Occorre riadattare il confronto alle stesse mappe e allo stesso ambiente della policy.

Il controllore guidato esistente usa guadagno angolare 2, saturazione ±1,5 rad/s, avanzamento modulato col coseno dell'errore e arresto longitudinale oltre 0,6 rad. Il protocollo riprende questi parametri come proposta, senza presentarli come programma già integrato negli ambienti finali.

## Policy sulle mappe speculari

Consultati i cinque summary.json in spot_rl_energy_navigation/multi_terrain_test13/results/.

Le due valutazioni sulle mappe di training (evaluation_20261023_150520_558091 e evaluation_20261023_150824_576455) usano lo stesso policy_final.pt: 5/5 successi per mappa, tutti sul ramo asfaltato, energia media circa 1182 J e 971 J rispettivamente. Le tre nuove disposizioni valutate il 24 ottobre hanno 0/5 successi ciascuna, con uscite o cadute.

Sono utili per studiare la simmetria e analizzare separatamente i limiti sulle nuove disposizioni. Non dimostrano un risparmio verso un riferimento eseguito nelle stesse condizioni o generalizzazione a mappe nuove.

## Mappa estesa

Consultato spot_rl_energy_navigation/train_general_big_map_test7_SAC/analysis/evaluation_stage3_650065_seed20000_20261030/REPORT.md e relativi input, oltre ai confronti di curriculum dello stesso riferimento.

Il checkpoint a 650065 transizioni riporta 6/10 successi, tre cadute e un timeout. È un modello del riferimento metodologico confermato, distinto dalla run iniziale locale da 200000 transizioni. Il modello finale va congelato prima della raccolta; non attribuire risultati di checkpoint diversi a un solo identificativo.

I campioni delle traiettorie sono pre-step, mentre alcune diagnostiche sono post-step. Il nuovo logging deve conservare timestamp e campione terminale, evitando di perdere l'ultimo tratto o associare energia e posa a istanti sbagliati.

## Ambienti destinati alle prove reali

Consultati missions_15.py e README.md nei riferimenti train_general_big_map_test12_SAC_Re e train_general_big_map_test13_SAC_Re. Letti anche i report in train_general_big_map_test11_SAC_Re/analysis/policy_best_9_episodi/ e train_general_big_map_test14_SAC_Re/results/evaluation_20261110_121740_346916/analysis/.

Le suite esistenti fissano start, goal e yaw, oltre alla mappa e al suo hash. Non basta cambiare seed per ottenere una ripetizione diversa: SuiteSampler restituisce la missione salvata. Il protocollo propone tre yaw iniziali espliciti e abbinati, da implementare nel manifest delle nuove prove.

I report degli altri riferimenti aiutano a valutare distanze e categorie, ma non sono evidenze dei modelli Re confermati dall'autrice. Nel documento per il relatore non vengono trasferiti risultati fra questi modelli.

La proposta limita inizialmente il reale a una scena con due superfici e contrasto di resistenza al moto; il caso scivoloso rimane una condizione ulteriore. M1–M3 devono essere definite sulla geometria realizzabile e riprodotte in simulazione, identificando eventuali nuove disposizioni.

## Logging e lavoro necessario

Nei file artifacts.py, play.py, evaluation.py e falls.py della mappa estesa sono disponibili numerose metriche di episodio, energia, fase, traiettorie e cadute. Non vengono prodotti automaticamente tutti i campioni articolari a 200 Hz e tutte le colonne del protocollo.

Prima delle prove occorrerà adattare i controllori di riferimento agli stessi ambienti e reset, fissare il manifest delle missioni, estendere il logging e definire la pipeline reale. Nessuno di questi interventi è stato eseguito durante la preparazione del documento.

Il risparmio è calcolato sulle coppie che completano lo stesso compito, riportandone il numero insieme ai successi/tentativi. I dati di tutti i tentativi del piano restano registrati; le figure illustrative sono distinte dai riepiloghi quantitativi.
