# Confronto energetico nuove mappe asfalto/rocce

40 traversate guidate: 10 ripetizioni per percorso su ciascuna delle due mappe. Tutte riuscite e aderenti al percorso. Velocità massima comandata 0,8 m/s, con riduzione/arresto in svolta. Nessun training eseguito.

| Mappa | Percorso asfalto, J (media ± SD) | Percorso rocce, J (media ± SD) | Differenza, J |
|---|---:|---:|---:|
| Rocce a destra | 1353.3 ± 26.8 | 1531.1 ± 28.3 | 177.8 |
| Rocce a sinistra | 1369.1 ± 28.3 | 1534.1 ± 55.0 | 165.0 |

Medie bilanciate: asfalto 1361,2 J; rocce 1532,6 J. Differenza 171,4 J (+12,6% rispetto al costo asfaltato; risparmio asfalto dell’11,2% rispetto al roccioso). Intervallo bootstrap esplorativo 95% della differenza media: 152–200 J, raggruppando le due mappe per ripetizione. Non è una misura di generalizzazione a nuovi terreni.

Lunghezze medie effettive: 7,875 m asfalto e 7,955 m rocce. Tempi: 10,84 s e 11,56 s. Differenza media dei consumi scontati con gamma=0,999: 163,9 J. Il risultato riguarda questi percorsi guidati (non ottimizzati) e questa velocità, non una proprietà energetica universale del materiale.

## Peso energetico

Con costo -lambda * E/100, per un vantaggio energetico scontato B serve lambda=100*B/163,914. Pesi: 6,10 per 10 punti; 15,25 per 25 punti; 61,01 per 100 punti.

Proposta iniziale per il prossimo training: energy_weight=6.0, energy_reference_j=100, bonus goal e penalità fallimento invariati a 100. Vantaggio energetico misurato circa 9,83 punti (vantaggio totale di ritorno scontato circa 10,18, includendo tempo/progresso). È un punto iniziale empirico, non una garanzia di convergenza o priorità vincolata del goal.

Ritorni scontati medi ricalcolati dai dati con reward di test11:

| Peso lambda | Asfalto | Rocce |
|---:|---:|---:|
| 1 | 95,54 | 93,56 |
| 6 | 29,28 | 19,10 |
| 15,25 | -93,33 | -118,68 |
| 61,01 | -699,72 | -800,07 |

Un ritorno negativo non è di per sé un errore. Il rischio è il confronto con un fallimento anticipato, che ha penalità terminale -100 e poco consumo pregresso: con peso 15,25 anche alcune prove asfaltate riuscite scendono sotto -100. Durante un nuovo training percorsi inizialmente meno efficienti possono accentuare questo incentivo indesiderato. Il +100 goal non si somma automaticamente al risparmio energetico come bonus: il risparmio è la differenza fra due penalità.

## Verifiche e file

Integrale meccanico PhysX trapezoidale a 200 Hz, assestamento escluso. Controllati totali energetici/ritorni contro steps.csv per tutti i 40 episodi e hash delle mappe. Velocità residua dopo assestamento <=0,022 m/s. Pose iniziali abbinate con jitter ±5 cm, seed fisso; geometria invariata fra ripetizioni.

episodes.csv: risultati individuali. steps.csv: traiettorie, comandi e joule per passo. analysis.json: statistiche complete. config.json: metodo e hash. Gli USD e il codice sono copiati in questa cartella. Non sono state modificate reward o mappe di test11.
