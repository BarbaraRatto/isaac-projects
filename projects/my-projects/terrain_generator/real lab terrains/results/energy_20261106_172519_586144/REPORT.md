# Confronto energetico grigio/verde

| Velocità comandata (m/s) | Materiale | Tratti completati | J/m avanzamento, media ± deviazione standard | Velocità effettiva (m/s) |
|---:|---|---:|---:|---:|
| 0.90 | grey | 5/5 | 147.7 ± 0.0 | 0.914 |
| 0.90 | green | 5/5 | 152.9 ± 0.0 | 0.908 |

| Velocità (m/s) | Coppie completate | Verde − grigio (J/m) | Aumento medio verde (%) |
|---:|---:|---:|---:|
| 0.90 | 5 | +5.2 | +3.5% |

## Interpretazione

- Misura meccanica dai giunti, non consumo elettrico della batteria.
- J/m di avanzamento usa la stessa distanza longitudinale per entrambi i materiali. trials.csv contiene anche J/m di percorso reale.
- La misura esclude assestamento e rincorsa; include eventuali arresti, oscillazioni o retromarcia fra i due traguardi.
- Confrontare anche velocità effettiva e frazione di tratti completati: un terreno che fa cadere Spot non soddisfa lo scopo del test.
- Il confronto a pari comando non è necessariamente a pari velocità reale. Se differiscono molto, ripetere regolando i comandi.
- Differenze piccole rispetto alla dispersione non dimostrano un costo maggiore. Ripetizioni deterministiche non sono prove statisticamente indipendenti.
