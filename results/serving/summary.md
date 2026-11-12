# Served model on the C-MAPSS test sets (source FD001, seed 0, 3-member ensemble, 90% conformal intervals)

One request per test subset, through the same Predictor the container serves. `auto` switches to health-anchored normalization when the operating-settings PSI exceeds 0.25.

| target | conditions (estimated / true) | PSI settings | PSI sensors | drift | requested | used | RMSE | coverage |
|---|---|---|---|---|---|---|---|---|
| FD001 | 1 / 1 | 0.00 | 0.35 | major | none | none | 14.90 | 0.90 |
| FD001 | 1 / 1 | 0.00 | 0.35 | major | anchored | anchored | 15.92 | 0.91 |
| FD001 | 1 / 1 | 0.00 | 0.35 | major | auto | none | 14.90 | 0.90 |
| FD002 | 6 / 6 | 5.38 | 4.63 | major | none | none | 66.91 | 0.27 |
| FD002 | 6 / 6 | 5.38 | 4.63 | major | anchored | anchored | 23.23 | 0.87 |
| FD002 | 6 / 6 | 5.38 | 4.63 | major | auto | anchored | 23.23 | 0.87 |
| FD003 | 1 / 1 | 0.00 | 1.32 | major | none | none | 42.36 | 0.82 |
| FD003 | 1 / 1 | 0.00 | 1.32 | major | anchored | anchored | 42.55 | 0.91 |
| FD003 | 1 / 1 | 0.00 | 1.32 | major | auto | none | 42.36 | 0.82 |
| FD004 | 6 / 6 | 5.37 | 5.10 | major | none | none | 63.79 | 0.30 |
| FD004 | 6 / 6 | 5.37 | 5.10 | major | anchored | anchored | 35.21 | 0.82 |
| FD004 | 6 / 6 | 5.37 | 5.10 | major | auto | anchored | 35.21 | 0.82 |
