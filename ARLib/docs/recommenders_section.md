# Sezione: Modelli di Raccomandazione

## 1. Panoramica
Questa sezione descrive in dettaglio i dieci modelli di raccomandazione utilizzati negli esperimenti. Per ciascuno vengono forniti: idea chiave, formulazione matematica, pipeline di training, iperparametri tipici, complessità, vulnerabilità agli attacchi adversariali, segnali di compromissione e strategie di difesa.

**Modelli analizzati:** GMF, NCF, WRMF, LightGCN, NGCF, NCL, SimGCL, XSimGCL, SGL, SSL4Rec.

---
## 2. Schede Dettagliate dei Modelli

### 2.1 GMF – Generalized Matrix Factorization
**Idea chiave:** Estende la Matrix Factorization classica usando un layer neurale sulla combinazione Hadamard tra embedding utente e item.

**Formulazione:**  
Embedding: `p_u, q_i ∈ R^d`  
Interazione: `h_{ui} = p_u ⊙ q_i`  
Logit: `s_{ui} = w^T h_{ui} + b`  
Predizione (implicit): `ŷ_{ui} = σ(s_{ui})`

**Loss (BCE / Negative Sampling):**  
`L = - Σ_{(u,i)∈D+} log σ(s_{ui}) - Σ_{(u,j)∈D-} log (1 - σ(s_{uj})) + λ (||p_u||² + ||q_i||² + ||w||²)`  
Alternativa BPR: `L_BPR = - Σ log σ(s_{ui} - s_{uj}) + λΩ`

**Pipeline Training:**  
1. Campionamento coppie (u,i+) e negative j  
2. Forward h_{ui}, s_{ui}  
3. Loss BCE/BPR  
4. Backprop (Adam)  
5. Early stopping su NDCG/HR

**Iperparametri tipici:** d=32–128, neg=4–10, lr=1e-3, batch=512, λ=1e-4–1e-3.  
**Complessità:** O(|D|·d)  
**Vantaggi:** Semplice, efficiente.  
**Limiti:** Interazioni solo bi-lineari.  
**Vulnerabilità:** Bandwagon (co-occorrenze artificiali), PoisonRec (embedding drift), PGA (gradient shaping).  
**Segnali di attacco:** Similarità media nuovi utenti ↑, varianza embedding ↓, target item sale globalmente.  
**Difese:** Clipping norme, filtraggio utenti anomali, regolarizzazione decorrelativa.

---
### 2.2 NCF – Neural Collaborative Filtering
**Idea:** Combina fattorizzazione e MLP per modellare interazioni non lineari.

**Architettura:**  
`z_G = p_u ⊙ q_i` (ramo GMF)  
`z_0 = [p_u || q_i] → ... → z_L` (ramo MLP)  
Fusione: `h = concat(z_G, z_L)`  
Output: `ŷ_{ui} = σ(w^T h + b)`

**Loss:** BCE o BPR.  
**Iperparametri:** Layer MLP: [2d, d, d/2], dropout 0.2–0.5, lr=1e-3.  
**Complessità:** O(B·d²)  
**Vantaggi:** Elevata espressività.  
**Limiti:** Rischio overfitting.  
**Vulnerabilità:** DLAttack, PoisonRec (profili realistici sintetici).  
**Segnali:** Kurtosi attivazioni ↑, gradient norm outlier per nuovi utenti.  
**Difese:** Dropout, spectral norm, mixup utenti.

---
### 2.3 WRMF – Weighted Regularized Matrix Factorization
**Scenario:** Feedback implicito (click, view).  
**Obiettivo:** Ponderare osservazioni e non-osservazioni.

**Funzione:**  
`L = Σ c_{ui}(p_u^T q_i - r_{ui})² + λ(||p_u||² + ||q_i||²)` con `c_{ui} = 1 + α r_{ui}`

**Ottimizzazione:** ALS (risoluzione locale chiusa).  
**Complessità:** O((|U|+|I|) d³) per iterazione (riducibile con solutori iterativi).  
**Vantaggi:** Robusto su implicit feedback.  
**Limiti:** Lineare, ignora struttura del grafo.  
**Vulnerabilità:** Flooding di falsi positivi (Random/Bandwagon).  
**Segnali:** Aumento brusco densità interazioni utente→item popolari.  
**Difese:** Capping c_{ui}, smoothing logaritmico dei pesi.

---
### 2.4 LightGCN
**Idea:** Semplifica GCN per CF: solo propagazione lineare di segnali.

**Propagazione:**  
`e_u^{(k+1)} = Σ_{i∈N(u)} 1/√(|N(u)||N(i)|) e_i^{(k)}`  (simmetrico per item)

**Aggregazione:** `e_u = Σ_{k=0}^K α_k e_u^{(k)}`  
**Predizione:** `ŷ_{ui} = e_u^T e_i`  
**Loss:** BPR.

**Vantaggi:** Semplice, performante su sparsità.  
**Limiti:** Sensibile a edge avvelenati.  
**Vulnerabilità:** GTA, GOAT, BiLevel (edge injection mirata).  
**Segnali:** Cambiamento assortatività, spike degree su item target.  
**Difese:** Edge filtering (Jaccard), neighbor sampling robusto, removal high-entropy nodes.

---
### 2.5 NGCF – Neural Graph Collaborative Filtering
**Estensione:** Introduce messaggi non lineari e interazione bilineare.

**Aggiornamento:**  
`m_u^{(k)} = Σ_{i∈N(u)} 1/√(|N(u)||N(i)|) [ W₁ e_i^{(k)} + W₂ (e_i^{(k)} ⊙ e_u^{(k)}) ]`  
`e_u^{(k+1)} = σ(m_u^{(k)})`

**Vantaggi:** Cattura segnali di alto ordine e interazioni feature-level.  
**Limiti:** Più pesante, rischio over-smoothing.  
**Vulnerabilità:** PoisonRec (rafforza segnali spuri), PGA.  
**Segnali:** Saturazione attivazioni (ReLU dead units), embedding collapse parziale.  
**Difese:** Layer dropout, residual connections leggere, smoothing constraints.

---
### 2.6 NCL (Neighborhood Contrastive Learning)
**Idea:** Combina raccomandazione + contrastive learning tra viste perturbate.

**Loss:** `L = L_rec + λ L_contrast`  
`L_contrast` (InfoNCE) spinge vicini autentici e separa non vicini.

**Vantaggi:** Migliora discriminatività e robustezza.  
**Limiti:** Se vicinati sono artificiosamente costruiti, il contrasto fallisce.  
**Vulnerabilità:** Cluster sintetici (GOAT), InfoAttack.  
**Segnali:** Silhouette score cluster improvviso ↑, densità interna artificiale.  
**Difese:** Hard negatives strutturali, τ adattivo, pruning vicinati anomali.

---
### 2.7 SimGCL – Simple Graph Contrastive Learning
**Idea:** Usa perturbazione minimalista sugli embedding per contrastive learning.

**Perturbazione:** `e' = e + ε sign(e)`  
**Loss:** InfoNCE (alignment + uniformity).

**Vantaggi:** Leggero, efficace.  
**Limiti:** Perturbazione prevedibile.  
**Vulnerabilità:** PGA (riduce margini), BiLevel (forza collasso).  
**Segnali:** Uniformity peggiora (spettro covarianza più concentrato).  
**Difese:** Adaptive ε, penalità spectral (λ_max ratio).

---
### 2.8 XSimGCL
**Estensioni:** Multi-perturbation (gauss, dropout, sign), adaptive temperature, normalizzazioni.

**Obiettivo:** Ridurre fragilità single-noise.

**Vulnerabilità:** PoisonRec multi-step, BiLevel (ottimizzazione su tutte le viste).  
**Segnali:** Divergenza tra viste ↓ troppo velocemente.  
**Difese:** Consistency regularization, detection varianza inter-view.

---
### 2.9 SGL – Self-Supervised Graph Learning
**Augmentations:** Edge dropout, node dropout, random walk subgraph.

**Loss:** `L = L_rec + λ Σ_v L_contrast^{(v)}`  
**Vantaggi:** Migliora cold-start & generalizzazione.  
**Limiti:** Edge dropout può mascherare edges malevoli persistenti.  
**Vulnerabilità:** GOAT, GTA (inserzione edges resilienti).  
**Segnali:** Persistenza edges nei sample di viste > baseline.  
**Difese:** Degree-aware dropout, edge persistence scoring.

---
### 2.10 SSL4Rec
**Framework generico multi-task** (ranking + contrast + predictive / masked modeling).

**Loss:** `L = L_rank + λ₁ L_contrast + λ₂ L_pred`  
**Vantaggi:** Flessibilità, sinergia segnali.  
**Limiti:** Superficie d’attacco ampia (multi-modulo).  
**Vulnerabilità:** InfoAttack (coerenza multi-view), BiLevel (co-ottimizzazione poisoning).  
**Segnali:** Disagreement multi-view ↓, gradient alignment artificiale ↑.  
**Difese:** View disagreement penalty, gradient isolation forest.

---
## 3. Tabella Comparativa Sintetica
| Modello | Tipo | Punti di Forza | Limite Principale | Vulnerabilità Chiave | Difesa Primaria |
|---------|------|----------------|-------------------|----------------------|-----------------|
| GMF | MF neurale | Semplicità, velocità | Capacità limitata | Bandwagon / PoisonRec | Filtering + clipping |
| NCF | Deep hybrid | Interazioni non lineari | Overfitting | DLAttack / PoisonRec | Spectral norm + mixup |
| WRMF | Implicit MF | Stabilità | Lineare | Flooding rating | Confidence capping |
| LightGCN | Graph light | Scalabilità | Edge sensitivity | GTA / GOAT | Edge score pruning |
| NGCF | Graph deep | Relazioni complesse | Over-smoothing | PoisonRec / PGA | Layer dropout |
| NCL | Contrastive | Robustezza semantica | Cluster sintetici | GOAT / InfoAttack | Hard negatives |
| SimGCL | Simple contrastive | Efficienza | Perturbazione semplice | PGA / BiLevel | Spectral penalty |
| XSimGCL | Multi-contrast | Ridondanza | Complessità | PoisonRec | View consistency |
| SGL | Multi-view graph | Cold-start | Edge persistence | GOAT / GTA | Degree-aware dropout |
| SSL4Rec | Multi-task SSL | Flessibilità | Superficie attacco ampia | InfoAttack / BiLevel | Disagreement regularization |

---
## 4. Osservazioni Globali
1. I modelli graph-based (LightGCN, NGCF, SGL) offrono performance elevate ma aprono vettori di attacco strutturali (edge poisoning).  
2. I modelli contrastive (SimGCL, NCL, XSimGCL) migliorano robustezza ma possono essere aggirati con attacchi che creano coerenza artificiale.  
3. I modelli multi-task (SSL4Rec) richiedono difese multi-livello perché l’avversario può sfruttare sinergie tra obiettivi.  
4. I modelli classici (GMF, WRMF) restano baseline importanti e mostrano pattern di degrado più prevedibili.  

---
## 5. Collegamento agli Attacchi
Questa sezione è propedeutica alla successiva: mappando vulnerabilità strutturali qui identificate, nella prossima parte categorizzeremo i 18 attacchi (black / gray / white box) mostrando come sfruttano:  
- Manipolazione delle co-occorrenze (GMF/NCF)  
- Alterazione topologica (LightGCN/NGCF/SGL)  
- Collasso contrastivo o falsi cluster (NCL/SimGCL/XSimGCL)  
- Incoerenza multi-view (SSL4Rec)  

---
(Se desideri, posso ora creare la sezione sugli attacchi con analisi similare.)
