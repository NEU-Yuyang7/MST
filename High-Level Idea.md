# High-Level Idea

We build upon Borůvka’s algorithm as the structural backbone, but replace its per-phase full scanning and implicit global ordering with three core ideas inspired by BMSSP:

---

## 1. Bounded Progression (Threshold Advancement)

We maintain a gradually increasing weight threshold \( \tau \).  

At each stage, only edges with weight \( \le \tau \) are *activated* and considered.

This avoids processing the entire edge set at once and instead advances in controlled weight layers.

---

## 2. Partial Ordering Instead of Full Sorting

We maintain a candidate structure \( D \), which stores for each current component \( C \) an estimate of its cheapest outgoing edge:

\[
\hat{w}(C)
\]

along with the corresponding edge.

The structure \( D \) supports the following operations:

- **Pull(M)**: Retrieve up to \( M \) components with the smallest estimated candidate weights (analogous to BMSSP’s `Pull`).
- **Insert/Update(C, candidateEdge)**: Insert or update a component’s candidate edge.
- **BatchPrepend(list)**: Insert a batch of newly discovered smaller candidates (analogous to BMSSP’s `BatchPrepend`).

This replaces global edge sorting with layered, partially ordered processing.

---

## 3. Pivot/Batch Focus (Selective Strong Validation)

At each stage, we only strongly validate the cheapest outgoing edges (COEs) of the small batch of components returned by `Pull`. 

These validated edges are then merged via union operations.

All other components are deferred to later stages.

If merging progress becomes too slow or the candidate structure grows excessively, the threshold \( \tau \) is increased (entering the next weight layer), analogous to BMSSP’s boundary update or partial execution mechanism.

---

# Correctness Invariant (Based on the Cut Property)

For any component \( C \), its true cheapest outgoing edge (COE) is a light edge of some cut and is therefore safe to include in the MST.

This algorithm guarantees that every edge added to the MST is validated as a true COE (via local scan or lazy verification when necessary).

Therefore, correctness follows directly from the cut property.



```
Algorithm BMS-Boruvka(G=(V,E,w)):
    UF.make_set(V)
    MST ← ∅
    τ ← initial_threshold()              // e.g., smallest bucket boundary
    D ← InitializeCandidateStructure()   // supports Pull / Update / BatchPrepend

    // Candidates from very small edges
    for (u,v,wu,v) in ActiveEdges(τ):
        Cu ← UF.find(u), Cv ← UF.find(v)
        if Cu ≠ Cv:
            D.Update(Cu, (u,v,wu,v))
            D.Update(Cv, (u,v,wu,v))

    while UF.number_of_components() > 1:
        // Pull a batch of "most promising" components
        S ← D.Pull(M)                 // |S| ≤ M, components with smallest estimated best
        merged ← 0
        NewlyDiscovered ← ∅

        for C in S:
            // Validate / refresh COE of component C under current τ
            e* ← ValidateCOE(C, τ, UF)        // returns cheapest outgoing edge of C if exists
            if e* == NIL:
                continue

            (u,v,w) ← e*
            Cu ← UF.find(u), Cv ← UF.find(v)
            if Cu == Cv:
                continue    // stale candidate

            UF.union(Cu, Cv)
            MST.add(e*)
            merged ← merged + 1

            // Local relax: activating/updating candidates around merged region
            NewlyDiscovered.add(NeighborsCandidates(Cu ∪ Cv, τ, UF))

        // Batch insert newly discovered smaller candidates (BMSSP-like)
        D.BatchPrepend(NewlyDiscovered)

        // If not enough progress, increase boundary and activate more edges
        if merged < progress_threshold():
            τ ← RaiseThreshold(τ)             // move to next bucket / larger weight range
            for (u,v,w) in NewlyActivatedEdges(τ):
                Cu ← UF.find(u), Cv ← UF.find(v)
                if Cu ≠ Cv:
                    D.Update(Cu, (u,v,w))
                    D.Update(Cv, (u,v,w))

    return MST

```

