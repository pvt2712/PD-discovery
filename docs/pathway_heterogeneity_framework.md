# Does every patient progress through the same pathway? Subgroup causal discovery for Parkinson's disease progression

> Investigation report, 2 November 2026, for the investigator, supervisors and the clinical panel. It asks three things: how subgroup causal discovery can test whether patients with Parkinson's disease (PD) progress through different pathways; which framework suits PPMI; and which tests would show novelty and clinical significance. It rests on three synthetic simulations and a literature review; §2 was updated on 3 November 2026 with the person-level test at larger cohort sizes. No PPMI data were read. It builds on protocol draft v0.3 (`PROTOCOL.md`), which stays the protocol of record until the investigator decides (§10).

## Summary

**The clinical question.** Clinicians want to know whether patients decline through *different pathways* or through *the same pathway at different speeds*. The answer decides whether care should be personalised by *what* to monitor or by *how often*. A pathway has four parts:
- how fast the disease moves (speed);
- where it pushes hardest (load);
- how strongly one domain drives another (strength);
- which links operate at all (structure).

"Does every patient have the same pathway?" therefore becomes: once speed and load are allowed to differ, do patients still differ in strength or structure?

**A problem that changes the design.** In yearly data, a group that moves faster along the same causal pathway shows stronger links and extra indirect links. We simulated 500 synthetic cohorts sharing one mechanism, with the PIGD group running twice as fast.
- Link-by-link comparison declared phenotype differences in every cohort.
- A link that exists only through an intermediate domain looked three times stronger in the fast group.
- Protocol v0.3's speed-or-pathway rule would have reported these as pathway differences.

A *speed-only test* asks whether one group's yearly dynamics equal the other group's run faster.
- When only speed differed, it rejected 5.6% of the time.
- When one real link was added, it rejected 91% of the time, and it pointed to that link in 97% of cohorts.
- Speed also varies between patients within a phenotype. In PPMI-sized synthetic cohorts, a test that ignored this declared a mechanism difference twice as often as its 5% rate when none existed, and far more often in larger cohorts. A test that gives each person their own speed stayed at 5%, also in cohorts 16 times larger, so the real design uses it (§2).

**Recommended framework: speed-adjusted subgroup causal discovery.** A latent-aware method learns one shared causal map. On that map, a transition model gives every patient their own speed and load. Heterogeneity in the links is then tested at three resolutions:
1. known phenotypes: TD versus PIGD, and REM-sleep behaviour disorder (RBD) status as a proxy for body-first disease, which the panel should consider;
2. data-driven mechanism classes, which group patients by how their domains drive each other, not by how severe they are;
3. individual variation in speed and in a few key links.

Every test is calibrated against a null fitted to the cohort. In that null every patient shares the same mechanism, patients differ in speed and load, and the artefacts of defining phenotypes from graph variables are present.

**Primary tests.**
- Do the phenotypes differ in mechanism beyond speed and load?
- Do data-driven progression classes differ in mechanism beyond speed and load?
- Link-level tests follow only if the phenotype test is positive.
- A methodological test shows, at PPMI's sizes, that naive comparisons read speed as mechanism and the speed-adjusted tests do not.

**Why it is new.** A second search on 2 November 2026 (addendum: `methods_and_novelty_update.md`) confirmed the gap, but in a narrower form than first stated.
- Comparing causal or temporal networks between patient groups is not new in itself. It has been done for Alzheimer's stages, psychiatric trajectory classes, boys and girls in a birth cohort, and molecular cancer subtypes. All of these compare dynamics directly, and several define groups by stage, outcome or speed, which is the trap of §2.
- In PD, directed models fitted by phenotype are confirmatory and mostly cross-sectional or based on brain imaging. No study has used causal discovery on longitudinal clinical data to compare phenotypes. Most PD evidence concerns speed and the order of events, not which domain drives which.
- No study tests whether mechanisms differ once groups and persons may differ in speed and load, against a null calibrated for those differences and for the artefacts of defining groups. No test of a shared time scale exists for multivariate causal mechanisms.

**Why it matters clinically.**
- If only speed and load differ, one map serves every patient, and care is personalised by review interval and by which domains to emphasise.
- If mechanisms differ, monitoring targets and referrals should differ by phenotype or class.
- Either way, clinicians learn how much of the real heterogeneity their phenotype labels capture.

## 1. What "pathway" means, and what PPMI can answer

| Component | What differs between patients | Example | What would change in care |
|---|---|---|---|
| State | Where the patient is at the visit | More advanced disease | Stage-appropriate care; not a pathway difference |
| Speed | How fast the same dynamics run | PIGD passes through the same states in less time | Shorter review intervals; earlier planning |
| Load | Where the disease pushes hardest: a steady extra push on some domains | Cognition worsens faster at every level of motor severity | Domain-specific screening |
| Strength | How strongly one domain drives another, beyond what speed explains | Gait problems drive loss of independence more strongly in one group | Earlier falls prevention and physiotherapy |
| Structure | Whether a link operates at all | Cognition drives independence in one group and negligibly in the other | Different monitoring targets and referrals |

**Same pathway** means equal strength and structure; speed and load may differ. **Different pathway** means a difference of strength or structure that speed and load do not explain. The framework reports all four components and its tests keep them apart.

Put simply, on the same pathway a fast patient's course is a slower patient's course played at a higher speed. On a different pathway, the course itself differs: from the same state, the domains push each other in different directions.

**What yearly data can and cannot show.** With 2 to 9 yearly visits per person, no individual's own map can be estimated. PPMI can show:
- whether known phenotypes differ in mechanism;
- whether hidden groups with different mechanisms exist;
- how much patients vary in speed and in a few pre-specified links, and how much of that variation phenotypes explain.

The question "does every patient have the same pathway?" is therefore answered as "how much do pathways vary between patients, and do phenotypes or data-driven classes capture that variation?"

## 2. In yearly data, speed looks like a different pathway

**Why.** Suppose two groups share one causal mechanism and one group runs it s times faster. Their yearly effects are then related by a matrix power: B(fast) = B(slow)^s, where B holds the year-to-year effects between domains. Two consequences follow:
- every direct link changes with speed. In the example below it is about 1.6 times stronger at twice the speed. The change is proportional only for small effects and bends for large ones;
- effects that pass through an intermediate domain within the year (gait → daily function → independence) appear as yearly "links" and grow faster than in proportion.

Both groups' true yearly maps contain the same links. A link that seems to be "missing" in the slow group is too weak there to be detected.

This is the time-step problem of the protocol's §1.3 (point 5), now arising between groups. It is known in two other literatures:
- causal discovery from time series sampled more coarsely than the causal process, where sampling adds an edge for every path that fits within the interval (Plis 2015; Hyttinen 2017; Gong 2015);
- panel models, where yearly coefficients depend on the interval and are total rather than direct effects (Voelkle 2012; Kuiper & Ryan 2018; Ryan & Hamaker 2022; Driver 2025).

We found no published test of whether two groups or persons share one time scale. The nearest analogues pool studies whose intervals are known and differ (Kuiper & Ryan 2020; Dormann 2020).

**Synthetic check.**
- Six domains with seven direct links.
- 800 TD and 250 PIGD patients, with five yearly transitions each.
- One shared continuous-time mechanism, with PIGD at twice the speed.
- A second scenario adds one real PIGD-only link (axial → independence).
- 500 cohorts per scenario.
- Hidden burden and phenotype-definition artefacts are left out on purpose, to isolate the speed effect. The real design handles those with the definition-matched null.
- Speed is the same for every patient within a group. The next check relaxes this.

| Result (500 synthetic cohorts per scenario) | Speed only | Speed plus one real extra link |
|---|---|---|
| Cohorts with at least one link declared different (link-wise tests, Benjamini–Hochberg 5%) | 100% | 100% |
| Links declared different, mean of 30 | 4.8 | 6.0 |
| Links classed "stronger in PIGD" by v0.3's rules, mean | 4.4 | 4.9 |
| Links significant in one group only (naive split), mean | 3.2 | 3.4 |
| Speed-only test rejects (Monte Carlo standard error) | 5.6% (1.0) | 90.8% (1.3) |
| Estimated speed (true value 2) | 2.01 | 2.15 |
| Largest speed-adjusted deviation falls on the true extra link | — | 97% of cohorts |

Mean yearly effects, TD versus PIGD, in the speed-only scenario:
- daily function → independence: 0.21 versus 0.34;
- axial → daily function: 0.16 versus 0.27;
- axial → independence, which has no direct link: 0.021 versus 0.068.

**Speed also varies within phenotypes.** Patients with the same phenotype do not all progress at the same speed. A second synthetic check asks whether the speed-only test still works when they do not.
- Three domains in a chain: axial → daily function → independence. The second scenario again adds one real PIGD-only link, axial → independence.
- 400 TD and 200 PIGD patients, with five yearly transitions each.
- Each patient has their own speed. The typical PIGD patient is twice as fast as the typical TD patient. Within each phenotype, the middle 95% of patients run at between about 0.5 and 2.2 times their phenotype's typical speed (standard deviation of log-speed 0.4).
- Two settings: slow dynamics, like persistent clinical scales, and fast dynamics, with stronger links and larger indirect effects within the year.
- 100 cohorts per setting with speed only, and 50 with the extra link.
- Two speed-only tests are compared:
  - the group-level test of the first check, which gives every patient in a phenotype the same speed;
  - a person-level test, which gives each patient their own speed as a random effect and asks whether the phenotype acts only on speed and load, or on the links as well.

| Result (synthetic cohorts) | Slow dynamics | Fast dynamics |
|---|---|---|
| Speed only: group-level test rejects (Monte Carlo standard error) | 12% (3.2) | 9% (2.9) |
| Speed only: person-level test rejects | 7% (2.6) | 3% (1.7) |
| Speed only: mean test statistic, group-level / person-level (8 when calibrated) | 9.0 / 8.2 | 9.7 / 8.4 |
| Speed only: estimated speed ratio, group-level / person-level (true value 2) | 1.96 / 2.03 | 1.93 / 2.01 |
| Extra link: group-level test rejects | 84% (5.2) | 98% (2.0) |
| Extra link: person-level test rejects | 88% (4.6) | 100% (50 of 50) |

- Across both settings, the person-level test rejected 10 of 200 speed-only cohorts (5.0%). The group-level test rejected 21 of 200 (10.5%), twice the nominal rate (one-sided binomial p = 0.001).
- With the extra link, the person-level test was at least as powerful.
- The group-level test fails because patients who run at different speeds, averaged together, no longer follow the matrix-power relation. The misfit is systematic, so it grows with the cohort. New speed-only cohorts were analysed with both tests. The person-level test was added on 3 November 2026, once exact gradients made it fast enough.

| Cohort size (TD + PIGD), five transitions each | 600 | 2,400 | 9,600 |
|---|---|---|---|
| Group-level test rejects, slow dynamics (Monte Carlo standard error) | 6.5% (1.7) | 10.5% (2.2) | 27% (4.4) |
| Group-level test rejects, fast dynamics | 12.5% (2.3) | 19% (2.8) | 65% (4.8) |
| Person-level test rejects, slow dynamics | 4.5% (1.5) | 5.0% (1.5) | 3.0% (1.7) |
| Person-level test rejects, fast dynamics | 5.0% (1.5) | 3.5% (1.3) | 7.0% (2.6) |
| Cohorts per cell | 200 | 200 | 100 |

- The first column repeats the setting of the table above with new cohorts, so it differs from it only by Monte Carlo error.
- PPMI's two cohorts hold roughly 5,000 yearly transitions, between the first two columns (3,000 and 12,000), and have more domains.
- The person-level test stayed calibrated at every size. It rejected 46 of 1,000 cohorts (4.6%), with a mean likelihood ratio of 7.9–8.3 (8 expected); the script is `toy_random_speed_scale_person.py`.
- The real design therefore gives each patient their own speed (Step 2, §5).

**Consequences for protocol v0.3.**
1. Its speed-or-pathway rule (§7.5) counts only a level difference as speed. A phenotype that runs the same mechanism faster would be reported as a pathway difference of strength.
2. Its strength difference δ (§2.3) compares yearly effects directly. It needs a speed-adjusted version: the PIGD effect minus the TD effect, after TD's dynamics are run at PIGD's speed.
3. Its separate per-phenotype graphs (Analysis B) inherit the problem: indirect links that are stronger in the faster phenotype look structural.
4. Its simulation has no scenario in which phenotypes differ only in time scale; scenario S1 varies burden, not speed.

## 3. What is known about pathway heterogeneity in PD

| Evidence | What it shows about "same pathway?" | What it leaves open |
|---|---|---|
| Brain-first versus body-first (the SOC model; Borghammer 2019, 2021) | Body-first disease starts in the gut, with RBD and autonomic loss before motor signs. It carries more pathology at diagnosis and so is predicted to progress faster. Imaging partly supports the model (Horsager 2020; Knudsen 2021); other studies do not (Lövdal 2024; Banwinkler 2022, in PPMI) | Whether domains drive each other differently, or the disease only starts elsewhere and runs faster |
| RBD and progression in PPMI | RBD-positive patients decline faster in cognition (Chahine 2016), in motor scores and cognition (Pagano 2018), and in motor, axial and cognitive scores with more dementia (Passaretti 2025). With RBD, non-motor change comes earlier and motor change later (Di Folco 2024, disease course mapping) | Speed, order or mechanism? The RBD questionnaire is weak in de novo PD: sensitivity 0.44 and specificity 0.84 at a score of 6 or more (Halsband 2018) |
| Data-driven progression subtypes | Nearly all differ in pace: fast, moderate and slow (Fereshtehnejad 2015a, 2017; Lawton 2018; Dadu 2022; Birkenbihl 2023; Su 2024). Across three cohorts, baseline differences were small and speed differences large (Hähnel 2024). A few show domains diverging (Krishnagopal 2020; Myers 2021) or non-sequential states (Severson 2021) | Whether pace subtypes also differ in mechanism |
| Sequence models | One shared order of events (Oxtoby 2021). Two opposite orders in a Chinese cohort, replicated in PPMI: RBD and autonomic change first in one, smell and cognitive change first in the other (Zhou 2023a). Three trajectories of α-synuclein pathology at autopsy (Mastenbroek 2024). Atrophy subtypes add little beyond severity (Shawa 2025) | Order of events is not the same as which domain drives which |
| Disease course mapping and latent-time models | Each patient has a time shift, a speed and a "space shift" (a change in the order of markers). In PPMI, the space shifts recover TD-like and PIGD-like patterns (Poulet & Durrleman 2023). A latent-time model in PPMI found 22% slower progression in women and 25% slower in LRRK2 G2019S carriers, in effect a speed-only model (Jin 2024) | Describes trajectories, not which domain drives which. A space shift can arise from different starting states under one mechanism, so it does not prove a different pathway |
| Cross-lagged models | In PPMI: depression precedes cognitive decline (Jones 2019); depression and self-reported motor function predict each other (Hodgson 2026); cognition precedes loss of daily activities (Cui 2026). In DeNoPa, cognition predicts later depression, the opposite of PPMI (Schroeders 2022) | None compared phenotypes or let the paths vary between patients; directions differ between cohorts |
| Stability of phenotypes | Within a year, 39% of PIGD and 18% of TD patients in PPMI change class (Simuni 2016). Patients drift from TD towards PIGD, and the differences fade by year 4 (Lee 2019). Diffuse-malignant status rises with disease duration (Erro 2020) | Phenotypes may partly be stages or speeds |
| α-synuclein seed amplification | Negative patients have fewer non-motor features (Brooker 2026; Mastrangelo 2026) but a similar course (Schumacher 2025) | Pathways untested; groups small |

**Reading of the evidence.**
- Published PD heterogeneity is mostly about *speed* (pace subtypes) and *order* (sequence models), not about *mechanism*.
- Both can arise without any mechanism difference. A faster patient passes through the same states sooner. A patient whose disease starts elsewhere has a different *load*, and so a different order, even if the domains drive each other in the same way.
- The SOC model itself proposes different starting points and speeds along one shared connectome.
- So the question that matters for care is open: do the domains drive each other differently? It is also testable. For the RBD axis, the SOC model predicts differences in load and speed with shared propagation.

**Group sizes in PPMI de novo PD, from published papers.**
- Motor phenotype: TD 68–71%, PIGD 18–23%, indeterminate 9–11% (Simuni 2016; Lee 2019).
- RBD questionnaire positive: 38% at a score of 5 or more (Marek 2018) and 25.5% at 6 or more (Chahine 2016).
- An RBD axis therefore has a larger minority group than PIGD, but more misclassification.

## 4. Ways to apply subgroup causal discovery

The methods fall into four families by resolution: known groups, latent groups, continuous moderators and individuals. Each family was judged against PPMI's shape: many patients, 2 to 9 yearly visits, about ten domains, hidden person-level burden, and phenotype labels computed from graph variables.

| Resolution | Approach | Examples and software | Fit for PPMI | Role in the recommended framework |
|---|---|---|---|---|
| Known groups | Separate graphs per group, compared against a null | Tiered FCI per phenotype (v0.3 Analysis B) | Possible. Power is low, and the null must contain the definition artefacts | Transparent reference |
| Known groups | Joint estimation: shared structure plus sparse group deviations | Data-shared lasso (Gross & Tibshirani 2016; Ollier & Viallon 2017); joint lasso (Dondelinger & Mukherjee 2020); joint Granger networks (Skripnikov & Michailidis 2019); fused and Bayesian multiple DAGs (Wang 2020; Oates 2016; Castelletti 2020; no maintained packages) | Good for screening with glmnet; needs a patient-level bootstrap | Core of speed-adjusted joint discovery (Step 2), with a speed-scaled reference in place of equality |
| Known groups | Multi-group panel models with random intercepts and equality tests | Multi-group random-intercept cross-lagged panel model (Hamaker 2015; Mulder & Hamaker 2021; lavaan, Mplus) | Good. Handles stable person-level burden; confirmatory | Confirmatory link tests (Step 5), with the speed relation added |
| Known groups | Difference graphs | DCI (Wang 2018; Belyaeva 2021); iSCAN (Chen 2023) | Possible, as sensitivity. Tiers give the shared order, but no hidden confounding is allowed | Sensitivity analysis |
| Known groups | Context methods | JCI (Mooij 2020); CD-NOD (Huang 2020); J-PCMCI+ (Günther 2023); invariant causal prediction (Peters 2016) | JCI-FCI with exogeneity only suits a shared map. The others need unconfounded contexts or long series | JCI-FCI is Step 1 |
| Latent groups | Mixtures of transition models with membership per patient | Grouped mixture regression (flexmix: Leisch 2004; Grün & Leisch 2008); ℓ1-penalised mixtures (Städler 2010); finite-mixture SEM (Jedidi 1997) | Good, for exploration, with about seven or more visits per person (§8). Choosing the number of classes is fragile (Bauer & Curran 2003; Nylund 2007) | Step 3, in a speed-class and a mechanism-class version |
| Latent groups | Mixture-of-DAG discovery and clustering by mechanism | Saeed 2020; Strobl 2023; Huang 2019; MCD (Varambally 2024); latent-class VAR (Ernst 2020, 2024); S-GIMME (Gates 2017; Lane 2019) | Poor to possible. These need long series per person or independent rows, and hidden burden is mistaken for class membership | Diagnostic only |
| Continuous moderators | Varying-coefficient models; model-based recursive partitioning; SEM trees and forests; moderated networks; generalized random forests | mgcv (Hastie & Tibshirani 1993); partykit (Zeileis 2008); semtree (Brandmaier 2013, 2016); mgm (Haslbeck 2021); grf (Athey 2019) | Good for exploration. Artefacts arise when a moderator is itself a graph variable | Supportive: baseline tremor and gait scores as continuous moderators |
| Individuals | Person-specific networks | GIMME (Gates & Molenaar 2012); IMaGES (Ramsey 2010); multilevel VAR with random slopes (Epskamp 2018) | Poor. These need about 50 to 100 occasions per person (Schultzberg & Muthén 2018) | Not used. Individual variation enters through random speed and load (Step 4) |

**Design lessons.**
1. **What annual data support.** Yearly panels support average pathways, how those pathways vary with patient characteristics, and latent classes. They do not support individual maps (Fisher 2018).
2. **Remove burden first.** Stable person-level burden must be removed before groups are compared, using random intercepts or latent centering. Centering on each person's observed mean is biased by about 1/T at 2 to 9 visits (Nickell 1981; Hamaker & Grasman 2015).
3. **Separate speed from mechanism.** Comparing yearly effects confuses the two (§2). Every comparison therefore includes a speed for each person, not one speed per group.
4. **No double dipping.** Pathways must not be compared between classes found in the same data using naive tests (Kriegeskorte 2009; Gao 2024). Two routes avoid it:
   - test inside the joint mixture likelihood, with a bootstrap that repeats the class search;
   - estimate class-specific links by sample splitting.
5. **Resample patients, not transitions.**
6. **Replicate across sub-cohorts.** PPMI 1.0 and PPMI 2.0 differ in era and follow-up. Test that results agree between them first, then discover in one and check in the other.
7. **Treat graph-defined labels with care.** Under the landmark design, the TD/PIGD label is a function of year-0 items that are not in the analysed rows, so it is not a deterministic function of graph nodes. It is still strongly associated with them, which is why the definition-matched null remains necessary. Baseline component scores used as continuous moderators avoid choosing cut-points, as a supportive analysis.

## 5. Recommended framework: speed-adjusted subgroup causal discovery

The framework keeps v0.3's foundations: the landmark design, pre-specified phenotype rules, the expert map, the definition-matched null, equivalence margins, the simulation benchmark and the compute discipline. What changes is what is compared, and at which resolution.

**Step 1. One shared causal map, all participants.**
- Latent-aware discovery with tiers: joint causal inference with the phenotype and other baseline characteristics as context nodes, under the exogeneity assumption only (JCI-FCI; Mooij 2020; Analysis A in v0.3).
- When a test conditions on the phenotype, it is stratified by phenotype, so that a link present in one phenotype only is not averaged away.
- Output: every link any group might use, marks for possible hidden confounding, and the domains whose progression differs by phenotype.

**Step 2. Known phenotypes: speed, load or mechanism?**
- *Speed-only test, with a speed for each person.* Each patient's dynamics are the shared mechanism run at the patient's own speed. Log-speed depends on phenotype and other baseline characteristics, plus a random person-level part. Load and noise may differ by phenotype.
  - The test asks whether phenotype acts on speed alone or also on the links. It is a likelihood ratio with p² − 1 degrees of freedom for p domains, calibrated against the definition-matched null.
  - Speed must be modelled per person. Averaging dynamics over patients whose speeds differ breaks the matrix-power relation, and a test on group-averaged dynamics can then mistake speed for mechanism (§2).
  - The group-level matrix-power test serves only as a quick screen.
- *How much of the difference speed explains.* With about 1,000 patients, trivial deviations become significant. So the test is reported with the share of the phenotype difference in dynamics that speed explains: 1 − ‖A(PIGD) − ŝ·A(TD)‖² / ‖A(PIGD) − A(TD)‖², where A holds the continuous-time effects. This is reported alongside the cross-validated fit, and the expert map's zeros reduce the degrees of freedom.
- *Speed-adjusted joint discovery.* Both phenotype maps are estimated together. PIGD's effects equal TD's at PIGD's speed, plus sparse deviations that are penalised so a link deviates only when the data support it. The deviations are the candidate pathway differences of strength or structure, each with a speed-adjusted effect and a patient-bootstrap interval. The method adapts the data-shared lasso (Gross & Tibshirani 2016; Ollier & Viallon 2017) and joint Granger networks (Skripnikov & Michailidis 2019), shrinking towards a reference scaled by speed rather than towards equality.
- *Separate graphs* per phenotype (v0.3's Analysis B) stay as the transparent reference that makes few assumptions.
- *Phenotype axes and what each predicts.*
  - **TD versus PIGD.** PIGD is expected to be faster, with more cognitive and psychotic load (Aleksovski 2018). Whether mechanisms differ is open.
  - **RBD status as a proxy for body-first disease.** The SOC model predicts earlier non-motor load and faster speed with shared propagation, so for most links it predicts the class "similar". Mechanism differences would refine the model.
  - **Misclassification.** The RBD questionnaire misclassifies many patients, which pulls differences towards zero. A stricter definition is a sensitivity analysis.
- *The definition-matched null is extended.* Person-level speed and load in the simulator may depend on baseline state, so simulated phenotypes inherit speed and load differences without any mechanism difference. Only mechanism differences should then exceed the null.

**Step 3. Data-driven mechanism classes: are there hidden groups with different pathways?**
- *Model.* A mixture of causal transition models. Each patient belongs to one of K latent classes. Each class has its own yearly dynamics on the shared map's links (its own causal map, with class-specific sparsity), its own levels and its own noise. Membership may depend on baseline characteristics.
- *Two versions are fitted.* In *speed classes*, the class dynamics are powers of one shared matrix. In *mechanism classes*, they are free. The test asks whether the classes the data support differ only in speed and load.
- *Number of classes.* K runs from 2 to 4 and is chosen by cross-validated likelihood on held-out patients, with the bootstrap likelihood-ratio test as a check (Nylund 2007).
- *Null distribution.* The whole procedure, including the choice of K, is repeated on data simulated from the fitted speed-class model.
- *Within-class burden and speed.* Random intercepts and a random person-level speed sit inside the mixture. Without them, stable burden and speed differences within a class would look like extra classes or class differences in mechanism.
- *Skewed scores.* Classes must survive monotone transformations of skewed scores (Bauer & Curran 2003).
- *Links within classes.* Class-specific links are estimated by sample splitting: classes are found in one random half and links estimated in the other. This avoids comparing pathways between classes found in the same data.
- *If mechanism classes exist, report:*
  - their maps and routes to loss of independence;
  - their overlap with TD/PIGD and RBD (adjusted Rand index);
  - biology not used to define them: dopamine-transporter binding, genetics, and seed amplification if obtained later;
  - stability under resampling and replication by era;
  - whether membership can be predicted at baseline.

**Step 4. Individual variation, in PPMI 1.0, where people have enough visits.**
- *Speed and load.* Person-level speed and load are random effects. Speed is one parameter per person, informed by every link, so it is far better identified than separate person-level slopes.
- *Key links.* Random deviations on at most two pre-specified key links are exploratory. Variances of person-level lagged effects need about 50 or more occasions per person to be estimated well (Schultzberg & Muthén 2018), and PPMI 1.0 offers about eight.
- *Outputs:*
  - how much patients vary in speed;
  - whether speed variation alone fits as well as models in which key links also vary;
  - the share of speed variation explained by phenotype, RBD and age at onset.

**Step 5. Confirmatory links and clinical outputs.**
- *Gated link tests.* The roughly ten pre-specified links (v0.3's Analysis C) are tested as speed-adjusted differences, and only if Step 2's speed-only test is positive (gatekeeping).
- *Classes per link,* on the speed-adjusted difference: similar, stronger beyond speed, structurally different, or unresolved, with the equivalence margin Δ of v0.3.
- *A pathway card per phenotype and per class:* speed relative to the reference, load profile, the links that differ, and the routes to loss of independence.

**One engine for Steps 2 to 4.** Steps 2 to 4 share one model: a causal transition model on the shared map, with a speed and a load for each person. Phenotypes, latent classes and baseline characteristics can act on speed, on load or on the links. Each test asks the same question at a different resolution: do the links themselves change?

**Continuous-time check.** The relation B(fast) = B(slow)^s is the discrete form of a continuous-time model with a person-level time scale.
- A hierarchical continuous-time state-space model with measurement error (Driver & Voelkle 2018) is fitted to the final models as a sensitivity analysis. Measurement error, floors and ceilings can distort discrete-time comparisons.
- The ctsem package (version 3.11; Driver, Voelkle & Oud 2026) allows random effects on any drift parameter, predictors that moderate them, and drift entries written as expressions of extra parameters. A random person-level speed that multiplies every drift entry should therefore be specifiable, but no worked example exists, so it is validated by simulation first.
- The measurement model is held equal across groups (Schuurman & Hamaker 2019).

## 6. Hypotheses, tests and the simulation benchmark

| Test | Question | Statistic and null | Decision |
|---|---|---|---|
| H-M1 (primary, methodological, simulation) | At PPMI's sizes, do naive and link-wise comparisons misread speed and load differences as mechanism differences, while the speed-adjusted tests do not? | False mechanism-difference rate per truly shared link, under shared mechanisms with speed and load differences, definition artefacts and hidden burden | Naive above 5%; speed-adjusted at or below 5%, within two Monte Carlo standard errors |
| H-P (primary, PPMI) | Do the primary phenotypes differ in mechanism beyond speed and load? | Likelihood ratio in the random-speed model (phenotype acting on speed only, or on the links too), against the extended definition-matched null (R = 200); reported with the share of the difference explained by speed | Holm with H-L: the smaller p-value is tested at 2.5%, the other at 5% |
| H-L (primary, PPMI 1.0) | Do data-driven progression classes differ in mechanism beyond speed and load? | Likelihood ratio of speed classes against mechanism classes at the selected K, with a random speed within each class; parametric bootstrap from the fitted speed-class model, repeating the choice of K (R = 200) | Holm with H-P; descriptive only if the power gate fails |
| H-C1 to H-C10 (gated) | Which pre-specified links differ beyond speed? | Speed-adjusted differences; Benjamini–Hochberg at 5% | Tested only if H-P rejects |
| H-I (key secondary, PPMI 1.0) | How much do individuals vary in speed, and how much of that do phenotypes explain? Do key links vary beyond speed? | Speed variance and explained variance; random-slope variances on at most two links, exploratory | Estimated with intervals |
| H-V (key secondary) | Are mechanism classes reproducible and useful? | Stability, replication by era, biology not used in fitting, baseline predictability, blinded clinician review; prognostic value in Part II | Thresholds fixed before analysis |

**Simulation benchmark (ADEMP; Morris 2019).** Data-generating mechanisms are fitted to PPMI over the frozen expert map. They include hidden burden and missingness, and the real phenotype rules are applied to the simulated baselines.

Scenarios:
- S0: shared mechanism, equal speed;
- S1: shared mechanism, phenotypes differing in speed (time scale) and load;
- S2: one or two links stronger beyond speed;
- S3: a link present in one phenotype only;
- S4: a shared curved mechanism over different ranges;
- S5: two latent mechanism classes that cut across the phenotypes;
- S6: latent speed classes only;
- S7: continuous individual variation in speed.

Procedures compared:
- naive split;
- v0.3's link-wise tests;
- separate graphs against the definition-matched null;
- the speed-only test;
- speed-adjusted joint discovery;
- the mixture tests;
- the random-effects models.

Measures: false mechanism-difference rate, power, localisation, class recovery (adjusted Rand index), bias of speed and of speed-adjusted effects, and cost per fit.

**Power gate.** If the latent-class test has power below 0.5 for a pre-specified mechanism difference at PPMI's follow-up, H-L is reported as descriptive.

## 7. Novelty and clinical significance

### 7.1 What exists

Two novelty searches on 2 November 2026 covered Europe PMC, OpenAlex, Crossref, arXiv, OpenReview, PMLR and the web.
- The first logged about 100 queries.
- The second made about 530 database queries and 44 web searches, with "cited by" checks on about 35 key papers. Its verified findings are in `methods_and_novelty_update.md`.

Comparing causal or temporal networks between groups of patients is more common than the first search suggested. What is missing is a comparison that separates speed and load from mechanism.

**In PD:**
- *Directed models fitted by phenotype* exist, but all are confirmatory:
  - a two-wave multiple-group model of TD versus PIGD (Johnson 2016). Postural severity predicted later working memory in PIGD only. The model has no baseline working memory, and disease duration was shorter in PIGD;
  - cross-sectional path and mediation models per phenotype (Fereshtehnejad 2015b; Chien 2026);
  - MRI path models of propagation (Pyatigorskaya 2021). Brainstem-to-cortex fitted patients with RBD, cortex-to-brainstem those without. The design is cross-sectional and compares two pre-set directions;
  - directed brain connectivity by phenotype (Florin 2016; Zeng 2019; Shan 2025);
  - a causal structural-covariance analysis in TD patients over disease duration (Zhang 2025).
- *Directed longitudinal studies of clinical domains* model the whole cohort, or PD against controls:
  - random-intercept cross-lagged models of anxiety, depression and motor function in PPMI (Hodgson 2026);
  - within-person mediation of depression, apathy and cognition in PPMI (Szymkowicz 2022);
  - cognition and depression in DeNoPa (Schroeders 2022);
  - cognition preceding anxiety and depression in PPMI (Petkus 2019).
- *Causal discovery in PD* covers the whole cohort:
  - undirected PC-algorithm graphs per year (Pan 2024);
  - dynamic Bayesian networks of motor and cognitive outcomes (Hayete 2017);
  - penalised FCI on cross-sectional biomarkers (Pal 2026).
- *Symptom networks* are undirected, cross-sectional or based on one patient (Heimrich 2023; van der Velden 2018). A directed, cross-sectional Bayesian network compared multiple sclerosis with PD (Al-Sharman 2026).
- *Progression-subtype models* describe trajectories, not how domains drive each other:
  - multi-cohort subtypes on a latent disease time (Hähnel 2024);
  - disease course mapping in PPMI (Poulet & Durrleman 2023);
  - subtype and stage inference (Young 2018);
  - imaging subtypes with their own linear dynamics, which differ in both speed and trajectory (Zhou 2021).
- *A shared graph across diagnoses.* A preprint proposes that Alzheimer's disease, PD and healthy ageing share one causal graph and differ in initial conditions and progression rates (Saadawy 2026). It compares diagnoses, not patients within PD. Its abstract reports no test calibrated for speed, load or group definition.

**In other diseases, the closest work:**
- *ALS.* Dynamic Bayesian networks were fitted separately for slow, average and fast progressors (Leão 2021; Gromicho 2022). The groups are defined by the rate of decline, a variable inside the model, so speed and mechanism are confounded. This is the trap §2 describes.
- *Alzheimer's disease.*
  - Latent-process dynamic models with stage-specific temporal influences, compared by Wald tests (Taddé 2020; Carles 2021).
  - Continuous-time models fitted separately by stage (Markova 2026).
  - Cross-lagged networks in MCI converters versus non-converters, groups defined by speed (Romano 2026).
  - Data-driven causal models per stage with person-level rates, without tests of structure (Zheng 2022; Petrella 2024).
  - A Bayesian network with edges varying along a latent pseudotime, cross-sectional (Glazman 2025, preprint; method: Zhou 2023b).
- *Development.* Temporal causal discovery in a birth cohort included a formal test of whether boys' and girls' graphs differ (Larsen 2026, preprint). Boys and girls differ in developmental tempo, and speed is not separated.
- *Psychiatry.*
  - One cross-lagged network per trajectory class or remission group, with groups defined from the modelled symptoms (Ma 2023; Zandstra 2026).
  - Networks compared between randomised trial arms by permutation (Kenny 2026).
  - Person-specific and subgroup networks need dense series of measurements (Gates 2017; Lane 2019; Lim 2026). Apparent individual differences can be sampling error (Hoekstra 2023).
- *COVID-19.* Mixture-of-DAGs causal discovery allows latent confounding but folds the classes into one summary graph (Abdollahi 2026; method: Strobl 2023).
- *Molecular data.* Subtype-specific causal networks are routine (e.g. Yajima 2015; Ni 2018; Lovelace 2026). The claim is therefore limited to clinical progression.

**Related methods, none applied this way:**
- *Clustering patients by mechanism:* Huang 2019; Saeed 2020; Strobl 2023; Mameche 2025; latent-class VAR models (Ernst 2024; Ernst & Haslbeck 2025).
- *Time-scale-only hypotheses:*
  - formally tested for survival (Stroustrup 2016), trait evolution (Clavel 2015), growth curves (Cole 2010) and univariate progression (Raket 2022);
  - a network diffusion model multiplies fixed connectome dynamics by a subject-level rate (Raj 2015);
  - rate-agnostic causal discovery allows unknown sampling rates without testing a shared one (Solovyeva 2023; Abavisani 2026, preprint).
- *Groups defined from network variables:*
  - when this biases a comparison depends on the question asked (Haslbeck 2021);
  - a correction exists for Ising models (Boot 2025);
  - endogenous-context theory treats such groups inside discovery (Günther 2024);
  - a simulated null with homogeneous dynamics was used within severity strata (Hoekstra 2026);
  - floor effects can create apparent differences (Haqiqatkhah 2024);
  - the critiques remain (Terluin 2016; de Ron 2021).
  None re-applies a definition rule inside a cohort-fitted null.

### 7.2 What this study would add

1. The first test within a disease of whether patients share causal progression pathways once groups and persons may differ in speed and load, against a null calibrated for those differences. In PD, it is also the first use of causal discovery on longitudinal clinical data to compare phenotypes. It works at three resolutions: clinical phenotypes, data-driven classes and individuals.
2. A decomposition that adds time scale (speed) to load and mechanism inside causal discovery. It shows that comparing yearly effects reads faster progression as a different mechanism. That affects published designs: ALS speed strata, Alzheimer's stage comparisons and psychiatric trajectory-class networks. It also gives a test of whether groups or persons share one multivariate causal mechanism up to a time-scale factor; we found no such test.
3. A cohort-fitted null that re-applies the phenotype-definition rule to data with one shared mechanism and person-level speed and load, so that only mechanism differences exceed it.
4. If they exist, mechanism-based PD subtypes: groups defined by how clinical domains drive each other. They are compared with clinical phenotypes and with the pace subtypes that data-driven work keeps finding.

**Limits of the search.** Most entries were read at the abstract level.
- *Full texts to read before any claim of priority:* Larsen 2026, for its test; Saadawy 2026, for how invariance was tested; Rocchi 2026, for how its groups are defined.
- *Unfinished checks:* the "cited by" checks on J-PCMCI+ (Günther 2023), MCD (Varambally 2024), Saeed 2020 and Strobl 2023 were not completed because free API quotas ran out. The WP1 search repeats them with an API key.

### 7.3 Clinical significance

Each possible answer changes care differently.

- **Speed only.** One progression map serves every patient.
  - Care is personalised by review interval and prognosis, using each patient's estimated speed.
  - Trials can enrich for fast progressors. In simulation this cut the required sample size by 43% (Hähnel 2024). In an exploratory analysis, prasinezumab slowed motor progression more in rapidly progressing patients (Pagano 2024).
- **Load differences.** The same propagation from different starting points.
  - This supports domain-specific screening from the first visit, as Fereshtehnejad 2015a recommends for mild cognitive impairment, orthostatic hypotension and RBD.
- **Mechanism differences.** Monitoring targets and referrals should differ by phenotype or class.
  - An axial route to dependence argues for earlier physiotherapy and falls prevention (NICE NG71; Fox 2018; Fasano 2017).
  - An autonomic-to-cognitive route argues for earlier blood-pressure checks, formal cognitive testing and counselling about dementia risk (Gibbons 2017; Litvan 2012; Seppi 2019).
  - Trials stratified by mechanism would need endpoints specific to each stratum.
- **Phenotype adequacy.** The study measures how much of the real variation in pathways the clinical labels capture.
  - If TD/PIGD differences are speed only, the label adds little beyond an estimate of speed.
  - If RBD differences are load and speed only, that supports the SOC model's shared spread.
  - If mechanisms differ, phenotype-specific care pathways are justified.

Half of PPMI participants reach a disability milestone within five years, most often cognitive, dependence or autonomic (Brumm 2023). The routes to those milestones, phenotype by phenotype and class by class, are the study's main clinical output.

## 8. Feasibility and risks

**Which data carry which analysis.** In the pilot build, PPMI 1.0 had 431 de novo participants with about 9 yearly rows each, and PPMI 2.0 had 957 with about 3 (protocol, Appendix B).
- Phenotype-level tests use both cohorts.
- Latent classes and individual variation rest mainly on PPMI 1.0, with PPMI 2.0 replicating the class-level results.

**Can hidden mechanism classes be found with this follow-up?** A third synthetic check:
- *Classes.* Two latent classes of 40% and 60% of patients. The minority runs the shared mechanism at twice the speed. In the mechanism scenario, it also has the extra axial → independence link.
- *Model.* Two-class mixtures of transition models, fitted by EM with membership per patient.
- *Test.* The speed-class version, in which class dynamics are powers of one matrix, is tested against the free version. The reference distribution is chi-square with 35 degrees of freedom, with no bootstrap.
- *Replicates.* 200 cohorts per cell.

| Follow-up | Truth | Mechanism test rejects (Monte Carlo SE) | Class recovery (median adjusted Rand index) |
|---|---|---|---|
| 430 patients × 7 transitions (like PPMI 1.0) | Speed classes only | 6.0% (1.7) | 0.76 |
| 430 × 7 | Mechanism classes | 69% (3.3) | 0.82 |
| 950 patients × 2 transitions (like PPMI 2.0) | Speed classes only | 12.0% (2.3) | 0.34 |
| 950 × 2 | Mechanism classes | 21% (2.9) | 0.47 |

- **About seven transitions per person.** Classes are recovered well. Speed classes are told apart from mechanism classes at the nominal false-positive rate, with moderate power for a single extra link.
- **Two transitions per person.** Classes are poorly recovered, the unadjusted test is too liberal (12% at a nominal 5%), and power is low.
- **Consequences.** The latent-class analysis rests on PPMI 1.0, its test is calibrated by bootstrap (§6), and PPMI 2.0 serves only as a limited check.
- **What the real data add.** In the real data the number of classes is unknown and there are more domains. The WP3 benchmark repeats this check with the PPMI-fitted simulator, and the power gate decides whether H-L stays a primary test.

| Assumption | Risk | Check |
|---|---|---|
| Dynamics roughly linear on Gaussian-copula scales | Curvature over different ranges mimics strength differences | Splines in the confirmatory models; scenario S4; the null contains the fitted curvature |
| One speed scales all disease dynamics | Some domains may speed up more than others | That is a pathway difference by definition; the deviation map shows where. A shared-modes model (same directions of change, different rates per mode) is an intermediate reading |
| Speed varies between patients within a group | Group-averaged dynamics break the matrix-power relation | Speed for each person, as a random effect (§2) |
| The modelled state is first-order Markov | A slow process left out of the model breaks the power relation | Check terms at a two-year lag |
| Initial state is linked to speed | Fast progressors are more advanced at entry | Model the correlation between initial state and speed (Gistelinck 2021) |
| Treatment effects are not time-scaled | Dopaminergic treatment improves motor scores on its own timetable (Holford 2006; Vu 2012) | Treatment enters as an input with its own effects, not scaled by speed; transitions with a treatment start are excluded in a sensitivity analysis |
| Floor and ceiling effects | Bounded scales (Schwab & England, MoCA) flatten effects near their limits | Copula scales; continuous-time state-space sensitivity analysis |
| Measurement error | Attenuated effects that differ by range | State-space sensitivity analysis; measurement error in the simulator |
| Hidden person-level burden | Spurious links and load differences | Random intercepts; burden in every null |
| Phenotypes computed from graph variables | Artefacts of the split | Landmark design; extended definition-matched null |

**Compute (indicative; measured in WP3 before any number is fixed).**
- The main new cost is the random-speed model, because each patient's speed is integrated out numerically.
- The table assumes about 5 minutes per fit at ten domains. That needs automatic differentiation (Stan or JAX); the numerical gradients used in the synthetic checks would be far slower.

| Component | Size | Hours, 4 parallel jobs |
|---|---|---|
| Random-speed phenotype tests on the real data, with the extended definition-matched null | 201 datasets × 2 models × 2 phenotype axes | About 35 |
| Speed-adjusted joint discovery on the real data, with nulls | Regression fits | Under 2 |
| Random-speed tests in the benchmark | S0 to S4 × 100 datasets × 2 models | About 20 |
| Latent classes with a random speed within each class, real data: bootstrap null repeating the choice of K | 200 datasets × 3 values of K × 2 versions | About 15 |
| Latent-class test in the benchmark, reduced nested design | S0, S5, S6 × 5 datasets × 30 bootstrap replicates | About 30 |
| Individual speed and load (Bayesian, PPMI 1.0); continuous-time sensitivity analysis | A few final fits | About 24 |
| v0.3 components kept: Analysis A, separate graphs as the reference, stability | As in v0.3, but without the nested evaluation of separate graphs | About 70 |
| Total | | About 195 (8 days), within v0.3's 10-day cap |

## 9. What would change in the protocol (v0.3 to v0.4)

- **Questions.**
  - Q1 becomes "Does every patient progress through the same pathway?", answered for phenotypes, data-driven classes and individuals.
  - Q2 (speed or pathway) uses the time-scale definition of speed and adds load.
  - Q3 (routes to loss of independence) is reported per phenotype and per class.
  - Q4 (fit of the expert map) is unchanged.
- **Estimands (§2.3).** Speed (time scale), load and the speed-adjusted strength difference are added. The five link classes apply to the speed-adjusted difference.
- **Phenotypes (§5).** RBD status (body-first proxy) is proposed as a second primary axis, with the SOC model's prediction. TD/PIGD stays. Continuous baseline tremor and gait scores enter as a supportive analysis.
- **Analyses (§7).**
  - A keeps JCI-FCI, with tests stratified by phenotype.
  - B becomes three parts: the speed-only test in a model with a speed for each person, speed-adjusted joint discovery, and separate graphs as the reference. The definition-matched null gains speed and load.
  - D (latent mechanism classes, PPMI 1.0) and E (individual speed and load) are new.
  - C is gated and speed-adjusted, and is estimated in random-intercept models.
  - The speed-or-pathway rule of §7.5 is replaced by the decomposition.
- **Hypotheses (§2.4).**
  - H-M1 is extended to speed.
  - H-P and H-L are primary, with Holm.
  - H-C is gated; H-I and H-V are key secondary.
  - H-S is retired, because H-P and H-L answer it.
- **Simulation (§8).** S1 is redefined as time scale plus load, and S5 to S7 are added. The procedures, nested designs and power gate are as in §6 above.
- **Compute (§8.8).** Re-estimated at about 8 days.
- **Part II.**
  - The shared map with each patient's estimated speed becomes the comparator that phenotype- and class-specific maps must beat for prognosis.
  - External replication (II-a) checks the speed and mechanism verdicts.
- **Expert questionnaire, round 2.** It adds:
  - the RBD axis;
  - the choice of the two key links;
  - a forecast: for each phenotype, will differences be speed, load or mechanism? The answers are recorded before analysis and compared with the results afterwards.
- **Unchanged.** Population, landmark design, domains, expert map, equivalence margins, governance, reporting standards and the gate to Part II.

## 10. Decisions for the investigator and the panel

| Decision | Recommendation | Who |
|---|---|---|
| Adopt speed-adjusted subgroup causal discovery as the core of experiment 1 (protocol v0.4) | Yes. v0.3's comparison of yearly effects confuses speed with mechanism | Investigator and supervisors |
| Phenotype axes | TD/PIGD and RBD status as co-primary axes, with Holm control | Clinical panel, round 2 |
| Status of the latent-class test (H-L) | Primary, in PPMI 1.0, with a power gate that turns it descriptive if WP3 shows power below 0.5 | Investigator and statistician |
| The two key links for individual-level analysis | Chosen by the panel from the confirmatory list | Clinical panel, round 2 |
| Clinicians' forecasts of speed, load or mechanism | Record them before analysis | Investigator |
| WP0 aggregate audit: follow-up per participant, phenotype and RBD frequencies, visit intervals | Approve, because it fixes the numbers this design depends on | Investigator |

## References

> Every entry below was checked on 2 November 2026 against at least one of Crossref, Europe PMC, OpenAlex, arXiv, PMLR, JMLR, NeurIPS or the publisher's page. The exception is NICE NG71, whose website refused automated access; its wording was checked through a secondary source. Most entries were read at the abstract level.
- Abavisani M, Solovyeva K, Danks D, Pearlson G, Calhoun V, Plis S. Recovering directional brain networks under temporal undersampling: an application to schizophrenia. bioRxiv 2026. doi:10.64898/2026.08.09.743708
- Abdollahi K, Rezaei Ghahroodi Z, Mansourian M. Causal discovery and inference in COVID-19: a multiscale longitudinal analysis using mixtures of DAGs. PLoS One 2026;21(10):e0355553. doi:10.1371/journal.pone.0355553
- Al-Sharman A, Khalil H, Malouche D, et al. Directional symptom dependencies in multiple sclerosis and Parkinson's disease: a comparative Bayesian network analysis. J Multidiscip Healthc 2026;19:1–19. doi:10.2147/JMDH.S607760
- Aleksovski D, Miljkovic D, Bravi D, Antonini A. Disease progression in Parkinson subtypes: the PPMI dataset. Neurol Sci 2018;39(11):1971–1976. doi:10.1007/s10072-018-3522-z
- Athey S, Tibshirani J, Wager S. Generalized random forests. Ann Stat 2019;47(2):1148–1178. doi:10.1214/18-AOS1709
- Banwinkler M, Dzialas V, Hoenig MC, van Eimeren T; PPMI. Gray matter volume loss in proposed brain-first and body-first Parkinson's disease subtypes. Mov Disord 2022;37(10):2066–2074. doi:10.1002/mds.29172
- Bauer DJ, Curran PJ. Distributional assumptions of growth mixture models: implications for overextraction of latent trajectory classes. Psychol Methods 2003;8(3):338–363. doi:10.1037/1082-989X.8.3.338
- Belyaeva A, Squires C, Uhler C. DCI: learning causal differences between gene regulatory networks. Bioinformatics 2021;37(18):3067–3069. doi:10.1093/bioinformatics/btab167
- Birkenbihl C, Ahmad A, Massat NJ, et al. Artificial intelligence-based clustering and characterization of Parkinson's disease trajectories. Sci Rep 2023;13:2897. doi:10.1038/s41598-023-30038-8
- Boot J, de Ron J, Haslbeck JMB, Epskamp S. Correcting for selection bias after conditioning on a sum score in the Ising model. Behav Res Methods 2025;57(12). doi:10.3758/s13428-025-02820-1
- Borghammer P, Van Den Berge N. Brain-first versus gut-first Parkinson's disease: a hypothesis. J Parkinsons Dis 2019;9(s2):S281–S295. doi:10.3233/JPD-191721
- Borghammer P. The α-synuclein origin and connectome model (SOC model) of Parkinson's disease: explaining motor asymmetry, non-motor phenotypes, and cognitive decline. J Parkinsons Dis 2021;11(2):455–474. doi:10.3233/JPD-202481
- Brandmaier AM, Prindle JJ, McArdle JJ, Lindenberger U. Theory-guided exploration with structural equation model forests. Psychol Methods 2016;21(4):566–582. doi:10.1037/met0000090
- Brandmaier AM, von Oertzen T, McArdle JJ, Lindenberger U. Structural equation model trees. Psychol Methods 2013;18(1):71–86. doi:10.1037/a0030001
- Brooker SM, Pasquini J, Choi SH, et al. Clinical and imaging characteristics of Parkinson's disease with negative alpha-synuclein seed amplification assay. Mov Disord 2026;41(5):1114–1127. doi:10.1002/mds.70197
- Brumm MC, Siderowf A, Simuni T, et al. Parkinson's Progression Markers Initiative: a milestone-based strategy to monitor Parkinson's disease progression. J Parkinsons Dis 2023;13(6):899–916. doi:10.3233/JPD-223433
- Carles S, Taddé BO, Berr C, et al. Dynamic reciprocal relationships between cognitive and functional declines along the Alzheimer's disease continuum in the prospective COGICARE study. Alzheimers Res Ther 2021;13(1):148. doi:10.1186/s13195-021-00887-4
- Castelletti F, La Rocca L, Peluso S, Stingo FC, Consonni G. Bayesian learning of multiple directed networks from observational data. Stat Med 2020;39(30):4745–4766. doi:10.1002/sim.8751
- Chahine LM, Xie SX, Simuni T, et al. Longitudinal changes in cognition in early Parkinson's disease patients with REM sleep behavior disorder. Parkinsonism Relat Disord 2016;27:102–106. doi:10.1016/j.parkreldis.2016.03.006
- Chen T, Bello K, Aragam B, Ravikumar P. iSCAN: identifying causal mechanism shifts among nonlinear additive noise models. Adv Neural Inf Process Syst 2023. arXiv:2306.17361
- Chien CY, Tan CH, Hong WP, Chen KC, Yu RL. The role of motor subtypes in modulating neuropsychiatric and cognitive function in Parkinson's disease: a group-stratified mediation analysis. J Psychiatr Res 2026;192:352–359. doi:10.1016/j.jpsychires.2025.10.069
- Clavel J, Escarguel G, Merceron G. mvMORPH: an R package for fitting multivariate evolutionary models to morphometric data. Methods Ecol Evol 2015;6(11):1311–1319. doi:10.1111/2041-210X.12420
- Cole TJ, Donaldson MDC, Ben-Shlomo Y. SITAR—a useful instrument for growth curve analysis. Int J Epidemiol 2010;39(6):1558–1566. doi:10.1093/ije/dyq115
- Cui X, Lu X, Du S, Yu H. Temporal sequence of cognitive function and ADLs and mediation effect of apathy in Parkinson's disease: cross-lagged analyses. Clin Gerontol 2026;49(1):120–132. doi:10.1080/07317115.2024.2426182
- Dadu A, Satone V, Kaur R, et al. Identification and prediction of Parkinson's disease subtypes and progression using machine learning in two cohorts. NPJ Parkinsons Dis 2022;8:172. doi:10.1038/s41531-022-00439-z
- de Ron J, Fried EI, Epskamp S. Psychological networks in clinical populations: investigating the consequences of Berkson's bias. Psychol Med 2021;51(1):168–176. doi:10.1017/S0033291719003209
- Di Folco C, Couronné R, Arnulf I, et al. Charting disease trajectories from isolated REM sleep behavior disorder to Parkinson's disease. Mov Disord 2024;39(1):64–75. doi:10.1002/mds.29662
- Dondelinger F, Mukherjee S. The joint lasso: high-dimensional regression for group structured data. Biostatistics 2020;21(2):219–235. doi:10.1093/biostatistics/kxy035
- Dormann C, Guthier C, Cortina JM. Introducing continuous time meta-analysis (CoTiMA). Organ Res Methods 2020;23:620–650. doi:10.1177/1094428119847277
- Driver C, Voelkle M, Oud H. ctsem: continuous time structural equation modelling. R package version 3.11.1; 2026. doi:10.32614/CRAN.package.ctsem
- Driver CC, Voelkle MC. Hierarchical Bayesian continuous time dynamic modeling. Psychol Methods 2018;23(4):774–799. doi:10.1037/met0000168
- Driver CC. Inference with cross-lagged effects: problems in time. Psychol Methods 2025;30:174–202. doi:10.1037/met0000665
- Epskamp S, Waldorp LJ, Mõttus R, Borsboom D. The Gaussian graphical model in cross-sectional and time-series data. Multivariate Behav Res 2018;53(4):453–480. doi:10.1080/00273171.2018.1454823
- Ernst AF, Albers CJ, Jeronimus BF, Timmerman ME. Inter-individual differences in multivariate time-series: latent class vector-autoregressive modeling. Eur J Psychol Assess 2020;36(3):482–491. doi:10.1027/1015-5759/a000578
- Ernst AF, Haslbeck JMB. Modeling qualitative between-person heterogeneity in time series using latent class vector autoregressive models. Behav Res Methods 2026;58(1):28 (online December 2025). doi:10.3758/s13428-025-02909-7
- Ernst AF, Timmerman ME, Ji F, Jeronimus BF, Albers CJ. Mixture multilevel vector-autoregressive modeling. Psychol Methods 2024;29(1):137–154. doi:10.1037/met0000551
- Erro R, Picillo M, Scannapieco S, et al. The role of disease duration and severity on novel clinical subtypes of Parkinson disease. Parkinsonism Relat Disord 2020;73:31–34. doi:10.1016/j.parkreldis.2020.03.013
- Fasano A, Canning CG, Hausdorff JM, Lord S, Rochester L. Falls in Parkinson's disease: a complex and evolving picture. Mov Disord 2017;32(11):1524–1536. doi:10.1002/mds.27195
- Fereshtehnejad SM, Romenets SR, Anang JB, et al. New clinical subtypes of Parkinson disease and their longitudinal progression: a prospective cohort comparison with other phenotypes. JAMA Neurol 2015;72(8):863–873. doi:10.1001/jamaneurol.2015.0703 (cited as Fereshtehnejad 2015a)
- Fereshtehnejad SM, Shafieesabet M, Farhadi F, et al. Heterogeneous determinants of quality of life in different phenotypes of Parkinson's disease. PLoS One 2015;10(9):e0137081. doi:10.1371/journal.pone.0137081 (cited as Fereshtehnejad 2015b)
- Fereshtehnejad SM, Zeighami Y, Dagher A, Postuma RB. Clinical criteria for subtyping Parkinson's disease: biomarkers and longitudinal progression. Brain 2017;140(7):1959–1976. doi:10.1093/brain/awx118
- Fisher AJ, Medaglia JD, Jeronimus BF. Lack of group-to-individual generalizability is a threat to human subjects research. Proc Natl Acad Sci USA 2018;115(27):E6106–E6115. doi:10.1073/pnas.1711978115
- Florin E, Pfeifer J, Visser-Vandewalle V, Schnitzler A, Timmermann L. Parkinson subtype-specific Granger-causal coupling and coherence frequency in the subthalamic area. Neuroscience 2016;332:170–180. doi:10.1016/j.neuroscience.2016.06.052
- Fox SH, Katzenschlager R, Lim SY, et al. International Parkinson and Movement Disorder Society evidence-based medicine review: update on treatments for the motor symptoms of Parkinson's disease. Mov Disord 2018;33(8):1248–1266. doi:10.1002/mds.27372
- Gao LL, Bien J, Witten D. Selective inference for hierarchical clustering. J Am Stat Assoc 2024;119(545):332–342. doi:10.1080/01621459.2022.2116331
- Gates KM, Lane ST, Varangis E, Giovanello K, Guskiewicz K. Unsupervised classification during time-series model building. Multivariate Behav Res 2017;52(2):129–148. doi:10.1080/00273171.2016.1256187
- Gates KM, Molenaar PCM. Group search algorithm recovers effective connectivity maps for individuals in homogeneous and heterogeneous samples. NeuroImage 2012;63(1):310–319. doi:10.1016/j.neuroimage.2012.06.026
- Gibbons CH, Schmidt P, Biaggioni I, et al. The recommendations of a consensus panel for the screening, diagnosis, and treatment of neurogenic orthostatic hypotension and associated supine hypertension. J Neurol 2017;264(8):1567–1582. doi:10.1007/s00415-016-8375-x
- Gistelinck F, Loeys T, Flamant N. Multilevel autoregressive models when the number of time points is small. Struct Equ Modeling 2021;28:15–27. doi:10.1080/10705511.2020.1753517
- Glazman N, Mangal J, Borges P, Ourselin S, Cardoso MJ. Dynamic causal discovery in Alzheimer's disease through latent pseudotime modelling. arXiv:2511.04619 (preprint, 2025)
- Gong M, Zhang K, Schölkopf B, Tao D, Geiger P. Discovering temporal causal relations from subsampled data. Proc ICML 2015, PMLR 37:1898–1906
- Gromicho M, Leão T, Oliveira Santos M, et al. Dynamic Bayesian networks for stratification of disease progression in amyotrophic lateral sclerosis. Eur J Neurol 2022;29(8):2201–2210. doi:10.1111/ene.15357
- Gross SM, Tibshirani R. Data shared lasso: a novel tool to discover uplift. Comput Stat Data Anal 2016;101:226–235. doi:10.1016/j.csda.2016.02.015
- Grün B, Leisch F. FlexMix version 2: finite mixtures with concomitant variables and varying and constant parameters. J Stat Softw 2008;28(4). doi:10.18637/jss.v028.i04
- Günther W, Ninad U, Runge J. Causal discovery for time series from multiple datasets with latent contexts. Proc UAI 2023, PMLR 216:766–776. arXiv:2306.12896
- Günther W, Popescu O-I, Rabel M, Ninad U, Gerhardus A, Runge J. Causal discovery with endogenous context variables. Adv Neural Inf Process Syst 2024;37. arXiv:2412.04981
- Hähnel T, Raschka T, Sapienza S, et al. Progression subtypes in Parkinson's disease identified by a data-driven multi cohort analysis. NPJ Parkinsons Dis 2024;10:95. doi:10.1038/s41531-024-00712-3
- Halsband C, Zapf A, Sixel-Döring F, Trenkwalder C, Mollenhauer B. The REM sleep behavior disorder screening questionnaire is not valid in de novo Parkinson's disease. Mov Disord Clin Pract 2018;5(2):171–176. doi:10.1002/mdc3.12591
- Hamaker EL, Grasman RPPP. To center or not to center? Investigating inertia with a multilevel autoregressive model. Front Psychol 2015;5:1492. doi:10.3389/fpsyg.2014.01492
- Hamaker EL, Kuiper RM, Grasman RPPP. A critique of the cross-lagged panel model. Psychol Methods 2015;20(1):102–116. doi:10.1037/a0038889
- Haqiqatkhah MM, Ryan O, Hamaker EL. Skewness and staging: does the floor effect induce bias in multilevel AR(1) models? Multivariate Behav Res 2024;59(2):289–319. doi:10.1080/00273171.2023.2254769
- Haslbeck JMB, Borsboom D, Waldorp LJ. Moderated network models. Multivariate Behav Res 2021;56(2):256–287. doi:10.1080/00273171.2019.1677207
- Haslbeck JMB, Ryan O, Dablander F. The sum of all fears: comparing networks based on symptom sum-scores. Psychol Methods 2021 (advance online). doi:10.1037/met0000418
- Hastie T, Tibshirani R. Varying-coefficient models. J R Stat Soc B 1993;55(4):757–779. doi:10.1111/j.2517-6161.1993.tb01939.x
- Hayete B, Wuest D, Laramie J, et al. A Bayesian mathematical model of motor and cognitive outcomes in Parkinson's disease. PLoS One 2017;12(6):e0178982. doi:10.1371/journal.pone.0178982
- Heimrich KG, Schönenberg A, Mühlhammer HM, Mendorf S, Santos-García D, Prell T. Longitudinal analysis of the Non-Motor Symptoms Scale in Parkinson's Disease (NMSS): an exploratory network analysis approach. Front Neurol 2023;14:972210. doi:10.3389/fneur.2023.972210
- Hodgson P, Jordan A, Sinani C, et al. Longitudinal dynamics of physical function with anxiety and depression in Parkinson's disease: a cross-lagged panel analysis of the PPMI dataset. Brain Behav 2026;16(2):e71257. doi:10.1002/brb3.71257
- Hoekstra RHA, Epskamp S, Borsboom D. Heterogeneity in individual network analysis: reality or illusion? Multivariate Behav Res 2023;58(4):762–786. doi:10.1080/00273171.2022.2128020
- Hoekstra RHA, Epskamp S, Nierenberg AA, Borsboom D, McNally RJ. Testing similarity in longitudinal networks: the Individual Network Invariance Test. Psychol Methods 2026;31(5):844–870. doi:10.1037/met0000638
- Holford NHG, Chan PLS, Nutt JG, Kieburtz K, et al. Disease progression and pharmacodynamics in Parkinson disease: evidence for functional protection with levodopa and other treatments. J Pharmacokinet Pharmacodyn 2006;33:281–311. doi:10.1007/s10928-006-9012-6
- Horsager J, Andersen KB, Knudsen K, et al. Brain-first versus body-first Parkinson's disease: a multimodal imaging case-control study. Brain 2020;143(10):3077–3088. doi:10.1093/brain/awaa238
- Huang B, Zhang K, Xie P, Gong M, Xing EP, Glymour C. Specific and shared causal relation modeling and mechanism-based clustering. Adv Neural Inf Process Syst 2019;32:13510–13521
- Huang B, Zhang K, Zhang J, Ramsey J, Sanchez-Romero R, Glymour C, Schölkopf B. Causal discovery from heterogeneous/nonstationary data. J Mach Learn Res 2020;21(89):1–53
- Hyttinen A, Plis S, Järvisalo M, Eberhardt F, Danks D. A constraint optimization approach to causal discovery from subsampled time series data. Int J Approx Reason 2017;90:208–225. doi:10.1016/j.ijar.2017.07.009
- Jedidi K, Jagpal HS, DeSarbo WS. Finite-mixture structural equation models for response-based segmentation and unobserved heterogeneity. Marketing Sci 1997;16(1):39–59. doi:10.1287/mksc.16.1.39
- Jin R, Yoshioka H, Sato H, Hisaka A. Data-driven disease progression model of Parkinson's disease and effect of sex and genetic variants. CPT Pharmacometrics Syst Pharmacol 2024;13:649–659. doi:10.1002/psp4.13112
- Johnson AR, Bucks RS, Kane RT, Thomas MG, Gasson N, Loftus AM. Motor subtype as a predictor of future working memory performance in idiopathic Parkinson's disease. PLoS One 2016;11(3):e0152534. doi:10.1371/journal.pone.0152534
- Jones JD, Kurniadi NE, Kuhn TP, et al. Depressive symptoms precede cognitive impairment in de novo Parkinson's disease patients: analysis of the PPMI cohort. Neuropsychology 2019;33(8):1111–1120. doi:10.1037/neu0000583
- Kenny EA, Brand BA, Ebrahimi OV, McCutcheon RA. Dopamine antagonists and dynamic symptom networks in schizophrenia. Br J Psychiatry 2026. doi:10.1192/bjp.2026.10775
- Knudsen K, Fedorova TD, Horsager J, et al. Asymmetric dopaminergic dysfunction in brain-first versus body-first Parkinson's disease subtypes. J Parkinsons Dis 2021;11(4):1677–1687. doi:10.3233/JPD-212761
- Kriegeskorte N, Simmons WK, Bellgowan PSF, Baker CI. Circular analysis in systems neuroscience: the dangers of double dipping. Nat Neurosci 2009;12(5):535–540. doi:10.1038/nn.2303
- Krishnagopal S, von Coelln R, Shulman LM, Girvan M. Identifying and predicting Parkinson's disease subtypes through trajectory clustering via bipartite networks. PLoS One 2020;15(6):e0233296. doi:10.1371/journal.pone.0233296
- Kuiper RM, Ryan O. Drawing conclusions from cross-lagged relationships: re-considering the role of the time-interval. Struct Equ Modeling 2018;25:809–823. doi:10.1080/10705511.2018.1431046
- Kuiper RM, Ryan O. Meta-analysis of lagged regression models: a continuous-time approach. Struct Equ Modeling 2020;27:396–413. doi:10.1080/10705511.2019.1652613
- Lane ST, Gates KM, Pike HK, Beltz AM, Wright AGC. Uncovering general, shared, and unique temporal patterns in ambulatory assessment data. Psychol Methods 2019;24(1):54–69. doi:10.1037/met0000192
- Larsen TE, Hernández-Lorca M, Ekstrøm CT, Vinding R, Bønnelykke K, Strandberg-Larsen K, Petersen AH. Mapping the dynamic interplay of mental health and weight across childhood: data-driven explorations using causal discovery. medRxiv 2026. doi:10.64898/2026.04.16.26350943
- Lawton M, Ben-Shlomo Y, May MT, et al. Developing and validating Parkinson's disease subtypes and their motor and cognitive progression. J Neurol Neurosurg Psychiatry 2018;89(12):1279–1287. doi:10.1136/jnnp-2018-318337
- Leão T, Madeira SC, Gromicho M, de Carvalho M, Carvalho AM. Learning dynamic Bayesian networks from time-dependent and time-independent data: unraveling disease progression in amyotrophic lateral sclerosis. J Biomed Inform 2021;117:103730. doi:10.1016/j.jbi.2021.103730
- Lee JW, Song YS, Kim H, Ku BD, Lee WW. Alteration of tremor dominant and postural instability gait difficulty subtypes during the progression of Parkinson's disease: analysis of the PPMI cohort. Front Neurol 2019;10:471. doi:10.3389/fneur.2019.00471
- Leisch F. FlexMix: a general framework for finite mixture models and latent class regression in R. J Stat Softw 2004;11(8). doi:10.18637/jss.v011.i08
- Lim KO, Kummerfeld E, Anderson LM, et al. Personal causal modeling of affect and eating behaviors in bulimia nervosa: implications for personalized treatment. J Eat Disord 2026;14(1):52. doi:10.1186/s40337-026-01524-x
- Litvan I, Goldman JG, Tröster AI, et al. Diagnostic criteria for mild cognitive impairment in Parkinson's disease: Movement Disorder Society Task Force guidelines. Mov Disord 2012;27(3):349–356. doi:10.1002/mds.24893
- Lövdal SS, Carli G, Orso B, et al. Investigating the aspect of asymmetry in brain-first versus body-first Parkinson's disease. NPJ Parkinsons Dis 2024;10:74. doi:10.1038/s41531-024-00685-3
- Lovelace TC, Benos PV. New causal discovery algorithm over censored variables identifies subtype-specific drivers of breast cancer progression. GigaScience 2026;15:giag060. doi:10.1093/gigascience/giag060
- Ma Z, Zhao J, Chen H, et al. Temporal network of depressive symptoms across college students with distinct depressive trajectories during the COVID-19 pandemic. Depress Anxiety 2023;2023:8469620. doi:10.1155/2023/8469620
- Mameche S, Kalofolias J, Vreeken J. Causal mixture models: characterization and discovery. Adv Neural Inf Process Syst 2025:50700–50729. doi:10.52202/085713-1511
- Marek K, Chowdhury S, Siderowf A, et al. The Parkinson's Progression Markers Initiative (PPMI): establishing a PD biomarker cohort. Ann Clin Transl Neurol 2018;5(12):1460–1477. doi:10.1002/acn3.644
- Markova TZ, Heston MB, Berry AS, et al. Tracking the continuous time dynamics of tau and depressive symptoms across the Alzheimer's disease continuum. J Neurosci 2026;46(36):e2309252026. doi:10.1523/JNEUROSCI.2309-25.2026
- Mastenbroek SE, Vogel JW, Collij LE, et al. Disease progression modelling reveals heterogeneity in trajectories of Lewy-type α-synuclein pathology. Nat Commun 2024;15:5133. doi:10.1038/s41467-024-49402-x
- Mastrangelo A, Wurster I, Ticca A, et al. Clinical correlates of a negative cerebrospinal fluid α-synuclein seed amplification assay result in Parkinson's disease. NPJ Parkinsons Dis 2026;12:97. doi:10.1038/s41531-026-01346-3
- Mooij JM, Magliacane S, Claassen T. Joint causal inference from multiple contexts. J Mach Learn Res 2020;21(99):1–108
- Morris TP, White IR, Crowther MJ. Using simulation studies to evaluate statistical methods. Stat Med 2019;38(11):2074–2102. doi:10.1002/sim.8086
- Mulder JD, Hamaker EL. Three extensions of the random intercept cross-lagged panel model. Struct Equ Modeling 2021;28(4):638–648. doi:10.1080/10705511.2020.1784738
- Myers PS, Jackson JJ, Clover AK, et al. Distinct progression patterns across Parkinson disease clinical subtypes. Ann Clin Transl Neurol 2021;8(8):1695–1708. doi:10.1002/acn3.51436
- National Institute for Health and Care Excellence. Parkinson's disease in adults (NICE guideline NG71). London: NICE; 2017. www.nice.org.uk/guidance/ng71
- Ni Y, Müller P, Zhu Y, Ji Y. Heterogeneous reciprocal graphical models. Biometrics 2018;74(2):606–615. doi:10.1111/biom.12791
- Nickell S. Biases in dynamic models with fixed effects. Econometrica 1981;49(6):1417–1426. doi:10.2307/1911408
- Nylund KL, Asparouhov T, Muthén BO. Deciding on the number of classes in latent class analysis and growth mixture modeling: a Monte Carlo simulation study. Struct Equ Modeling 2007;14(4):535–569. doi:10.1080/10705510701575396
- Oates CJ, Smith JQ, Mukherjee S, Cussens J. Exact estimation of multiple directed acyclic graphs. Stat Comput 2016;26(4):797–811. doi:10.1007/s11222-015-9570-9
- Ollier E, Viallon V. Regression modelling on stratified data with the lasso. Biometrika 2017. doi:10.1093/biomet/asw065
- Oxtoby NP, Leyland LA, Aksman LM, et al. Sequence of clinical and neurodegeneration events in Parkinson's disease progression. Brain 2021;144(3):975–988. doi:10.1093/brain/awaa461
- Pagano G, De Micco R, Yousaf T, Wilson H, Chandra A, Politis M. REM behavior disorder predicts motor progression and cognitive decline in Parkinson disease. Neurology 2018;91(10):e894–e905. doi:10.1212/WNL.0000000000006134
- Pagano G, Taylor KI, Anzures Cabrera J, et al. Prasinezumab slows motor progression in rapidly progressing early-stage Parkinson's disease. Nat Med 2024;30(4):1096–1103. doi:10.1038/s41591-024-02886-y
- Pal S, Ghosh D, Yang S. Penalized FCI for causal structure learning in a sparse DAG for biomarker discovery in Parkinson's disease. Ann Appl Stat 2026;20(3):1873–1893. doi:10.1214/26-AOAS2191
- Pan W, Su C, Maasch JRMA, Chen K, Henchcliffe C, Wang F. Learning phenotypic associations for Parkinson's disease with longitudinal clinical records. AMIA Jt Summits Transl Sci Proc 2024;2024:374–383; preprint medRxiv 2020, doi:10.1101/2020.03.15.20036657
- Passaretti M, Veréb D, Mijalkov M, et al. Clinical progression and genetic pathways in body-first and brain-first Parkinson's disease. Mol Neurodegener 2025;20:74. doi:10.1186/s13024-025-00866-5
- Peters J, Bühlmann P, Meinshausen N. Causal inference by using invariant prediction: identification and confidence intervals. J R Stat Soc B 2016;78(5):947–1012. doi:10.1111/rssb.12167
- Petkus AJ, Filoteo JV, Schiehser DM, Gomez ME, Petzinger G. Worse cognitive performance predicts increased anxiety and depressive symptoms in patients with Parkinson's disease: a bidirectional analysis. Neuropsychology 2019;33(1):35–46. doi:10.1037/neu0000498
- Petrella JR, Jiang J, Sreeram K, Dalziel S, Doraiswamy PM, Hao W. Personalized computational causal modeling of the Alzheimer disease biomarker cascade. J Prev Alzheimers Dis 2024;11(2):435–444. doi:10.14283/jpad.2023.134
- Plis S, Danks D, Freeman C, Calhoun V. Rate-agnostic (causal) structure learning. Adv Neural Inf Process Syst 2015;28:3303–3311
- Poulet PE, Durrleman S. Multivariate disease progression modeling with longitudinal ordinal data. Stat Med 2023;42(18):3164–3183. doi:10.1002/sim.9770
- Pyatigorskaya N, Yahia-Cherif L, Valabregue R, et al. Parkinson disease propagation using MRI biomarkers and partial least squares path modeling. Neurology 2021;96(3):e460–e471. doi:10.1212/WNL.0000000000011155
- Raj A, LoCastro E, Kuceyeski A, Tosun D, Relkin N, Weiner M. Network diffusion model of progression predicts longitudinal patterns of atrophy and metabolism in Alzheimer's disease. Cell Rep 2015;10(3):359–369. doi:10.1016/j.celrep.2014.12.034
- Raket LL. Progression models for repeated measures: estimating novel treatment effects in progressive diseases. Stat Med 2022;41(28):5537–5557. doi:10.1002/sim.9581
- Ramsey JD, Hanson SJ, Hanson C, Halchenko YO, Poldrack RA, Glymour C. Six problems for causal inference from fMRI. NeuroImage 2010;49(2):1545–1558. doi:10.1016/j.neuroimage.2009.08.065
- Rocchi N, Zanga A, Bernasconi A, et al. A causal discovery workflow for rare diseases: experts-in-the-loop analysis of sparse longitudinal data. J Med Syst 2026;50(1):9. doi:10.1007/s10916-025-02327-4
- Romano D, Tosi G; ADNI. Divergent network trajectories in mild cognitive impairment: converters versus non-converters to Alzheimer's disease. J Alzheimers Dis 2026;110(3):1414–1427. doi:10.1177/13872877261423962
- Ryan O, Hamaker EL. Time to intervene: a continuous-time approach to network analysis and centrality. Psychometrika 2022;87:214–252. doi:10.1007/s11336-021-09767-0
- Saadawy M, Saadawy E. NeuroCausal prior-weighted structure learning (NCP-WSL): disentangling a shared latent causal manifold across Alzheimer's, Parkinson's and healthy aging. SSRN preprint 2026. doi:10.2139/ssrn.7119099
- Saeed B, Panigrahi S, Uhler C. Causal structure discovery from distributions arising from mixtures of DAGs. Proc ICML 2020, PMLR 119:8336–8345. arXiv:2001.11940
- Schroeders U, Zimmermann J, Wicke T, et al. Dynamic interplay of cognitive functioning and depressive symptoms in patients with Parkinson's disease. Neuropsychology 2022;36(4):266–278. doi:10.1037/neu0000795
- Schultzberg M, Muthén B. Number of subjects and time points needed for multilevel time-series analysis: a simulation study of dynamic structural equation modeling. Struct Equ Modeling 2018;25(4):495–515. doi:10.1080/10705511.2017.1392862
- Schumacher JG, Zhang X, Macklin EA, et al. Baseline α-synuclein seeding activity and disease progression in sporadic and genetic Parkinson's disease in the PPMI cohort. EBioMedicine 2025;119:105866. doi:10.1016/j.ebiom.2025.105866
- Schuurman NK, Hamaker EL. Measurement error and person-specific reliability in multilevel autoregressive modeling. Psychol Methods 2019;24:70–91. doi:10.1037/met0000188
- Seppi K, Ray Chaudhuri K, Coelho M, et al. Update on treatments for nonmotor symptoms of Parkinson's disease: an evidence-based medicine review. Mov Disord 2019;34(2):180–198. doi:10.1002/mds.27602
- Severson KA, Chahine LM, Smolensky LA, et al. Discovery of Parkinson's disease states and disease progression modelling: a longitudinal data study using machine learning. Lancet Digit Health 2021;3(9):e555–e564. doi:10.1016/S2589-7500(21)00101-1
- Shan AD, Zhang H, Gao MX, et al. Altered effective connectivity in Parkinson's disease patients with rapid eye movement sleep behavior disorder. Quant Imaging Med Surg 2025;15(1):352–369. doi:10.21037/qims-24-1196
- Shawa Z, Shand C, Taylor B, et al. Neuroimaging-based data-driven subtypes of spatiotemporal atrophy due to Parkinson's disease. Brain Commun 2025;7(2):fcaf146. doi:10.1093/braincomms/fcaf146
- Simuni T, Caspell-Garcia C, Coffey C, Lasch S, Tanner C, Marek K; PPMI Investigators. How stable are Parkinson's disease subtypes in de novo patients: analysis of the PPMI cohort? Parkinsonism Relat Disord 2016;28:62–67. doi:10.1016/j.parkreldis.2016.04.027
- Skripnikov A, Michailidis G. Joint estimation of multiple network Granger causal models. Econom Stat 2019;10:120–133. doi:10.1016/j.ecosta.2018.08.001
- Solovyeva K, Danks D, Abavisani M, Plis S. Causal learning through deliberate undersampling. Proc CLeaR 2023, PMLR 213:518–530
- Städler N, Bühlmann P, van de Geer S. ℓ1-penalization for mixture regression models. TEST 2010;19(2):209–256. doi:10.1007/s11749-010-0197-z
- Strobl EV. Causal discovery with a mixture of DAGs. Mach Learn 2023;112(11):4201–4225. doi:10.1007/s10994-022-06159-y
- Stroustrup N, Anthony WE, Nash ZM, et al. The temporal scaling of Caenorhabditis elegans ageing. Nature 2016;530(7588):103–107. doi:10.1038/nature16550
- Su C, Hou Y, Xu J, et al. Identification of Parkinson's disease PACE subtypes and repurposing treatments through integrative analyses of multimodal data. NPJ Digit Med 2024;7:184. doi:10.1038/s41746-024-01175-9
- Szymkowicz SM, Jones JD, Timblin H, et al. Apathy as a within-person mediator of depressive symptoms and cognition in Parkinson's disease: longitudinal mediation analyses. Am J Geriatr Psychiatry 2022;30(6):664–674. doi:10.1016/j.jagp.2021.11.007
- Taddé BO, Jacqmin-Gadda H, Dartigues JF, Commenges D, Proust-Lima C. Dynamic modeling of multivariate dimensions and their temporal relationships using latent processes: application to Alzheimer's disease. Biometrics 2020;76(3):886–899. doi:10.1111/biom.13168
- Terluin B, de Boer MR, de Vet HCW. Differences in connection strength between mental symptoms might be explained by differences in variance: reanalysis of network data did not confirm staging. PLoS One 2016;11(11):e0155205. doi:10.1371/journal.pone.0155205
- van der Velden RMJ, Mulders AEP, Drukker M, Kuijf ML, Leentjens AFG. Network analysis of symptoms in a Parkinson patient using experience sampling data: an n = 1 study. Mov Disord 2018;33(12):1938–1944. doi:10.1002/mds.93
- Varambally S, Ma Y-A, Yu R. Discovering mixtures of structural causal models from time series data. Proc ICML 2024, PMLR 235:49171–49202. arXiv:2310.06312
- Voelkle MC, Oud JHL, Davidov E, Schmidt P. An SEM approach to continuous time modeling of panel data: relating authoritarianism and anomia. Psychol Methods 2012;17:176–192. doi:10.1037/a0027543
- Vu TC, Nutt JG, Holford NHG. Progression of motor and nonmotor features of Parkinson's disease and their response to treatment. Br J Clin Pharmacol 2012;74:267–283. doi:10.1111/j.1365-2125.2012.04192.x
- Wang Y, Segarra S, Uhler C. High-dimensional joint estimation of multiple directed Gaussian graphical models. Electron J Stat 2020;14(1):2439–2483. doi:10.1214/20-EJS1724
- Wang Y, Squires C, Belyaeva A, Uhler C. Direct estimation of differences in causal graphs. Adv Neural Inf Process Syst 2018;31. arXiv:1802.05631
- Yajima M, Telesca D, Ji Y, Müller P. Detecting differential patterns of interaction in molecular pathways. Biostatistics 2015;16(2):240–251. doi:10.1093/biostatistics/kxu054
- Young AL, Marinescu RV, Oxtoby NP, et al. Uncovering the heterogeneity and temporal complexity of neurodegenerative diseases with Subtype and Stage Inference. Nat Commun 2018;9:4273. doi:10.1038/s41467-018-05892-0
- Zandstra MG, Scheepers FE, Lunansky G, et al. Symptom network dynamics during antipsychotic treatment in first-episode psychosis. Schizophr Bull 2026;52(2):sbag016. doi:10.1093/schbul/sbag016
- Zeileis A, Hothorn T, Hornik K. Model-based recursive partitioning. J Comput Graph Stat 2008;17(2):492–514. doi:10.1198/106186008X319331
- Zeng Q, Guan X, Guo T, et al. The ventral intermediate nucleus differently modulates subtype-related networks in Parkinson's disease. Front Neurosci 2019;13:202. doi:10.3389/fnins.2019.00202
- Zhang M, Zhou S, Yang P, et al. The precuneus region drives brain network changes in tremor-dominant Parkinson's disease: insights from a morphological causal analysis. MedComm 2025;6(11):e70441. doi:10.1002/mco2.70441
- Zheng H, Petrella JR, Doraiswamy PM, Lin G, Hao W; ADNI. Data-driven causal model discovery and personalized prediction in Alzheimer's disease. NPJ Digit Med 2022;5:137. doi:10.1038/s41746-022-00632-7
- Zhou C, Wang L, Cheng W, et al. Two distinct trajectories of clinical and neurodegeneration events in Parkinson's disease. NPJ Parkinsons Dis 2023;9:111. doi:10.1038/s41531-023-00556-3 (cited as Zhou 2023a)
- Zhou F, He K, Ni Y. Individualized causal discovery with latent trajectory embedded Bayesian networks. Biometrics 2023;79(4):3191–3202. doi:10.1111/biom.13843 (cited as Zhou 2023b)
- Zhou Y, Tinaz S, Tagare HD. Robust Bayesian analysis of early-stage Parkinson's disease progression using DaTscan images. IEEE Trans Med Imaging 2021;40(2):549–561. doi:10.1109/TMI.2020.3031478
