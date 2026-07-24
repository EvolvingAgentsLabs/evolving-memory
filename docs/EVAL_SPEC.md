> **Salvaged eval design.** This came from `EvolvingAgentsLabs/agent-loadout`,
> a design-only repo archived 2026-07-24 whose thesis — learning good agent
> configurations from execution history — this engine already implements. The
> metric design is the part worth keeping: **Cost per Successful Plan** as the
> headline, **Learning Slope** as the differentiator, and an explicit
> falsification clause. It is a spec, not a measurement; nothing here has been
> run.

---

# Artifact 2 — Cognitive Configuration Benchmark (CCB) Specification

**Project:** agent-loadout (working name)
**Audience:** Claude Code building this project from scratch
**Status:** Design specification, v0.2
**Companion to:** ARCHITECTURE.md, DEMO_BIOENG.md
**Depends on:** evolving-memory (CTE), skillos (markdown format conventions)

---

## 0. Read this first

This document specifies how to validate that agent-loadout actually produces value. The headline metric is **Cost per Successful Plan (CPSP)** in USD, and the differentiating metric is **Learning Slope** — the rate at which CPSP decreases as the engine accumulates execution history for a plan family.

This benchmark exists because the engine's claim ("loadouts learned from execution history outperform hand-tuned and RAG-selected loadouts, and improve with use") is non-obvious and easy to dispute. Without numbers, it's a marketing claim. With numbers, it's a defensible category.

You are reading this in order to implement the benchmark. Sections 1-3 explain what we are measuring and why. Sections 4-9 specify the implementation. Section 10 is the recommended order.

---

## 1. The validation thesis in one paragraph

If agent-loadout does what it claims, then a plan executed with the engine should be cheaper, faster, and more reliable than the same plan executed with vanilla Claude, with manually-configured Claude, or with a RAG-based configuration selector. Furthermore, after N executions of plans in the same family, the engine's cost-per-plan should *decrease measurably* while the baselines stay flat — because the engine consolidates the configurations that worked. CCB measures both the headline cost and the learning slope. If neither materializes, the engine has no defensible value and we should publish the negative result and move on.

## 2. Why CPSP and not just "task success"

Three reasons CPSP is the right headline:

First, "task success" alone is a misleading metric in agent infra because frontier models can solve almost anything given enough budget. The interesting question is *what does it cost to succeed*. A baseline that succeeds at $4.00/plan loses to a configuration that succeeds at $0.40/plan, even though both have 100% success rate.

Second, CPSP decomposes into terms a buyer recognizes: LLM cost, escalation cost, security-rejection cost. A buyer reading the result can replace our cost assumptions with their own and re-derive the conclusion. This makes the benchmark *usable* by buyers, not just citable.

Third, CPSP creates a clean falsification path. If we cannot beat baselines on CPSP for any meaningful plan family, the engine has no reason to exist. The metric is uncomfortable in a useful way.

## 3. The differentiating metric: Learning Slope

CPSP at run 1 measures whether the engine helps on first contact. **Learning Slope** measures whether the engine compounds with use. It is the slope of CPSP over consecutive runs of plans in the same family:

```
Learning Slope = (CPSP_run_1 - CPSP_run_N) / N
```

For a baseline with no learning (vanilla Claude, hardcoded config, RAG-selected config), Learning Slope ≈ 0 — every run costs roughly the same as the first. For agent-loadout, Learning Slope should be *positive and significant*, because each successful execution contributes to consolidated ConfigurationNodes that the router returns for subsequent runs.

This is the metric that sells. Vendors compete on absolute cost; *no vendor competes on improvement-with-use* because no vendor's architecture supports it. If the chart shows agent-loadout's CPSP curve dropping while the others stay flat, that single image is the entire pitch.

## 4. Scope and non-goals

**In scope:**
- Measuring cost-per-plan for plan-execution agents across different memory/configuration strategies
- Measuring how cost evolves with repeated execution in the same plan family
- Measuring whether the security pass blocks unsafe configurations correctly
- Producing reproducible results with a single command and bounded API cost

**Out of scope (explicitly):**
- General agent capability ranking (use other benchmarks for this)
- Conversational longitudinal memory (different problem)
- Multi-tenant performance under load
- Latency-bound use cases (e.g., voice agents)
- Cross-customer transfer of consolidated configs

## 5. The headline metric: CPSP

For each system evaluated:

```
CPSP = (LLM_cost + Escalation_cost + Security_rejection_cost) / Plans_succeeded
```

Where:

**LLM_cost** = sum of input + output tokens × current Anthropic pricing (pinned per release). Includes ALL tokens consumed by the system: agent calls, sub-agent calls, skill invocations, memory operations, dream-cycle calls (amortized per execution), security-pass calls (amortized per execution), embedding calls. Honest accounting. No hidden cost.

**Escalation_cost** = number of plans that escalated to human × $50 USD (industry-standard fully-loaded cost of a human reviewing an automated agent's failed plan; configurable per release). Escalation is detected by the agent emitting `[ESCALATE]` token or by the judge marking the plan as requiring human intervention.

**Security_rejection_cost** = number of plans where the security pass blocked the configuration × $20 USD (cost of falling back to a manual or default configuration; configurable). Important: a security rejection is *not* a failure — it is a correct refusal — but it does have operational cost. Reporting this honestly prevents the engine from gaming the metric by being permissive.

**Plans_succeeded** = count of plans that the judge marked as completed correctly without escalation. A plan succeeds only if (a) all required deliverables were produced, (b) deliverables passed the domain-specific quality check, (c) no security violation occurred during execution.

This is the headline. Everything else decomposes or contextualizes it.

## 6. Secondary metrics

**M1 — Per-plan token cost** (USD), trended over consecutive runs. The interesting visual is cost-per-plan over time. Baselines stay flat; agent-loadout drops. This is the source of the money-shot chart.

**M2 — Plan success rate** (%), reported per system per plan family. A system can have low CPSP but also low success rate; reporting both prevents the engine from looking good by simply giving up cheaply.

**M3 — Plan steps** (count). Number of distinct agent steps to complete a plan. Lower is better given equal success rate.

**M4 — Configuration reuse rate** (% of executions where the engine returned a stored configuration vs cold-started). Measures whether the engine is actually consolidating learning, not just executing well by luck on each plan.

**M5 — Learning Slope** (USD/run reduction). Computed as defined above. The differentiating metric.

**M6 — Security catch rate** (% of deliberately-misconfigured plans that the security pass blocked). For the security-validation track of the benchmark.

**M7 — Security false-positive rate** (% of valid plans incorrectly blocked by security pass). The companion to M6 — if M6 is high but M7 is also high, the security pass is uselessly paranoid.

## 7. Plan domains

Two domains in v0.1, chosen for high repetition + clear subtype variation + measurable cost of bad configuration. Both have markets in the billions and are recognizable to general technical audiences.

### 7.1 Domain R — Recruiting / Candidate Screening

Simulated mid-stage SaaS company with an in-house talent team. Plans are candidate screening tasks across heterogeneous role types — the loadout that works for screening a Senior ML Engineer is *very different* from the one for a Sales Executive or a Clinical Research Coordinator. This heterogeneity is exactly what forces the engine to learn role-family-specific configurations rather than collapsing to one generic config.

**Plan families** (each family will be exercised 10 times in the longitudinal track):

- R-1: "Screen candidates for Senior ML Engineer role given resumes + job description; produce ranked shortlist with rationale, technical fit assessment, red flags, and suggested interview questions."
- R-2: "Screen candidates for Enterprise Account Executive role given resumes + job description; produce shortlist with deal-size fit, industry vertical match, ramp-up risk assessment, and reference-check priority."
- R-3: "Screen candidates for Clinical Research Coordinator role given resumes + protocol; produce shortlist with regulatory experience match, therapeutic-area fit, GCP-certification status, and trial-management capability assessment."
- R-4: "Screen candidates for Senior Product Designer role given resumes + portfolio links; produce shortlist with portfolio-quality assessment, design-system maturity match, fidelity to JD scope, and presentation-skill estimation."
- R-5: "Screen candidates for Site Reliability Engineer role given resumes + on-call expectations; produce shortlist with incident-response experience match, observability-stack fit, scale-of-systems handled, and burnout-risk assessment."

Each family requires *materially different* sub-agents, skills, and KB slices. R-1 needs an ML-domain agent + skills for evaluating GitHub/papers + a KB on ML role rubrics. R-3 needs a clinical-domain agent + skills for parsing IRB and GCP credentials + a KB on therapeutic areas. The engine has to learn that R-1 and R-3 share *almost nothing* in their loadouts despite both being "screening tasks."

**Why recruiting:** Repetition is massive (every company does this hundreds of times per year). Cost of bad config is famous (mis-hire = 1-2x annual salary; missed great hire = permanent opportunity cost). The market is enormous (LinkedIn $15B+ ARR, plus the entire HR-tech ecosystem). And it forces the security pass into the spotlight: recruiting in the US is governed by EEOC, in the EU by GDPR + national equality laws, in the UK by the Equality Act. A configuration that quietly leaks protected attributes into screening logic is a compliance violation. The security pass blocking those configs is a feature buyers actually pay for.

### 7.2 Domain C — Competitive Intelligence

Simulated product/strategy team at a mid-stage company that produces competitive analyses regularly. Plans are deep-dive competitive analyses across heterogeneous industry verticals — the loadout that works for analyzing competition in B2B SaaS is *very different* from the one for consumer apps, fintech, biotech, or DevTools. As with recruiting, this heterogeneity forces the engine to specialize loadouts per industry vertical.

**Plan families:**

- C-1: "Analyze a B2B SaaS competitor given their public marketing, pricing page, and recent funding announcements; produce a competitive brief with positioning, pricing strategy, ICP overlap with us, GTM motion, and 3 strategic implications."
- C-2: "Analyze a consumer mobile app competitor given app store presence, recent updates, and visible engagement signals; produce a competitive brief with feature parity assessment, monetization model, growth trajectory, UX differentiation, and 3 strategic implications."
- C-3: "Analyze a fintech competitor given their public disclosures, regulatory filings, and product documentation; produce a brief with market segment, regulatory posture, unit economics estimate, partner ecosystem, and 3 strategic implications."
- C-4: "Analyze a biotech competitor given their clinical pipeline, publications, and patent activity; produce a brief with therapeutic-area focus, pipeline maturity, IP position, partnership signals, and 3 strategic implications."
- C-5: "Analyze a DevTools/infrastructure competitor given their documentation, GitHub activity, and developer community presence; produce a brief with technical positioning, adoption signals, OSS strategy, integration ecosystem, and 3 strategic implications."

Each family requires materially different loadouts: C-1 needs a SaaS-economics agent + skills for parsing pricing tiers + a KB on SaaS metrics; C-4 needs a biotech-domain agent + skills for clinical-trial parsing + a KB on therapeutic areas and FDA pathways. The engine has to learn industry-specific clusters.

**Why competitive intelligence:** Sustained repetition in every product/marketing org. Cost of bad analysis is real but indirect (wrong roadmap calls, wrong pricing decisions, wrong M&A signals). The market is broad — every PM, every strategy team, every VC analyst, every marketing agency. And it exhibits the *evolution* dimension of the engine especially well: competitive landscapes change month-to-month, so the engine should learn not just configs but also *when configs need to mutate* as a vertical evolves.

### 7.3 Why these two and not others

Both domains share three properties critical to the benchmark: high plan repetition with strong subtype variation; clear separation between "good loadout" and "bad loadout"; recognizable to a general technical audience without specialized credentials. They differ deliberately on operational tempo (recruiting is transactional, competitive intel is analytical) so the engine demonstrates breadth across modes.

Domains explicitly NOT chosen for v0.1: contracts (legal review fatigue in the AI space; market is crowded with vertical SaaS), security incident response (too narrow to general audiences), DevOps (too "demo for engineers"), customer support (overlaps with conversational benchmarks; not the engine's strongest fit). These could appear in v0.3+.

## 8. The four systems evaluated

All systems have access to the same underlying LLM (Claude Sonnet 4.6) and the same SkillOS-format library of agents, skills, and KB chunks. The independent variable is *how the system selects and uses the configuration*.

**B1 — Vanilla Claude.** Single Sonnet 4.6 invocation per plan with the plan text as the user message and a generic system prompt. No sub-agents, no skill loading, no KB injection beyond what fits in the prompt. Represents "do nothing — just ask the model."

**B2 — Hardcoded configuration.** Each plan family has a hand-tuned configuration of agents/skills/KB the developer would write today. Configuration does not change across runs. Represents the current state of the art for production agent deployments.

**B3 — RAG-selected configuration.** For each incoming plan, embed the plan text and retrieve the top-K most similar agents, skills, and KB chunks from the SkillOS library by cosine similarity. Pass them all into context. Represents the obvious "use embeddings to pick stuff" approach, which is what most teams reach for first.

**B4 — agent-loadout (SUT).** The engine described in Artifact 1. Returns a `LoadoutDecision`, executes with the resulting configuration, captures the ExecutionTrace, and runs dream cycles between batches.

Four baselines is the limit. Adding more (Letta, Mem0, LangGraph) would be valuable but those frameworks are about reasoning memory, not configuration selection — they would all collapse into a variant of B1 or B3 for this benchmark. Note this limitation in the report.

## 9. Protocol

### 9.1 The longitudinal track (measures Learning Slope)

For each plan family (10 families across both domains):

1. Generate 10 distinct plan instances in the family. Plans share family-level structure but differ in specifics — same shape, different details. (E.g., 10 different ML Engineer screening tasks with different role specifics, different candidate pools, different seniority bands.)
2. For each system, execute the 10 plans in sequence.
3. Between plans, give B4 a chance to run a dream cycle (other systems are unaffected because they don't have one).
4. Record CPSP per execution.
5. Plot CPSP across executions 1-10. The Learning Slope is computed from this curve.

Total: 10 families × 10 plans × 4 systems = **400 plan executions** for the longitudinal track.

### 9.2 The security track (measures M6 and M7)

A separate set of 30 plans per domain (60 total) designed to test the security pass:

- 20 valid plans that should pass security (tests M7 false-positive rate).
- 10 misconfigured plans that should be blocked (tests M6 catch rate). Examples of misconfigurations exercised:
  - Recruiting: a configuration that includes a sub-agent whose system prompt references candidate name or photo as ranking signals (EEOC violation); a configuration that loads KB content with discriminatory language; a configuration where one sub-agent says "always escalate borderline cases" and another says "never escalate to keep throughput high."
  - Competitive intel: a configuration that loads a KB slice containing leaked NDA material from a previous engagement; a configuration whose sub-agents are designed for public-data analysis but have access to a proprietary customer-data KB (scope creep); a configuration that includes a sub-agent prompted to "be aggressive in claims" alongside one prompted to "be cautious and citation-heavy."

The security track only meaningfully runs against B4 — the other systems have no security pass. Report B4's M6 and M7 with explicit acknowledgment that the other systems would all have M6=0% (block nothing) and M7=0% (also block nothing). The point is not "B4 wins security"; the point is "B4 has security at all and here is whether it works."

Total: **60 additional plan executions** for the security track, all on B4.

### 9.3 Plan generation and judging

Plan instances are generated by a *plan author LLM* (Claude Opus 4.7) given the family description plus a randomization seed. Plans are reviewed by hand for the v0.1 release to ensure family coherence. For ongoing benchmark releases, automated structural checks (does the plan have the required components for its family?) replace manual review.

Plan execution is judged by a *judge LLM ensemble* (Claude Opus 4.7 + GPT-4-class + Llama 3.3 70B) using domain-specific rubrics, with median verdict and human spot-check on 20% of judgments. Judge prompts are published in the repo. Disagreement >1 point on the 5-point rubric flags the case for human review.

For recruiting, the rubric explicitly excludes any candidate-attribute scoring that could surface protected-class bias — judges score *quality of fit assessment*, not *candidate quality* directly. This keeps the benchmark itself EEOC-clean even though the security track explicitly probes EEOC violations in evaluated configurations.

## 10. Cost budget

Estimated cost per full v0.1 benchmark run: **~$120-180 USD** in API costs at current Anthropic pricing.

Decomposition:
- 400 longitudinal plan executions × ~$0.10-0.30 per execution (varies by system; B1 is cheap, B4 has dream-cycle overhead) ≈ $80
- 60 security-track executions × ~$0.30 ≈ $18
- Judge ensemble × all executions ≈ $30-50
- Plan author LLM ≈ $5-10

This is intentionally affordable so independent replication is realistic. If costs grow significantly between v0.1 and v0.2 (e.g., longer plan families, more domains), pin and document.

## 11. Reproducibility requirements

- Single command runs the full benchmark: `ccb run --systems all --domains all --track all`
- Total cost pinned in `pricing.lock.yaml` at release time
- Plans, judge prompts, rubrics version-locked per release
- Random seeds fixed; runs deterministic given identical model versions
- Model versions pinned with API dates
- Full transcripts archived for every execution
- CI runs a smoke-test (1 plan family, all systems, 3 plans each) on every PR

## 12. Conflict of interest disclosure

EvolvingAgentsLabs (Matías Molinas, Ismael Faro) authors both CCB and agent-loadout (one of the systems evaluated). This is a structural bias.

Mitigations:
1. Full reproducibility — anyone can rerun and verify.
2. Explicit baselines that include the strongest available alternatives at each level (vanilla, hardcoded, RAG).
3. Honest reporting — if B4 loses on a family, that family's losing transcripts are included prominently.
4. External replication invited; we will publicly link any independent replications, including unfavorable ones.
5. v0.2 onwards: invite a co-maintainer from outside our org to review plans and rubrics.

## 13. The money shots

Two charts that will anchor the launch:

### Chart 1 — CPSP by system (averaged across plan families)

Bar chart, four bars (one per system), CPSP on Y-axis. If B4 wins, the bar is visibly shorter. Decomposition stacked: LLM cost in one color, escalation cost in another, security rejection (B4 only) in a third.

### Chart 2 — Learning Slope (the differentiator)

Line chart, X-axis = execution number 1-10, Y-axis = CPSP. Four lines (one per system). Expected pattern: B1, B2, B3 are roughly flat — they do not learn. B4 starts comparable to B2/B3 and *drops measurably* by execution 5-7, then plateaus near a lower CPSP. **This is the chart that sells.**

If Chart 1 shows B4 winning by ≥30% and Chart 2 shows B4 with a Learning Slope ≥3x any baseline, we have a publishable result. If neither materializes, we publish the negative result and pivot. If only Chart 1 materializes, we have a useful product but not a defensible category — still publishable, with weaker positioning.

## 14. Implementation layout

Mirrors the agent-loadout repo structure where possible.

```
ccb/
├── pyproject.toml
├── README.md
├── BENCHMARK.md                   # this document
├── pricing.lock.yaml
├── plan_families/
│   ├── domain_R/
│   │   ├── R-1_ml_engineer.yaml
│   │   ├── R-2_account_executive.yaml
│   │   ├── R-3_clinical_research_coordinator.yaml
│   │   ├── R-4_product_designer.yaml
│   │   └── R-5_sre.yaml
│   └── domain_C/
│       ├── C-1_b2b_saas.yaml
│       ├── C-2_consumer_mobile.yaml
│       ├── C-3_fintech.yaml
│       ├── C-4_biotech.yaml
│       └── C-5_devtools.yaml
├── plan_instances/                # generated; one subdir per family
│   ├── domain_R/
│   └── domain_C/
├── security_track/
│   ├── valid_plans/               # 20 per domain
│   └── misconfigured_plans/       # 10 per domain
├── customer_skillos/              # the SkillOS-format library shared across systems
│   ├── agents/                    # diverse agents covering both domains
│   ├── skills/                    # diverse skills
│   └── kb/                        # diverse KB chunks
├── src/ccb/
│   ├── __init__.py
│   ├── cli.py                     # ccb run / ccb smoke
│   ├── plan_generator.py          # generates instances from families
│   ├── judges/
│   │   ├── ensemble.py
│   │   ├── rubrics/
│   │   │   ├── domain_R_rubric.md
│   │   │   └── domain_C_rubric.md
│   │   └── human_spot_check.py
│   ├── systems/
│   │   ├── base.py                # PlanExecutor ABC
│   │   ├── b1_vanilla.py
│   │   ├── b2_hardcoded.py
│   │   ├── b3_rag_selected.py
│   │   └── b4_agent_loadout.py    # imports from agent-loadout
│   ├── runner.py                  # orchestrates longitudinal + security tracks
│   ├── metrics/
│   │   ├── cpsp.py                # the headline metric
│   │   ├── learning_slope.py
│   │   └── security_metrics.py
│   ├── reporting/
│   │   ├── report_generator.py
│   │   ├── chart_cpsp.py          # Chart 1
│   │   └── chart_learning_slope.py # Chart 2 (the money shot)
│   └── transcripts/
│       └── archive.py
├── tests/
└── results/                       # pinned per release
```

## 15. Recommended implementation order

If you have one focused workweek for the benchmark (in parallel or after agent-loadout itself):

1. **Repo scaffolding** — `pyproject.toml`, dependencies, basic CLI. ~2h.
2. **Plan family schemas + 1 family fully written** — define the YAML schema, write R-1 in full as the reference family. ~3h.
3. **Plan generator** — given a family + seed, generate 10 instances using Claude Opus. Manual review pass. ~3h.
4. **Customer SkillOS library** — agents/skills/KB markdown files that all four systems can use. Start with what's needed for R-1 only. ~4h.
5. **B1 vanilla executor** — simplest baseline. Establishes the harness. ~2h.
6. **B2 hardcoded executor** — needs config files per family. Establishes the "what would a developer do today" baseline. ~3h.
7. **B3 RAG-selected executor** — embeddings + retrieval + assembly. Establishes the "obvious smart approach" baseline. ~4h.
8. **Judge ensemble + R-1 rubric** — the rubric is family-specific; R-1 first. ~3h.
9. **Runner + CPSP computation** — orchestrate longitudinal track for R-1 across B1, B2, B3. Get first numbers. ~4h.
10. **B4 agent-loadout integration** — depends on agent-loadout being functional through its step 11. ~3h.
11. **First end-to-end longitudinal run for R-1** — all four systems, 10 plans. Look at numbers honestly. ~2h (mostly waiting).
12. **Expand to other plan families** — add R-2 through R-5, then C-1 through C-5. ~6h.
13. **Security track** — 60 plans + B4-only execution + M6/M7 reporting. ~5h.
14. **Reporting + charts** — generate Chart 1 and Chart 2 from results. ~3h.
15. **Launch artifact** — blog post draft, repo README polished, transcripts archived. ~4h.

That is roughly 51 hours, distributed over 4-6 weeks at 10h/week. Front-load steps 1-9 to get the harness producing baseline numbers before agent-loadout itself is fully ready. If the baseline numbers look weird, the harness has bugs — fix before introducing B4 to avoid attributing harness bugs to B4.

## 16. What this benchmark deliberately does not claim

- It does not claim agent-loadout is "the best agent framework." It claims that for plan-execution use cases with repeated families, agent-loadout produces lower CPSP and a steeper Learning Slope than three named alternatives.
- It does not claim that synthetic plan instances predict real production plans with high fidelity. It claims they provide a controlled, reproducible stress test.
- It does not claim CPSP dollar values predict real production economics. It claims relative comparison is defensible and the absolute model is customizable per buyer.
- It does not claim the security pass catches all real-world misconfigurations. It claims it catches the categories specified in the security track and provides a framework that can be extended.
- It does not claim B4 will win every plan family. If B4 loses on a family, that result is reported with full transparency and used to identify where the engine needs work.

## 17. Honest constraints worth naming

Three things will only be known after running:

1. Whether 10 executions per family is enough to demonstrate Learning Slope. If the slope is real but takes 20-30 executions to manifest, v0.2 needs longer runs and bigger budget. If 10 is plenty, we're fine.
2. Whether judging plan execution is reliable enough across the ensemble. Plans have richer outputs than conversational replies; judge agreement may be lower. If inter-judge agreement is below 80%, the rubric needs sharpening before publishing.
3. Whether the synthetic security misconfigurations are realistic. Real misconfigurations in production might be subtler than the ones we generate, making M6 look better than it really is. v0.2 should add a "wild" security track sourced from real (anonymized) production incidents from at least one design partner.

If any of these surface during v0.1 execution, name them in the report rather than papering over.

---

## Appendix A — Why these two domains have natural buyer demand

Recruiting screening is not just a benchmark domain — it's a buyer-funded category. Companies building AI screening tools (Eightfold, Paradox, HireVue, plus dozens of YC-funded startups) are spending heavily on configuration tuning. They would all benefit from learned configurations per role family. The benchmark gives them a number they can compare their internal tooling against. That's distribution.

Competitive intelligence is similar but earlier in market formation. Tools like Crayon and Klue exist but most competitive analysis is still done in Notion docs and Google Slides by humans. An engine that produces high-quality competitive briefs at low CPSP creates a wedge into product/strategy teams. For the benchmark, this means our results land in front of PMs, founders, and strategy consultants — a different audience from the recruiting one, doubling our distribution surface.

Both domains have the property that the *output is a written artifact* (shortlist + rationale; competitive brief). Written artifacts are easy to judge, easy to share, easy to compare across systems. This makes the benchmark's results visually compelling and qualitatively defensible, not just numerically defensible.

## Appendix B — Real-customer validation in v0.2

Synthetic plans validate the engine mechanically. Real plans validate it commercially. v0.2 should add a real-customer validation track per domain. Acquire one design partner per domain (see ARCHITECTURE.md Appendix B for partner profiles) and run 30-day measurement on their actual plan stream. Even one customer's worth of real data is more valuable to the launch story than 1,000 synthetic plans.

Do not block v0.1 on partner acquisition. Ship v0.1 with synthetic results, use those results to attract partners, then ship v0.2 with real-customer numbers.
