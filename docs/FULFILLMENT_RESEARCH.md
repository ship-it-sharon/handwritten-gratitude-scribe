# Fulfillment & Font Flexibility Research

_Researched 2026-07-10. Question: can we offer the SAME font library across
all three tiers (print-at-home / printed / pen-written), or do pen-robot
vendors significantly limit font options?_

## TL;DR

- **Printed tiers: zero constraint.** With Lob/PostGrid-class vendors we
  render the card artwork ourselves (PDF), so any font we license works.
  Print-at-home is likewise unconstrained.
- **Pen-written tier: real constraint, but solvable.** Writing robots
  require **single-stroke ("open contour") vector fonts** — standard
  TTF/OTF outline fonts describe letter *outlines*, which a pen would
  trace as hollow double-line letters. No vendor documents accepting
  arbitrary customer TTFs; they operate from their own style catalogs
  plus paid custom-font recreation.
- **Path to a unified library:** source library fonts that ship in both
  outline **and** single-stroke formats (these exist commercially), and/or
  commission the pen vendor to onboard our fonts as custom styles. Verify
  per-font onboarding cost before committing to the "same library, three
  tiers" promise.

## The technical crux

Standard TrueType/OpenType fonts are *closed contour*: every glyph is an
outline shape, which is filled when printed. A plotter/robot driving a real
pen needs a *centerline stroke path* — a true single-line font can't even
be represented as a valid TTF (TTF paths must close), so stroke fonts are
distributed as SVG fonts or "cheated" TTFs where the pen draws each line
twice. Vendors like Quantum Enterprises sell handwriting font packages
containing **both** a single-line SVG/TTF stroke version and a regular
outline version of the same typeface — exactly the dual format our tier
model needs — and they also build custom handwriting fonts from samples.

**Implication for our font pipeline:** when licensing the launch library,
prefer typefaces available (or commissionable) in stroke + outline pairs,
even though MVP only needs outline. Retrofit is possible but per-font work.

## Vendor notes

### Handwrytten (real pen, robots)
- Fleet of writing robots; API available (REST, documented publicly).
- Catalog of house handwriting styles; **custom font service**: they
  recreate a customer's handwriting from a sample (their team programs
  it). This proves per-style onboarding is a supported motion — the open
  question is cost/timeline per style at our scale (40–60 fonts).
- No documented support for uploading arbitrary TTFs.

### Simply Noted (real pen, robots)
- 220+ writing robots, real ballpoint, real indentation. REST API
  (`api.simplynoted.com`), plus Zapier/CSV triggers. Handwritten
  envelopes too.
- Claims a **900-style handwriting catalog** — large enough that
  "best match from our library to theirs" is a viable bridge strategy
  even without custom onboarding.
- Also offers custom handwriting recreation from a worksheet sample
  (AI-assisted, with letter-shape variation and ligatures).

### Thanks.io (digitally printed handwriting-style)
- **Not a pen service** — prints handwriting fonts with AI-added jitter
  (pen pressure, line angle). Cheap: postcards from ~$0.59, notecards
  from ~$1.79 including postage, on business plan. Full REST API.
- Fixed house style catalog addressed by style ID; no custom font upload
  documented. Useful as a budget printed-tier vendor, but rendering our
  own PDFs via Lob/PostGrid preserves font freedom — prefer that unless
  unit economics say otherwise.

### Lob / PostGrid (digital print-and-mail APIs)
- We submit fully-rendered artwork (PDF/HTML), so **our fonts, our
  layouts, our realism tricks** — no font constraint at all. Photo-front
  cards and inserts supportable. This is the MVP fulfillment lane.
- Bake-off criteria: notecard + envelope quality, folded card support,
  photo print quality, per-piece cost at volume, address printing on
  envelopes (handwriting font on the envelope is a believability lever),
  turnaround time, API ergonomics, webhook quality.

## Strategy

1. **MVP:** Lob-class digital print behind a `FulfillmentProvider`
   abstraction. We own the PDF renderer; the full font library works
   everywhere. Print-at-home PDF ships from the same renderer.
2. **In parallel (business development, not code):**
   - Ask Handwrytten + Simply Noted: can you onboard N of our licensed
     fonts as custom styles? Per-style cost, timeline, exclusivity, API
     addressing?
   - License-check the launch library for stroke-format availability.
3. **Post-MVP pen tier, in order of preference:**
   a. Pen vendor onboards our fonts → true unified library.
   b. Best-match mapping from our library to the vendor's catalog
      (Simply Noted's 900 styles make this credible) → "closest pen
      style" UX, clearly labeled.
   c. Long-term/if volume justifies: own plotter fleet (AxiDraw/NextDraw
      class) using our stroke fonts — maximum control, real ops burden.

## Answer to the founding question

Yes — relying on pen vendors' *stock catalogs* would significantly limit
the premium tier's font fidelity to our library. But it does **not** block
the tiered vision: dual-format font sourcing plus vendor custom-style
onboarding (a motion both major vendors already sell) preserves the "same
handwriting at every tier" promise. The MVP printed tiers are entirely
unconstrained, so nothing about the pen-tier question needs to be resolved
before building V1 — it only shapes which fonts we license.

## Sources

- Handwrytten custom fonts & API: handwrytten.com/features, /api-redoc
- Simply Noted API & catalog: simplynoted.com/pages/api-automation,
  simplynoted.com (900 styles, 220+ robots claims)
- Thanks.io pricing & styles: thanks.io/pricing,
  help.thanks.io/api-handwriting-style-id
- Stroke vs outline fonts for plotters: quantumenterprises.co.uk
  single-line font documentation; TypeDrawers discussion on single-stroke
  TTF impossibility
