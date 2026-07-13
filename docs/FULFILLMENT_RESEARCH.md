# Fulfillment & Font Flexibility Research

_Researched 2026-07-10, updated 2026-07-11 with the matching-envelope
requirement. Questions: (1) can we offer the SAME font library across all
three tiers (print-at-home / printed / pen-written)? (2) which printed-tier
vendor can address the envelope in the same handwriting style as the note?_

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

### Lob / PostGrid (digital print-and-mail APIs) — ELIMINATED for the card product
- We submit fully-rendered artwork (PDF/HTML), so full font freedom **on
  the piece itself** — but the piece shapes are wrong. Product lines are
  postcards, letters, self-mailers, checks: business direct mail.
- **Envelope findings (the disqualifier):** Lob letters ship in #10
  window envelopes (address shows through a window, printed on the
  letter) or via a custom-envelope program that is enterprise-gated,
  uses pre-printed envelope stock (6-month adhesive expiry, batch
  ordered) — not per-piece envelope artwork, and not recipient
  addressing in an arbitrary handwriting font. PostGrid's template
  flexibility applies to the letter, not the envelope; addressing is
  standardized for USPS automation. A window envelope on a wedding
  thank-you reads as junk mail and gets tossed.
- Keep on file as infrastructure for any future *letter-shaped* product,
  and as a reference for API/webhook design quality.

### Scribeless (AI-handwriting digital print)
- Handwritten-style notes, cards, letters, postcards via CSV, CRM
  integrations, or API. US/UK/Canada production. Premium stock.
- Marketing mentions images, **trackable QR codes**, and text in
  **custom brand fonts** — the most font-flexible claim of any digital
  vendor found. Whether "custom brand fonts" extends to the handwriting
  itself and the envelope needs a sales call to confirm.
- Business-oriented; per-piece pricing at consumer scale TBD.

### Print.one (folded greeting cards API)
- Square folded cards with matching envelopes — right form factor — but
  envelopes use a round address **window**, which fails the matching-
  envelope requirement. EU-centric. Pass, but useful as a format
  reference.

### IgnitePost (real pen, robots)
- 5x7 folded greeting cards on 100 lb stock by default, 12+ handwriting
  styles or custom mimic, real pen and ink, API available. Smaller than
  Handwrytten/Simply Noted; potentially more flexible to partner with.
  Candidate for the pen tier alongside the big two.

## Envelope analysis (added 2026-07-11)

Requirement: recipient (and return) address on the envelope in the same
handwriting style as the note interior.

- **Generic direct-mail APIs cannot do this** — their envelope is a
  standardized carrier optimized for USPS automation discounts, not a
  design surface. This is structural, not a missing feature: their unit
  economics depend on machine-readable address blocks.
- **Handwriting-specialist vendors do this natively** — the addressed
  envelope in the matching style IS their product (Thanks.io notecards,
  Handwrytten/Simply Noted/IgnitePost pen-written envelopes). Simply
  Noted sells handwritten envelopes as a standalone service.
- **USPS deliverability note:** handwriting-font addresses are fine
  (USPS OCRs real handwriting all day), but may forgo presort automation
  discounts — priced into specialist vendors' per-piece rates.
- **Postage authenticity is the sibling issue:** printed indicia/meter
  marks read as corporate; real stamps read as personal. Pen vendors use
  real stamps. Ask every printed-tier candidate what postage looks like.
- **Return address** should render in the same style (or a clean serif —
  test both); also doubles as the QR/voice-message brand moment on the
  flap if we want it.

## Strategy

1. **MVP:** a handwriting-specialist digital printer (shortlist:
   Thanks.io, Scribeless) behind a `FulfillmentProvider` abstraction.
   Accept that the *mailed* tiers launch constrained to the vendor's
   style catalog; our full font library still powers print-at-home and
   previews, with best-match mapping onto the vendor's catalog for
   mailed cards. Validate demand before owning more of the stack.
2. **In parallel (business development, not code):**
   - Sales calls with Thanks.io + Scribeless: custom font onboarding?
     QR placement control? photo-front + photo-insert support? postage
     type (stamp vs indicia)? envelope return-address control? white-
     label packaging? consumer-volume pricing tiers?
   - **Data-protection gate (per DATA_PROTECTION.md, disqualifying):**
     DPA available? What happens to recipient addresses after mailing —
     retention period, deletion on request (API or process)? Any use of
     our recipient data for their marketing/analytics? SOC 2 / ISO 27001
     or equivalent? Breach-notification terms? We send order data at
     placement time only — no address-book syncing.
   - Ask Handwrytten + Simply Noted + IgnitePost: can you onboard N of
     our licensed fonts as custom styles? Per-style cost, timeline,
     exclusivity, API addressing? (Feeds the pen tier AND pressure-tests
     the custom-onboarding market.)
   - License-check the launch library for stroke-format availability.
3. **Post-MVP pen tier, in order of preference:**
   a. Pen vendor onboards our fonts → true unified library.
   b. Best-match mapping from our library to the vendor's catalog
      (Simply Noted's 900 styles make this credible) → "closest pen
      style" UX, clearly labeled.
   c. Long-term/if volume justifies: own plotter fleet (AxiDraw/NextDraw
      class) using our stroke fonts — maximum control, real ops burden.
4. **Long-term full-control option: in-house micro-fulfillment.**
   Digital press + envelope printer + real stamps (Postable's original
   model). Unlocks the entire font library on card AND envelope, real
   stamps, premium stock, photo inserts — at the cost of becoming an
   operations company. Revisit when volume and margin data exist.

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
