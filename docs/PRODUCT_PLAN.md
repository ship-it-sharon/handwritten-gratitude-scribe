# Product Plan — Handwritten Gratitude Scribe

_Last updated: 2026-07-10. Status: pre-code planning._

## Vision

Take the burden out of thank-you notes without taking away the heart. The
user gets real, physical, personal-feeling thank-you cards — written in her
voice, in her handwriting (or the closest match) — generated, printed, and
mailed in minutes instead of evenings.

**The bar every feature must clear:** would the recipient believe she wrote
it herself? The product sells relief from a social obligation *without the
guilt of a shortcut*.

## Audience

Busy women facing a stack of expected thank-yous after a life event:
weddings, bridal and baby showers, birthdays, graduations, holidays.
Consumer, not business. She is time-poor, cares about how the card lands,
and lives on her phone.

## Core loop

1. **Collect recipients** — import, enter, or capture addresses.
2. **Capture the occasion** — speak it, jot it, or photograph the gift.
3. **Generate the note** — heartfelt, personalized, editable, in her voice.
4. **Choose the look** — handwriting style, card design, optional photo(s)
   and QR voice message.
5. **Send** — we print and mail to the declared addresses.

## Decisions made (2026-07-10)

| # | Decision | Choice |
|---|----------|--------|
| 1 | MVP fulfillment | **Printed** (digital print-and-mail API). Real-pen robotic writing is a post-MVP premium tier; investigate vendor font flexibility in parallel. |
| 2 | Platform order | **Web first**, but the mobile app is required for GTM — social-platform advertising leans mobile. |
| 3 | Handwriting at launch | **Robust font library + "best match"** from a photographed handwriting sample. Custom font-from-sample and ML synthesis come later. |
| 4 | Business model | **Pay-per-card** with volume discounts. Upsells: included photo prints, QR voice messages. |

## Output tiers (target state, post-MVP)

Same font library across all tiers wherever possible:

| Tier | What it is | Price posture |
|------|-----------|---------------|
| Print at home | We render a print-ready PDF; she prints, stuffs, stamps | Free (funnel / trial) |
| Printed typescript | Digitally printed card in a handwriting font, mailed for her | Core, most popular |
| Pen-written | Robot writes with a real ballpoint pen; ink, indentation | Premium, ~2x printed |

**Known constraint (researched 2026-07-10):** pen-writing robots cannot use
standard outline fonts (TTF/OTF); they require single-stroke ("open
contour") vector fonts. Vendor catalogs (Handwrytten, Simply Noted) are
style-limited to their own libraries unless a custom font is commissioned
per style. See `docs/FULFILLMENT_RESEARCH.md` for options that preserve a
unified library across tiers — the short version: source library fonts that
ship in *both* outline and single-stroke formats, and/or negotiate custom
stroke-font onboarding with a pen vendor.

## Features

### Recipients & addresses
- **Wedding platform imports first:** Zola, The Knot, Joy export guest
  lists with collected addresses (CSV at minimum; APIs where available).
- **Spreadsheet upload** (CSV / Excel / Google Sheets) with smart column
  mapping for messy real-world lists.
- **Phone contacts** via native picker (mobile app).
- **Google Contacts** import (web).
- **Paste/share a text message** — iOS and Android both prohibit reading
  SMS directly, so the flow is share-sheet or paste; an LLM extracts the
  name + address from the message text.
- **Photo OCR** — snap an envelope return address or gift tag.
- **Address-request links** — she texts a link; the recipient fills in
  their own address (also a viral loop).
- **Validation** — USPS/SmartyStreets-class verification on every address
  before an order is accepted.

### Occasion & gift capture ("gift log")
- **Speak:** short voice memo per gift → transcription → structured entry.
- **Jot:** quick notes per recipient, or one bulk brain-dump the app
  splits into per-recipient entries.
- **Photograph:** picture of the gift (or gift pile with tags) → vision
  model identifies items and pairs them with givers.
- **Live gift-opening mode (mobile, post-MVP):** snap + speak as gifts are
  opened; the thank-you queue builds itself in real time. This is the
  mobile app's hero feature.

### Note generation
- LLM-generated (Claude) from: occasion, gift, relationship, tone dial
  (warm / playful / formal), and optional samples of her past writing so
  the *voice* matches, not just the handwriting.
- **Batch variety:** across an 80-note wedding batch, no two guests who
  compare cards should find identical sentences. Structural variation is a
  hard requirement of the generator, not a nice-to-have.
- **Fast editing:** swipe through drafts, tap-to-regenerate a sentence,
  bulk approve. Target: 80 notes reviewed in ~15 minutes.
- Length always constrained to fit the chosen card layout.

### Handwriting
- **Launch:** 40–60 licensed handwriting fonts with programmatic realism
  (glyph alternates, baseline/rotation jitter) + **best-match**: user
  photographs a handwriting sample, we rank the library by similarity
  (slant, print-vs-cursive, letterforms, spacing).
- **Fast follow:** custom font from a guided sample sheet (Calligraphr-
  style), with 2–3 variants per letter via OpenType contextual alternates.
- **R&D track:** ML handwriting synthesis (One-DM / HiGAN+ class models)
  — high wow, high productionization risk. Never on the critical path.

### Card design & extras
- Curated card style families (cover art, layouts); occasion-appropriate.
- **Photo upsell:** photo-front cards and/or printed photo inserts from
  the event.
- **QR voice message upsell:** she records a spoken thank-you; we host it
  at an unlisted short URL and print a small, elegant QR on the card
  ("scan to hear a message from Sharon"). Frictionless playback page,
  optional expiry, tasteful presentation.

### Fulfillment
- **Envelope requirement (decided 2026-07-11):** the envelope must be
  addressed in the same handwriting style as the note. A window envelope
  or machine-format address block reads as junk mail and breaks the
  product promise. This rules out generic direct-mail APIs (Lob,
  PostGrid) for the card product — see `docs/FULFILLMENT_RESEARCH.md`.
- MVP: one handwriting-specialist print partner (Thanks.io-class) behind
  a vendor abstraction; print-ready assets rendered/parameterized by us
  where the vendor allows.
- Print-at-home PDF export as free tier and fallback.
- Post-MVP: pen-written premium tier via robotic vendor or (long-term)
  own plotter fleet. See `docs/FULFILLMENT_RESEARCH.md`.

## Pricing model

- **Pay-per-card.** Card price covers printing, envelope, postage.
- **Volume discounts** at natural batch sizes (e.g. 10 / 25 / 50 / 100+)
  — weddings are the volume case and the marquee use case.
- **Upsells per card:** photo print(s), QR voice message.
- **Free tier:** print-at-home PDF (watermark-free; the funnel is the
  product experience, not a crippled demo).
- No subscription at launch; revisit once repeat-occasion behavior is
  observed (holiday cards are the obvious subscription wedge later).

## Go-to-market notes

- Acquisition on social platforms (Instagram/TikTok/Pinterest), which
  lean mobile → the **mobile app is a GTM requirement**, even though the
  product builds web-first.
- Bridge while mobile ships: mobile-optimized web funnel so ad clicks can
  convert before the native app exists.
- Wedding season and registry ecosystems are the beachhead; showers and
  holidays follow.

## Phases

**V1 — the full loop, web only**
- Manual + CSV address entry, validation
- Occasion + typed/jotted notes in
- Generated notes with batch editing
- Font library (manual pick + best-match from uploaded sample photo)
- One card style family
- Mail via one print partner; print-at-home PDF
- Stripe pay-per-card checkout with volume discounts

**V2 — mobile + magic**
- Expo mobile app: camera, contact picker, voice capture, share-sheet
- Live gift-opening mode
- Voice occasion capture end-to-end
- QR voice message upsell
- Photo cards / photo inserts upsell
- Wedding-platform CSV importers (Zola / The Knot / Joy)

**V3 — differentiation**
- Custom handwriting font from guided sample sheet
- Pen-written premium tier (vendor integration per fulfillment research)
- Address-request links
- Deeper integrations (registry APIs, Google Contacts sync)
- ML handwriting synthesis R&D graduation, if quality clears the bar

## Open questions

- Which print partner wins on notecard quality + photo support + unit
  economics? (Narrowed 2026-07-11: Lob/PostGrid eliminated by the
  matching-envelope requirement; shortlist is Thanks.io vs Scribeless,
  with in-house micro-fulfillment as the long-term full-control path.
  Details in `docs/FULFILLMENT_RESEARCH.md`.)
- Pen-vendor font flexibility: will Handwrytten / Simply Noted onboard
  *our* fonts as custom styles, and at what per-font cost? (Outreach task.)
- Font licensing: which foundries allow commercial rendering-as-a-service
  use, and which offer stroke + outline dual formats?
- QR audio retention policy (forever? 1 year? user-controlled?).
- Postage authenticity: real stamps vs. printed indicia. Metered/indicia
  postage reads as business mail; pen-robot vendors use real stamps.
  Where does our printed-tier vendor land, and does it matter enough to
  drive vendor choice?
