# Architecture — Posy

_Last updated: 2026-07-11. Status: planned, pre-code._

## North-star constraints (from PRODUCT_PLAN "Posy" section)

V1 builds thank-you notes only, but the data model is shaped so the
occasion-reminder/gifting future is additive, not a migration:

- **Contacts are the durable center**, not events. Everything sent to a
  person links back to them; contacts carry important dates.
- **Events are occasion instances** ("Our Wedding", "Christmas 2027",
  "Jane's Birthday") — some past/reactive, some future/recurring. V1
  only creates past-style events; the shape allows future-dated ones.
- **Received vs sent gifts are different entities** — named as such from
  day one.
- **Proactive entry points will exist later** (reminders). V1 builds
  nothing proactive, but transactional email (order updates) establishes
  the outbound channel plumbing.
- **Accessible/tactile formats will exist later** (enlarged print →
  raised print → braille; textured premium designs). Three shapes
  follow now: format is a **per-card** property (never per-event), a
  card's text-length budget derives from its format spec, and
  fulfillment routing is capability-based because one order may split
  across vendors (79 standard cards + 1 braille card).

## Shape of the system

Monorepo, TypeScript throughout:

```
apps/
  web/        React (Vite) — primary product surface, batch flows
  mobile/     React Native + Expo — camera, contacts, voice, share-sheet
packages/
  shared/     Domain types, API client, validation, note/card models
  renderer/   Card layout engine → print-ready PDF (bleed, DPI, vendor specs)
backend/
  supabase/   Postgres, Auth, Storage, Edge Functions
```

- **Web first** (V1 is web-only), but `shared/` is written from day one so
  the Expo app (V2) reuses domain logic and the API client.
- **Supabase** for auth, Postgres, file storage (handwriting samples, gift
  photos, audio), and edge functions for server-side work. Familiar stack,
  fast to ship, row-level security for PII.

## Services & responsibilities

| Concern | Approach |
|---|---|
| Note generation | Claude API. Inputs: occasion, gift, relationship, tone, optional writing samples. Batch mode enforces structural variation across a set. |
| Voice capture | Transcription API (server-side) → structured gift-log entry via LLM extraction. |
| Gift photos | Vision (Claude) → item identification, giver pairing from tags. |
| Address parsing | LLM extraction from pasted texts/emails; OCR for envelope/gift-tag photos. |
| Address validation | USPS/SmartyStreets-class verifier; hard gate before order acceptance. |
| Handwriting match | Vision analysis of a sample photo (slant, cursive-ness, letterforms, spacing) → ranked matches against the font library's feature index. |
| Rendering | Own PDF renderer (`packages/renderer`). Fonts + realism layer (contextual alternates, jitter). One renderer feeds print-at-home, print vendor, and previews. |
| Fulfillment | `FulfillmentProvider` interface with **capability flags** (folded card, photo front, inserts, QR, envelope-in-font, raised print, braille, pen-written). Cards route to a vendor by required capabilities, so one order can split across vendors. First adapter = handwriting-specialist digital printer. Webhooks → order status timeline. |
| QR voice | Audio upload → storage → unlisted short URL + playback page → QR embedded by renderer. |
| Payments | Stripe. Pay-per-card checkout, volume discount tiers, upsell line items (photos, QR). |

## Data model (first cut, Posy-shaped)

- `users`
- `contacts` — name, household grouping, relationship; **the durable
  relationship graph**. V1 populates from imports/manual entry; later
  features hang off this table rather than off events. Includes a
  nullable `format_preferences` field (e.g. large-print, braille) —
  empty in V1; set once, honored at every future occasion.
- `contact_addresses` — validated + raw address, source (csv | manual |
  contacts | parsed-text | ocr | request-link), current flag. Separate
  table: addresses change over time and Posy's long game spans years.
- `contact_dates` — birthday, anniversary, custom dates per contact.
  Empty in V1 (CSV import may populate it opportunistically); powers
  future reminders. Costs one table now, saves a migration later.
- `events` — occasion type (wedding | shower | birthday | holiday | …),
  event date, **temporal mode (past-reactive in V1; schema permits
  future-dated)**, optional recurrence hint (null in V1)
- `gifts_received` — event, giver(contact), description, photo(s),
  capture source. Named for direction; `gifts_sent` arrives with the
  gifting flows, as a sibling not a retrofit.
- `messages` — per contact+event: generated text, edit history, status
  (draft | approved), tone settings. (Named `messages`, not `notes` —
  thank-you notes now, holiday/birthday messages later.)
- `cards` — message + design + font + extras (photo ids, audio id) +
  **format spec** (size, print mode: standard | large-print | raised |
  braille; standard-only in V1) + component list (front / interior /
  inserts), rendered artifact ref. Format lives per-card so one event
  can mix formats; the message's character budget is a function of
  the format spec, enforced back in the studio at edit time.
- `designs` — card design catalog with attribute tags (occasion,
  style family, premium, textured/raised, high-contrast,
  large-print-friendly) so recommendation is tag-driven, not a
  segregated accessibility catalog.
- `send_history` — denormalized per-contact record of everything ever
  sent (card, date, occasion, message text ref). Powers "what did I say
  last year" in generation and the contact timeline later. Written from
  day one, cheap.
- `orders` / `order_items` — fulfillment vendor refs, status timeline,
  pricing snapshot
- `fonts` — library metadata, feature vector (for matching), license info,
  formats available (outline / stroke)
- `handwriting_samples` — user uploads, extracted feature vector
- `audio_messages` — storage ref, short URL slug, expiry policy

## Cross-cutting concerns

- **PII / data protection:** governed by `docs/DATA_PROTECTION.md`
  (binding). Architectural consequences: RLS on every table from the
  first migration with no bypass paths; signed short-lived URLs for all
  stored media; PII banned from logs/analytics/URLs; data-diet on AI
  calls (no street addresses to the LLM); vendors receive data only at
  the moment of action (no address-book pre-syncing to fulfillment);
  audit logging of admin/service access from V1; retention/expiry
  fields on media from the first schema; delete-account really deletes,
  including vendor-side purge calls.
- **Rendering correctness:** the PDF is the product. Golden-file tests on
  renderer output; vendor spec conformance (bleed, safe area, DPI) tested
  per adapter.
- **Generation pacing (per PRODUCT_PLAN case-by-case decision):** no
  batch pre-generation. Cards generate on demand in the studio flow;
  at most ONE next-card draft may be prefetched for snappiness, and a
  prefetched draft is discarded unseen when settings change. Sign-off
  is a recorded state transition (timestamp) on the message, and the
  order pipeline hard-gates on it: nothing renders to print without a
  signed-off message.
- **Vendor abstraction:** no fulfillment vendor types outside the adapter.
  Adding pen-written tier later = new adapter + tier metadata, no product
  rewrite.

## Deliberately deferred

- ML handwriting synthesis (One-DM class) — R&D track, isolated service if
  it ever graduates; never in the V1 critical path.
- Own plotter fleet for the pen tier.
- Registry/wedding-platform API integrations (V1 uses their CSV exports).
- Subscriptions (model is pay-per-card at launch).
- **Reminder/notification engine** (scheduled jobs + push/email) — the
  Posy proactive layer. V1's only outbound sends are transactional order
  emails, which establish the email plumbing the reminder engine will
  reuse.
- **Social-app context ingestion** for message generation — opt-in,
  privacy-heavy, post-north-star-validation. Generation inputs stay
  pluggable (a context object, not hardcoded fields) so this slots in.
- **Gift recommendation/procurement** (gift cards, experiences) — new
  vertical; `gifts_sent` + a `GiftProvider` abstraction when it comes.
- **Accessible formats** (enlarged print → raised print → braille) and
  tactile premium designs — the schema hooks above (per-card format
  spec, contact format preferences, design tags, capability routing)
  exist from V1; the renderers, transcription (UEB), and specialty
  vendors arrive with each release.
