# Architecture — Handwritten Gratitude Scribe

_Last updated: 2026-07-10. Status: planned, pre-code._

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
| Fulfillment | `FulfillmentProvider` interface; first adapter = digital print-and-mail vendor (Lob-class). Webhooks → order status timeline. |
| QR voice | Audio upload → storage → unlisted short URL + playback page → QR embedded by renderer. |
| Payments | Stripe. Pay-per-card checkout, volume discount tiers, upsell line items (photos, QR). |

## Data model (first cut)

- `users`
- `contacts` — name, address (validated + raw), source (csv | manual |
  contacts | parsed-text | ocr | request-link), relationship
- `events` — occasion (wedding | shower | birthday | …), date
- `gifts` — event, giver(contact), description, photo(s), capture source
- `notes` — gift/contact, generated text, edit history, status
  (draft | approved), tone settings
- `cards` — note + design + font + extras (photo ids, audio id), rendered
  artifact ref
- `orders` / `order_items` — fulfillment vendor refs, status timeline,
  pricing snapshot
- `fonts` — library metadata, feature vector (for matching), license info,
  formats available (outline / stroke)
- `handwriting_samples` — user uploads, extracted feature vector
- `audio_messages` — storage ref, short URL slug, expiry policy

## Cross-cutting concerns

- **PII:** addresses, voice recordings, photos of homes/gifts. Row-level
  security everywhere; encryption at rest via platform; strict retention
  policy for audio and samples; delete-account really deletes.
- **Rendering correctness:** the PDF is the product. Golden-file tests on
  renderer output; vendor spec conformance (bleed, safe area, DPI) tested
  per adapter.
- **Batch UX performance:** generation for 80+ notes runs as a queued job
  with streaming progress, not a single blocking request.
- **Vendor abstraction:** no fulfillment vendor types outside the adapter.
  Adding pen-written tier later = new adapter + tier metadata, no product
  rewrite.

## Deliberately deferred

- ML handwriting synthesis (One-DM class) — R&D track, isolated service if
  it ever graduates; never in the V1 critical path.
- Own plotter fleet for the pen tier.
- Registry/wedding-platform API integrations (V1 uses their CSV exports).
- Subscriptions (model is pay-per-card at launch).
