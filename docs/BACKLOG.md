# Backlog

_Things we've explicitly deferred, so they don't evaporate. Sharon adds,
reprioritizes, and strikes items; Claude adds items when a decision
defers something._

## UX debt

- **M1 recipient flows — full UX pass** (noted 2026-07-14, Sharon):
  functionality confirmed working; interaction polish deliberately
  deferred. Candidates when we get there: inline editing, undo on
  remove, add-another-without-scroll, empty states, loading states,
  mobile ergonomics review.
- Event create form: date field UX, occasion iconography.

## Engineering debt

- Server-side idempotency for create actions (submission tokens): the
  pending-disabled buttons stop double-taps, but a retried request
  (flaky network, browser re-POST) could still duplicate. Worth doing
  before launch; not urgent while testing. (2026-07-14)

## Import sources roadmap (noted 2026-07-15, Sharon)

Shipped: manual, household, CSV (with template + auto-mapping),
add-from-address-book. Next, in rough priority order:

1. **Google Contacts** (People API, OAuth `contacts.readonly`) — the big
   one. Caveat: it's a Google "sensitive scope," so production use
   requires app verification (weeks of lead time) — start that clock
   before building the feature.
2. **vCard (.vcf) upload** — how Apple/iCloud and Outlook contacts
   export; one file format covers the whole Apple ecosystem without any
   API partnership.
3. **Excel (.xlsx) upload** — same import flow as CSV, one parsing
   library away; removes the "download as CSV" step.
4. **Wedding-platform mapping presets** — Zola / The Knot / Joy exports
   already work via CSV; presets would recognize their exact column
   headers for zero-fiddle imports.
5. **Paste-a-text/email parsing** (LLM extraction) — already in the V1
   plan, lands with M2+ once the Anthropic key exists.
6. **Phone contacts** — V2 mobile app, native picker.
7. Outlook/Microsoft Graph — only if user demand shows up.

## Product parking lot

- Kids'-party recipient case (thank-you to the parent, names the kid) —
  V1 or V1.1 call still open in V1_FLOWS S4.
- Guest mode (account-optional first card) — post-launch conversion
  experiment.
- Migration automation: GitHub Action applies Supabase migrations on
  merge (~15 min setup + one access token from Sharon) — adopt when
  manual SQL pasting gets old.

## Awaiting Sharon (external)

- PostHog verdict (analytics/flags/experiments single-vendor proposal).
- Thanks.io account + physical card samples order (long lead time —
  affects M6 vendor decision).
- Vendor sales calls: Thanks.io / Scribeless question list in
  FULFILLMENT_RESEARCH.md.
