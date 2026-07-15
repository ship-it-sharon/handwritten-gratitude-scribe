# Data Protection — Posy's non-negotiables

_Adopted 2026-07-11. This document is **binding**: when any product,
architecture, or vendor decision conflicts with it, this document wins or
the decision escalates to Sharon. It is written to be enforceable in
code review, not aspirational._

## Why this is existential, not compliance

Posy's entire premise is trust: a user hands over her relationship graph —
the names, home addresses, birthdays, gift histories, and private
sentiments of the people she loves. Most of those people **never signed
up for anything**. A single breach or creepy use of this data doesn't
just hurt the company; it makes the user complicit in exposing people
she was trying to honor. So the bar is set accordingly.

## The data we hold, ranked by sensitivity

| Class | Examples | Notes |
|---|---|---|
| **Third-party PII** | Recipients' names, home addresses, birthdays, households, relationships | The crown jewels. These people are not our users and cannot consent. |
| **Intimate content** | Message text, gift descriptions, voice recordings, event photos | Personal sentiment; sometimes reveals health, finances, family details. |
| **User biometric-adjacent** | Handwriting samples, writing-style samples, voice | Identifying by nature; used only for the feature the user invoked. |
| **User account** | Email, auth identity, order history | Standard, still protected. |
| **Payment** | Card details | Never touches our systems — Stripe-hosted fields only. |

## Principles (product ethos)

1. **Minimum data, stated purpose.** We collect only what a feature
   needs, use it only for that feature, and keep it only as long as
   needed. No speculative collection "because it might be useful."
2. **Never monetized as data.** No selling, no sharing for advertising,
   no lookalike audiences built from contacts, no enrichment of
   recipient profiles from outside sources. Revenue is cards, not data.
3. **Never used to train models.** Ours or anyone's. AI vendors must
   contractually not train on our traffic (Anthropic API does not train
   on API customer data; this stays a hard vendor requirement).
4. **Recipients are protected people, not leads.** We never contact a
   recipient except to deliver what the user sent. QR voice pages carry
   no trackers, no accounts, no upsells to the recipient. Address-request
   links collect the address and nothing else, and say plainly who is
   asking and why.
5. **Delete means delete.** Account deletion purges contacts, addresses,
   messages, media, and vendor-side copies where the vendor allows —
   within a stated window, verifiably. Per-contact and per-event
   deletion get the same treatment.
6. **Humans don't read user content.** No employee/support access to
   messages, contacts, or media without the user's explicit, per-incident
   consent. Debugging works from metadata.
7. **Plain-language transparency.** The privacy policy reads like the
   product speaks — a busy person can understand what we hold and why in
   two minutes.

## Technical commitments (binding on architecture & every PR)

- **Row-level security on every table, from the first migration.**
  Supabase RLS: a user's session can only ever read/write her own rows.
  No API path, admin panel, or serverless function bypasses RLS to read
  user content in bulk.
- **Encryption:** TLS everywhere in transit; encryption at rest via
  platform. Stored media (voice, photos, handwriting samples) accessible
  only via short-lived signed URLs — no public buckets, ever.
- **PII never appears in:** logs, error reports, analytics events, URLs,
  or email subject lines. Analytics are event-shaped ("order_placed,
  n_cards=42"), never content-shaped. This is checked in code review.
- **Data-diet for AI calls:** generation gets the minimum context —
  recipient first name, relationship, gift, occasion. Street addresses
  never go to the LLM; they aren't needed for a heartfelt note.
- **Vendors see data on a need-to-act basis:**
  | Vendor | Sees | When |
  |---|---|---|
  | Supabase | everything (it is the database) | always — hence RLS + platform vetting |
  | Vercel | transient request data | serving requests |
  | Anthropic | minimal generation context (above) | at generation |
  | Fulfillment (Thanks.io-class) | recipient name+address+message+artwork | at order placement only — no pre-syncing of address books |
  | Google Maps Platform | the address text typed into the autocomplete box | while the user types in the address form (no names attached, nothing stored there) |
  | Stripe | payment + billing identity | at checkout |
- **Vendor gate (before any integration):** DPA available · no training
  on our data · deletion honored (API or process) · SOC 2 / ISO 27001 or
  credible equivalent · breach-notification terms. A vendor that fails
  the gate is out, regardless of features. (Added to the fulfillment
  sales-call checklist.)
- **Retention defaults:** voice recordings and handwriting samples keep
  a user-visible expiry; order artifacts (rendered PDFs) purge after a
  bounded window post-delivery; backups are encrypted and age out on a
  fixed schedule that honors deletions.
- **Compliance posture:** build to CCPA/GDPR-grade norms from day one
  (export, deletion, purpose limitation) even before any legal threshold
  applies — retrofitting rights onto a data model is how companies end
  up unable to honor them.
- **Secrets hygiene:** per INFRASTRUCTURE.md — keys in platform env
  vaults, never in the repo; leaked key = immediate revoke/reissue.
- **Breach readiness:** a written 48-hour plan (contain → assess scope
  via audit logs → notify affected users plainly and promptly). Audit
  logging of admin/service access exists from V1 so scope assessment is
  possible at all.

## Process commitments

- **Every feature spec includes a "Data touched" section** (template
  updated in WORKING_TOGETHER.md): what new data, which class, why,
  retention, which vendors see it. No section, no build.
- **Every milestone gets a privacy pass** alongside functional review;
  security review runs on code changes touching auth, PII tables,
  storage, or vendor integrations.
- **New-vendor and new-data-class decisions always escalate to Sharon** —
  they are product decisions, not implementation details.

## What this rules out, permanently

- Selling, renting, or sharing contact data; ad-based monetization of any
  user or recipient data.
- Marketing to recipients, or using one user's recipients to seed
  another's suggestions.
- Building shadow profiles of recipients from external sources.
- "Anonymized" contact-graph analytics sold or shared externally.
- Ambient social-app ingestion of any kind (decided 2026-07-11): Posy
  never watches feeds or monitors recipients. Social context enters only
  as individual posts/screenshots the user explicitly shares into Posy,
  processed for that one message and nothing else.
