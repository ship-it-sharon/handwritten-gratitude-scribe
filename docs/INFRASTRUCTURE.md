# Infrastructure Plan — from zero

_Written 2026-07-11. Status: nothing procured yet. This is the shopping
list and the order to buy it in._

## Principles

- **Boring, managed, cheap-until-traction.** Every service below has a
  free or near-free tier that covers build + early users; nothing needs
  ops attention.
- **Sharon owns every account** — created with her email, her billing.
  API keys get shared into the project's secret storage, never committed
  to the repo.

## The stack (and monthly cost trajectory)

| Layer | Service | Why | Cost now → at traction |
|---|---|---|---|
| Web hosting + deploys | **Vercel** | Every git push gets a live preview URL (the PM review loop depends on this); zero-config for React | $0 → $20/mo |
| Database + auth + file storage | **Supabase** | Postgres, sign-in (email + Google), storage for photos/audio/samples, serverless functions — one account, one bill | $0 → $25/mo |
| Domain | **Cloudflare Registrar** (or Namecheap) | At-cost pricing, solid DNS | ~$10–15/yr |
| Payments | **Stripe** | Pay-per-card checkout, volume discounts, upsell line items | $0 + ~2.9% + 30¢ per transaction |
| AI (note generation, matching, parsing) | **Anthropic API** | Claude for generation/vision/extraction | Pay-per-use; pennies per note batch while testing |
| Print fulfillment | **Thanks.io** | Per FULFILLMENT_RESEARCH.md | Pay-per-card (~$1.79+ per mailed notecard) |
| Transactional email | **Resend** (later) | Receipts, order updates | $0 → ~$20/mo |
| Analytics + flags + experiments | **PostHog** (proposed over Amplitude + LaunchDarkly) | One vendor covering product analytics, feature flags, and A/B tests; generous free tier; one fewer PII processor | $0 → usage-based |
| Company email / identity | **Google Workspace** on the Posy domain | All vendor accounts created under @posy identity; professional footing for BD | ~$7/mo (1 seat) |
| Address validation | Via fulfillment vendor or **Smarty** free tier (later) | Hard gate before orders | $0 early |

Realistic total: **≈ $1–2/mo while building** (domain amortized), rising
toward **~$70/mo + per-card costs** only when there are real users.

## Account setup order (each ~10 minutes, Claude walks through each live)

_Sequence decided 2026-07-13: domain → Google Workspace → everything
else under the @posy identity._

1. **Domain** — ✅ DONE (2026-07-13): **sendposy.com is the primary**
   (site serves at both sendposy.com and www.sendposy.com); Sharon also
   holds related domains, which will 301-redirect to the primary once
   DNS is set up.
2. **Google Workspace** (1 seat, ~$7/mo) on sendposy.com →
   **sharon@sendposy.com** becomes the identity every other account is
   created with. MX records go into whichever DNS host serves
   sendposy.com (guided setup during Workspace signup).
3. **Vercel** — sign up with the Workspace Google account; connect this
   repo; previews start working immediately. Needed at milestone M0.
4. **Supabase** — create org + project; grab API keys. Needed at M0.
5. **PostHog** — analytics/flags/experiments; wire in during M0 so
   funnel data exists from the first preview.
6. **Anthropic API** — create key, set a monthly spend cap. Needed when
   note generation is built (M2).
7. **Stripe** — business details required (can start in test mode with
   nothing real). Needed at checkout milestone (M5); test mode from day 1.
8. **Thanks.io** — account + API key; also the account through which we
   evaluate card stock samples. Needed at fulfillment milestone (M6);
   order physical samples EARLY (week 1–2) since stock quality could
   change the vendor decision.

## Secrets handling

API keys live in Vercel/Supabase environment settings (encrypted), never
in the repo. A `.env.example` file lists what's needed without values.
If a key ever lands in chat or a commit by accident: revoke and reissue —
keys are free to replace.

## Environments

- **Production** — the real site at the real domain.
- **Preview** — automatic per-branch deploys (Vercel) with a test
  database, fake payments (Stripe test mode), and fulfillment in
  sandbox/draft mode so no card ever accidentally mails during review.

## Deliberately not procured

- No servers, containers, or cloud consoles (AWS/GCP) — nothing here
  needs them.
- No email marketing / analytics / CRM yet — GTM tooling waits until
  there's something to market.
- App Store / Play Store developer accounts — V2, with the Expo app.
