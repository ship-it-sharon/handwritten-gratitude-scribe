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
| Address validation | Via fulfillment vendor or **Smarty** free tier (later) | Hard gate before orders | $0 early |

Realistic total: **≈ $1–2/mo while building** (domain amortized), rising
toward **~$70/mo + per-card costs** only when there are real users.

## Account setup order (each ~10 minutes, Claude walks through each live)

1. **Domain name** — blocked on branding: the product needs a name before
   buying a domain. (Decide name → check availability → buy.)
2. **Vercel** — sign up with GitHub; connect this repo; previews start
   working immediately. Needed at milestone M0.
3. **Supabase** — create org + project; grab API keys. Needed at M0.
4. **Anthropic API** — create key, set a monthly spend cap. Needed when
   note generation is built (M2).
5. **Stripe** — business details required (can start in test mode with
   nothing real). Needed at checkout milestone (M5); test mode from day 1.
6. **Thanks.io** — account + API key; also the account through which we
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
