# Expenses — running tally

_Every real dollar Posy costs, so nothing sneaks up on us. Sharon owns
the accounts and the actuals; Claude keeps structure and pricing notes
current. Update when a new service is added or a price changes._

## Recurring

| Service | Plan | Cost | Started | Notes |
|---|---|---|---|---|
| Google Workspace | Business Starter, 1 seat | ~$7/mo | 2026-07 | sharon@sendposy.com identity |
| Vercel | Hobby | $0 | 2026-07 | Upgrade to Pro ($20/mo) at launch or when limits bite |
| Supabase | Free | $0 | 2026-07 | Upgrade to Pro ($25/mo) before launch (backups, no project pausing) |
| Google Cloud (OAuth, Places) | Pay-as-you-go | $0 so far | 2026-07 | See usage-based below |

## One-time / annual

| Item | Cost | When | Notes |
|---|---|---|---|
| Domains: sendposy.com + related | ~$10–15/yr each — Sharon to fill actuals | 2026-07 | Primary + defensive registrations |

## Usage-based (free until real traffic)

| Service | Free tier | Then | Current expectation |
|---|---|---|---|
| Google Places API (New) — autocomplete | 10K requests/mo | $2.80 per 1K | $0 — testing volume is nowhere near 10K |
| Anthropic API | none (pay per token) | ~pennies per note batch | Not yet created (M2) |
| Stripe | no fee until transactions | 2.9% + 30¢ per charge | Test mode only until M5 |
| Thanks.io | pay per card | ~$1.79+/notecard mailed | Account not yet created; samples order upcoming |
| Resend (email) | 3K emails/mo free | ~$20/mo | Not yet needed (Supabase built-in mailer during dev) |
| PostHog (if adopted) | ~1M events/mo free | usage-based | Pending Sharon's verdict |

## Monthly picture

- **Today:** ~$7/mo (Workspace) + domains amortized ≈ **$9–10/mo**
- **At launch (projected):** Workspace $7 + Vercel Pro $20 + Supabase Pro
  $25 + Resend ~$20 ≈ **$72/mo** fixed, plus per-card/per-token/per-charge
  variable costs that scale with actual orders (i.e., with revenue).

## Watchpoints

- Places autocomplete pricing: with sessions/token batching, 10K free
  requests ≈ thousands of address entries/mo — fine until real scale;
  revisit if an import feature ever fires autocomplete programmatically
  (it shouldn't).
- Google Cloud billing alerts: set a budget alert (e.g. $10/mo) in
  Billing → Budgets & alerts as a tripwire against surprises.
- Supabase free tier pauses projects after ~1 week of inactivity — if
  the app ever seems dead after a quiet stretch, check the dashboard
  before debugging.
