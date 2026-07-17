# Posy

_Repository name predates the product name — this is **Posy**._

**The wedge:** take the burden out of thank-you notes without taking away
the heart. Heartfelt, personalized notes in the user's own handwriting
style (or the closest match), printed and mailed for her — for weddings,
showers, birthdays, and every occasion where gratitude is expected but
time is short.

**The north star:** a relationship-tending app. Posy reminds you when the
people you love have occasions coming (Christmas, Jane's birthday),
helps you say something that sounds like you, and sends the card — and
eventually the gift — for you.

## Docs

- [Product plan](docs/PRODUCT_PLAN.md) — vision, north star, decisions, features, tiers, phases
- [Data protection](docs/DATA_PROTECTION.md) — **binding** privacy non-negotiables; wins over any conflicting decision
- [V1 flows](docs/specs/V1_FLOWS.md) — screen-by-screen sketch + build milestones (PM markup wanted)
- [Architecture](docs/ARCHITECTURE.md) — planned system shape (pre-code)
- [Fulfillment research](docs/FULFILLMENT_RESEARCH.md) — print vs. real-pen vendors, font flexibility, envelope requirement
- [Working together](docs/WORKING_TOGETHER.md) — PM ↔ Claude Code workflow
- [Design system](docs/DESIGN_SYSTEM.md) — Radix Themes, Posy tokens, usage rules
- [Expenses](docs/EXPENSES.md) — running tally of what Posy costs
- [Backlog](docs/BACKLOG.md) — deferred work, parked decisions, import-sources roadmap
- [Infrastructure](docs/INFRASTRUCTURE.md) — hosting/domain/services shopping list, costs, setup order

## Code

- `apps/web` — the Next.js app (Radix Themes UI), deployed on Vercel
- `packages/shared` — domain types shared with the future mobile app
- `supabase/migrations` — the database, RLS-first

Docs before code, on purpose — the planning docs above are the source
of truth for what gets built and why.
