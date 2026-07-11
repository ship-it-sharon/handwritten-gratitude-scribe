# V1 Flows — screen-by-screen sketch

_Draft 2026-07-11, written by Claude for Sharon to mark up. Every screen
has **PM questions** — answer inline, strike out what's wrong, add what's
missing. This document becomes the V1 spec._

## Platform decision

**V1 is a mobile-first responsive web app with PWA installability.**
Designed for a phone screen first (ad traffic is mobile), comfortable on
desktop (batch editing is desk work). PWA features — installable icon,
app-like feel — come nearly free and help the ad→use funnel. What a PWA
can NOT deliver (iPhone especially): native contact picker, share-sheet
import, reliable camera/voice ergonomics. Those stay in V2's Expo app as
planned; the PWA does not replace it.

## The spine: one "Event" from start to mailbox

Everything hangs off an **Event** ("Sarah & Tom's Wedding", "Baby shower
for Emma"). The user's journey:

```
Landing → Sign up → Create event → Add recipients (1 or 80, her call)
→ Card studio (per-card: context → generate → shape → sign off)
→ Design card → Checkout (address deadline) → Track orders
```

She can leave and return at any step; the event remembers where she was.

## FTU principle: first card before first hurdle (decided 2026-07-11)

The first-time user must reach the payoff — a finished card, in her
handwriting, for a real person — **within minutes**, before any bulk
work is asked of her. Two design rules make this possible:

1. **A name is enough to start a card.** Recipient entry is decoupled
   from address entry: the studio needs "Aunt Carol — KitchenAid," not
   a validated street address. **Addresses are only required at
   checkout** — that's the address deadline, not the front door.
2. **Recipient entry is optional-bulk, never forced-bulk.** She can
   upload the whole CSV up front (power move), select existing contacts
   (returning user — trivial), or add exactly one person and go straight
   to the studio. The event's recipient list is alive: add, import, or
   remove people at any time, and the studio queue grows to match.

The FTU golden path: create event → type one name + gift → studio →
signed-off card in her handwriting ≈ 5 minutes. THEN "add everyone
else" (CSV or manual) lands as a natural next step with the payoff
already proven — not as a wall in front of it.

North-star FTU metric: **time to first signed-off card.**

---

### S1. Landing page
**Job:** convince a skeptical, busy woman this is legitimate relief, not
a tacky shortcut. Show a real-looking card immediately.
- Hero: finished card + envelope (matching handwriting) photo/render
- How it works in 3 steps; pricing transparency (per-card, volume tiers)
- Sample gallery: occasion-specific examples she can flip through
- **PM questions:** brand voice? Does pricing show on landing or after
  value demo? Social proof strategy at launch (no users yet)?

### S2. Sign up / sign in
**Job:** zero-friction entry. Email magic link + "Continue with Google".
No passwords. Sign-up can wait until she tries to *save* work (guest
mode through the first generation = stronger hook) — or gate everything.
- **PM question:** guest-first or account-first? (Guest-first converts
  better but complicates "come back later"; my lean is account-first for
  V1 simplicity, guest-first as a post-launch experiment.)

### S3. Create event
**Job:** name the occasion, set the context the generator will use.
- Occasion type (wedding / bridal shower / baby shower / birthday /
  graduation / holiday / other), event date, her role (bride, mom, …)
- This context flows into every note; she never re-explains
- **PM question:** anything else worth capturing once per event? (Venue?
  "we/I" voice? Couple's names for weddings?)

### S4. Recipients (revised 2026-07-11 — non-blocking)
**Job:** get people in with minimum typing, never as a wall. Per the
FTU principle: a name is enough; addresses are due at checkout.
- Entry modes, all optional, all repeatable mid-event:
  - **Quick add:** name (+ gift, optionally) — enough to start a card
  - **CSV/spreadsheet upload** with smart column mapping (LLM maps
    "Auntie's addr" → fields); addresses come along when present
  - **Select from address book** — returning users pick from existing
    contacts in seconds (the baby shower reuses the wedding list)
- Per-recipient completeness chip: ✍️ ready to write / 📮 needs address /
  ✓ address valid / ⚠ check this. Address gaps surface passively here
  and hard-gate only at checkout.
- "Fill in addresses" is its own focused pass she can do in one sitting
  (or fix one-offs at checkout) — deliberately separate from the
  emotional card-writing work.
- Recipients belong to her address book; an event *selects* from it
- **PM questions:** household handling ("Mr. & Mrs. Chen" vs two
  people)? Kids' parties where the thank-you goes to the parent but
  names the kid? Required now or V1.1?

### S5. Gifts & notes-to-self
**Job:** capture what each person gave + anything personal, fast.
- Per-recipient: gift description, relationship, optional freeform
  detail ("came from Ohio", "third shower she's thrown me")
- **Bulk brain-dump:** one big text box ("Aunt Carol - KitchenAid;
  Jess - the blanket, she knitted it herself…") → LLM splits it into
  per-recipient entries she confirms
- Voice/photo capture: V2 (mobile). Typed/pasted only in V1.
- **PM question:** is "no gift, just showed up / helped" a first-class
  case? (I think yes — "thank you for celebrating with us" notes.)

### S6. Tone & voice
**Job:** make the notes sound like her.
- Tone dial: warm / playful / formal (+ brief preview of each in situ)
- Optional: paste 2–3 past notes or texts she's written → generator
  mimics her phrasing
- Sign-off setting ("Love,", "xo", "With gratitude,") + name(s)
- **PM question:** per-event or per-recipient-group tone? (Grandma gets
  formal, college friends get playful — V1 or V1.1?)

### S7. The card studio — THE screen (revised 2026-07-11)
**Job:** one card at a time, with the user present at each card's
creation. She co-writes every card; nothing is mass-produced.

**Decision (2026-07-11):** no mass auto-generation. Cards are generated
**case-by-case**, in a per-card flow, and **every card requires an
explicit sign-off** before it can be printed. There is no
"approve all" button — sign-off is individual, always. (This answers
the earlier PM question: approve-all is out.)

Per-card flow:
1. Card opens with the recipient's context (gift, relationship, her
   notes-to-self) visible — she's *thinking about this person* now
2. She can add a last thought before generating (optional, one line or
   voice-to-text later)
3. Draft appears, rendered in her handwriting style
4. She shapes it: **Edit** (inline) / **Redo** (with an optional steer
   like "mention the beach") / adjust
5. **Sign off** (primary action) — recorded per card with timestamp;
   this is the gate to print. Then the next card slides in.
- Progress: "34 of 80 signed off"; fully resumable across sessions —
  a wedding batch is expected to span evenings, and returning mid-batch
  must feel seamless
- Variety guarantee still holds across the event: generation remembers
  what's been used so no two cards share openers/closers
- Performance nuance (invisible to her): the *next* card's draft may be
  quietly prefetched while she reviews the current one, so the flow
  never waits — but never more than one ahead, and a prefetched draft
  is discarded unseen if she changes tone/settings. The experience is
  one-at-a-time; the prefetch is only there so it's a *snappy*
  one-at-a-time.
- **PM questions:** should her edits teach the generator mid-batch
  (card 30 starts sounding more like her card-5 edits)? Target pace:
  is ~45–90 seconds per card the right feel — present but not slow?

### S8. Handwriting
**Job:** "that looks like MY writing" (or at least: unmistakably human).
- Gallery of styles rendered live with HER actual note text, not lorem
  ipsum
- **Best match:** upload/snap a photo of her handwriting → top 3 closest
  styles presented first
- V1 launch constraint (per FULFILLMENT_RESEARCH): mailed cards use the
  fulfillment vendor's style catalog; print-at-home uses our full
  library. UI presents one gallery; mailed-tier availability shown per
  style.
- **PM question:** where does this step live — before generation (she
  sees drafts in her handwriting from the first moment, my lean) or
  after review?

### S9. Card design
**Job:** pick the physical card. Curated, not overwhelming.
- Style families (elegant / playful / minimal / floral …) × occasion
- Front: design or (upsell, V1.1+) photo from the event
- Inside: the note, her handwriting, layout auto-fit with length warnings
  back at edit time
- Envelope preview: recipient address in the SAME handwriting style
  (hard requirement), return address, postage
- **PM question:** one design for the whole event (my lean for V1) or
  per-recipient variation?

### S10. Checkout
**Job:** clear price, no surprises, confidence everything is correct.
- **Only signed-off cards can enter an order** — hard gate, no
  exceptions. Un-signed cards are listed as "still in the studio."
- **Checkout is the address deadline:** any card whose recipient still
  needs an address (📮) or has a ⚠ validation flag gets fixed here —
  inline, one at a time, or "mail these 60 now, finish the rest later"
- Order summary: N cards × tier price, volume discount applied and
  *shown* ("You saved $12"), any upsells later
- Print-at-home path forks here: free, generates the PDF, done
- Stripe checkout; Apple Pay / Google Pay enabled (mobile-first!)
- **PM question:** partial sends allowed (mail the 60 approved now, 20
  later) — V1 or V1.1?

### S11. Orders & tracking
**Job:** trust. She paid; show her it's happening.
- Per-event order page: each card's status (rendering → sent to print →
  mailed → estimated delivery window)
- Email notifications at "mailed" milestone
- **PM question:** how much vendor status detail is reassuring vs
  noisy?

### S12. Account & address book
- Saved recipients (reusable across events), past orders/receipts,
  saved handwriting choice + tone defaults, sign-out, delete account
  (really deletes — PII promise)

---

## Explicitly NOT in V1 (parked, per plan)

Voice capture · gift photos · live gift-opening mode · QR voice messages ·
photo-front cards · wedding-platform importers · pen-written tier ·
custom font from sample · address-request links · native mobile app

## Build order (proposed milestones)

- **M0 — Walking skeleton:** deployed app, sign-in, empty event shell.
  *Sharon gets a URL she can open on her phone.*
- **M1 — Recipients:** manual + CSV + validation + address book
- **M2 — Gifts & generation:** capture → tone → batch generate → review
  loop (plain text, no handwriting yet)
- **M3 — Handwriting & rendering:** font gallery, best-match, card/PDF
  renderer, print-at-home free tier end-to-end
- **M4 — Card design:** style families, full card + envelope preview
- **M5 — Checkout:** Stripe, pricing tiers, order records
- **M6 — Fulfillment:** vendor integration, order tracking, email
  notifications → **V1 complete: real card lands in a real mailbox**

Each milestone is demoable on its own. M0–M2 need zero vendor decisions;
the Thanks.io/Scribeless call must land before M6 (and stock samples
should be ordered during M0–M1).
