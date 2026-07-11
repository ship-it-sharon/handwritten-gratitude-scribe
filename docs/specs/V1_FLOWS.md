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
Landing → Sign up → Create event → Add recipients → Add gifts/notes
→ Set tone & handwriting → Generate → Review each card → Design card
→ Checkout → Track orders
```

She can leave and return at any step; the event remembers where she was.

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

### S4. Recipients
**Job:** get the list in with minimum typing.
- V1 sources: **manual entry** + **CSV/spreadsheet upload** with smart
  column mapping (LLM maps "Auntie's addr" → address fields)
- Address validation on entry; per-recipient status chip
  (✓ valid / ⚠ check this / ✗ undeliverable)
- Recipients belong to her address book; an event *selects* from it
  (so the baby shower next year reuses the wedding list)
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

### S7. Generate & review — THE screen
**Job:** review 80 notes in 15 minutes and feel in control the whole time.
- Batch generation runs with visible progress; cards become reviewable
  as they finish (no waiting for the whole batch)
- One card at a time: recipient context on top, note below, rendered in
  her chosen handwriting (preview)
- Actions: **Approve** (primary, fast) / **Edit** (inline text edit) /
  **Redo** (regenerate; optional one-line steer like "mention the beach")
- Batch-level guarantee: structural variety across notes (no two
  identical openers/closers)
- Progress bar: 34 of 80 approved
- **PM questions:** approve-all-remaining button — empowering or
  dangerous? Should edits retrain the batch's voice on the fly?

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
- Order summary: N cards × tier price, volume discount applied and
  *shown* ("You saved $12"), any upsells later
- Final address confirmation pass (anything still ⚠ gets fixed here)
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
