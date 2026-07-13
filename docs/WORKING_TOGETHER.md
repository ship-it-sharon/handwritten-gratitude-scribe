# Working Together — PM + Claude Code

_How a non-coding product manager drives this project. Written 2026-07-11._

## The division of labor

**You (PM):** what to build, for whom, in what order. Specs, flows, copy,
priorities, taste. You review the *running app*, never the code.

**Claude (engineering):** everything between your spec and a deployed URL —
code, database, integrations, testing, deployment. Also: flagging when a
spec is ambiguous, expensive, or has a cheaper alternative, before building.

## The workflow loop

1. **Spec** — you describe a feature: in chat, or by editing a spec doc in
   `docs/specs/`. Bullet points and screenshots-of-sketches are fine;
   user-story format ("As a bride, I want…") works but isn't required.
   The most useful spec ingredients, in order of value:
   - the user's goal and emotional state on each screen
   - what "done" looks like (acceptance criteria)
   - what's explicitly out of scope
   - edge cases you care about (and which ones you don't)
2. **Build** — Claude implements on a branch and pushes.
3. **Review** — you open the preview URL (every push gets its own live
   preview once hosting is set up) and click through it like a user.
4. **Feedback** — plain language in chat: "the button feels buried,"
   "wrong tone in the empty state," "this flow needs a back button."
   Screenshots with scribbles are excellent input.
5. **Merge** — when you're happy, it ships to the main site.

## Ground rules that make this go well

- **Small slices.** One feature or flow per cycle, not "build V1."
  Small slices = fast previews = fast course-correction.
- **Walking skeleton first.** The first milestone is a nearly-empty app,
  deployed, with sign-in working — so the review loop exists from week
  one and every later feature lands somewhere you can click.
- **Decisions live in docs, not chat scrollback.** When we decide
  something, it gets written into `docs/` and committed. Chat is for
  deciding; the repo is for remembering.
- **You own all the accounts.** Hosting, domain, database, payments, AI
  keys — created under your email, your billing. Claude uses them but
  never owns them. (List in `docs/INFRASTRUCTURE.md`.)
- **Ask "what are my options?" freely.** Any technical decision can be
  translated into product terms — cost, speed, risk, user experience.
  If a technical answer doesn't come with a product-terms translation,
  ask for one.

## Spec template (copy into docs/specs/ per feature)

```markdown
# Feature: <name>

## Who & why
<user + the job this does for her>

## Flow
<numbered steps or screen list>

## Done means
- [ ] <observable behavior>

## Data touched (required — see docs/DATA_PROTECTION.md)
- <new data collected, its sensitivity class, why, retention, which
  vendors see it — or "none">

## Out of scope
- <explicitly not now>

## Open questions
- <things you want Claude's input on>
```

## Cadence suggestion

Weekly rhythm that fits PM life: spec 1–2 slices → Claude builds →
review previews mid-week → feedback → merge by week's end. Planning docs
get updated whenever a decision changes them.
