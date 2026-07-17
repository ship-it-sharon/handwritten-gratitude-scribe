# Design System — Radix Themes

_Adopted 2026-07-16. Decision: Radix Themes (the styled system built on
Radix Primitives) — not raw primitives (all styling would be on us), not
shadcn/ui (more owned code than we need)._

## Why Radix Themes

- **Design decisions become token decisions** a PM can make in plain
  language ("warmer gray", "rounder corners", "deeper rose") — not CSS.
- **Accessibility built-in** (focus management, contrast, keyboard
  navigation) — which quietly serves Posy's accessibility roadmap.
- One dependency, versioned, maintained; no component code to own.

## Posy's theme tokens (set in `apps/web/app/layout.tsx`)

| Token | Value | In plain language |
|---|---|---|
| `accentColor` | `ruby` | The brand color for buttons/links — a warm rose |
| `grayColor` | `sand` | Warm-toned grays instead of cold blue-grays |
| `radius` | `large` | Soft, friendly corners |
| `appearance` | `light` | Light mode only for now (dark mode = one-token change later) |
| Background | `#faf7f2` (warm paper) | Set in `globals.css` via Radix's variable |
| Font | Georgia serif | Stationery voice; a proper brand font can swap in later at one place |

**To change the look, Sharon says the token change** ("try `crimson`
accent", "smaller radius") — one-line edits, instantly consistent
everywhere.

## Usage rules (enforced in code review)

1. **Radix components first** for all UI: `Button`, `Card`, `TextField`,
   `Select`, `Badge`, `Callout`, `Checkbox`, `Heading`, `Text`, `Flex`.
   No hand-rolled buttons/inputs/cards.
2. **Custom CSS only for brand moments:** the wordmark, paper textures,
   and (M3+) rendered card/envelope previews — those must look like
   stationery, not UI, and live in `globals.css` with a comment saying
   why.
3. **Bespoke layouts** (e.g. the CSV mapping table) may use custom CSS
   but Radix color variables (`var(--gray-4)` etc.) so theme changes
   propagate.
4. **Semantic colors:** success = `grass`, warning = `amber`, error =
   `red`, brand = accent (`ruby`). Badges/callouts pick from these, not
   arbitrary colors.

## Component vocabulary (what things are called)

| In the app | Radix component |
|---|---|
| Primary / secondary button | `Button` solid / soft |
| Quiet inline action ("remove") | `Button` ghost, size 1 |
| Content panel | `Card` |
| Status chip (address state, counts) | `Badge` |
| Success/error/info banner | `Callout` |
| Form field | `TextField.Root` (+ `Select.Root`, `Checkbox`) |
| Page copy | `Text` / `Heading` |

## Later

- Dark mode: flip `appearance` (design pass on the paper background
  needed).
- Brand typeface: swap the Georgia stack in one CSS variable.
- Card-preview aesthetic (M3): deliberately NOT Radix — real paper
  textures, handwriting fonts, print fidelity.
