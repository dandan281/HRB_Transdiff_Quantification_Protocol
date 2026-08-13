# Blind repeat review — instructions for the operator

**Time needed:** ~10–15 minutes for 30 cases.
**What you do:** re-decide 30 fibres you've reviewed before, without any hint of your earlier call.
**Why it matters:** this is the single readiness check (**G-SO1**) that unlocks model training. Read §1 so you understand *why* before you start — it changes how you should do it.

---

## 1. Why this exists (please read)

Normally, quality of annotation is checked by having **two people** label the same images and measuring how often they agree. This project has **one operator — you**. So instead of two people, we check whether **you agree with yourself**: you re-review a sample of cases *blind* (not knowing what you decided the first time), and we measure how consistent your two passes are.

That consistency number is the evidence that your annotations are reliable enough to train a model on. If you and your earlier self agree ≥85% of the time (with no unsafe errors), training is greenlit. If not, we look at exactly which cases you flip-flopped on and fix the guidance — which is *good*, it means we caught a real ambiguity before it poisoned the model.

**So the goal is not to "pass." The goal is an honest second opinion.** Decide each case genuinely. If you disagree with your past self, that's useful signal, not a failure.

---

## 2. Before you start — the washout

- **Do this after a break** — ideally the next day, or at least after a coffee/lunch gap. The point is that you should *not remember* your first-pass calls.
- **Do not** open your earlier decision files, the well pages, or your notes for these cases first. Go in cold.
- The page is already blind for you: it shows the fibres in **random order**, with **no earlier decision** and **no model suggestion**. Every card starts on **Ambiguous**, so nothing is pre-decided.

---

## 3. How to decide each fibre — the conservative rules

For each card, ask one question: **"Is this ONE real, complete myotube?"**

| Choose | When | Key | Rule |
|---|---|---|---|
| **Accept** | You are **confident** it's one real, complete myotube | `A` | Only accept when sure. If you'd hesitate, don't. |
| **Reject** | It's clearly **not** a myotube — debris, a stray fragment, background, a fat blob | `R` | |
| **Ambiguous** | You **genuinely can't tell** — unclear identity, unclear boundary, or you're just not sure | `X` | **This is the safe default. When in doubt, pick Ambiguous — never Accept a doubtful fibre.** |

**The one rule that matters most: when unsure, do NOT Accept.** A wrong Reject or Ambiguous costs us a little data. A wrong Accept puts a bad fibre into the training set. So the conservative choice (Reject or Ambiguous over Accept) is always the right tiebreaker.

### The tightened length rule (round 2 — grounded in your own data)

Round 1 showed your accept/ambiguous line wobbles on **short fibres and edge fibres**. Your own decisions show the call is driven almost entirely by **length** (not aspect or width). Accept-rate by length: `<60µm → 14%`, `60–90 → 24%`, `90–120 → 37%`, `120–250 → ~42%`, `≥250µm → 65%`. So use length as the anchor:

- **≥ 250 µm**, clearly one elongated fibre → **Accept.**
- **120–250 µm** → **judgment zone.** Accept only if it's a clean, continuous single fibre; **if you hesitate at all → Ambiguous.**
- **< 120 µm → default Ambiguous.** Accept a short fibre **only** if it is unmistakably a crisp, continuous thin fibre with zero doubt. You accept only ~1 in 5 of these, so **Ambiguous is the correct default and Accept is the rare exception** — never accept a short fibre "to be safe."
- **Aspect and width do not rescue a short fibre** — the length call stands.
- **Edge/border fibre:** Accept the visible part if it's clearly a real fibre (auto-recorded as border-truncated); if it's messy or you can't tell it's real → **Ambiguous**. **Do NOT Reject a fibre just because it's cut off by the edge** — Reject is only for things that clearly aren't myotubes. (In round 1 a real 299 µm edge fibre got rejected — that should have been Accept or Ambiguous.)

The single sentence: **length ≥ 250 → Accept if clean; under 120 → Ambiguous unless unmistakable; in between, when in doubt → Ambiguous; never Reject a real fibre just for touching the edge.**

### Special cases

- **Fibre runs off the edge of the image** (marked with an orange **`edge`** tag): still **Accept** it if it's a real myotube. The system automatically records it as **border-truncated** and keeps it *out* of the length statistics. **Do not** try to guess or draw its missing length — just accept the visible part.
- **The mask is wrong but the fibre is real:** Accept, then fix the mask with the editor —
  - **too short** (machine stopped before the fibre ends): enlarge, use **✏ Add** (`B`) to extend along the fibre. There's extra image around each proposal for exactly this.
  - **spilled over** (mask grabbed extra): use **Erase** (`E`) to trim it back.
- **Looks like two fibres merged into one proposal:** enlarge, press **⟳ Hypothesis** (`H`) to auto-split, fix colours with **◫ Assign** (`G`) if needed, then Accept — each colour becomes its own fibre.
- **Don't overthink.** Spend a few seconds per card and go with your genuine first read. Your natural judgement is exactly what we're measuring — deliberating for a minute defeats the point.

---

## 4. Step by step

1. Open the blind page (I'll give you the link / it opens in your browser).
2. Read the blue **"blind repeat"** box at the top once.
3. Go through **all 30 cards.** Fastest way is the keyboard:
   - `A` = Accept & next, `R` = Reject & next, `X` = Ambiguous & next
   - `←` / `→` = move between cards, click a card to enlarge and edit
   - `?` = shortcut list
4. Make sure **every one of the 30** has a decision (none left on the default Ambiguous *by accident* — Ambiguous is fine as a real choice, just make sure you actually looked).
5. When done: click **⇅ Save / Restore** (bottom bar) → **⧉ Copy all** → paste the text back to me (or save it and send the file).

That's it. Don't compare against your first pass afterward — the comparison is done automatically.

---

## 5. What happens after you send it back

1. I run a quick reference check (`blind-compare`) so you can see the agreement number.
2. Codex runs the **official G-SO1 gate**, which needs all of:
   - **≥ 85%** disposition agreement (accept vs reject vs ambiguous) between your two passes,
   - **no unsafe border-complete error** (a truncated fibre must never be treated as full-length),
   - **median mask overlap (IoU) ≥ 0.80** on at least **8** cases you accepted as complete both times,
   - genuine disagreements are simply excluded / marked ambiguous, not forced.
3. **If it passes:** model training (the bootstrap on your 377 masks + synthetic data) is unlocked.
   **If it doesn't:** we look at the specific cases you flipped on, tighten the rule for that situation, and you may do one more small repeat. No wasted effort — every disagreement teaches us something.

---

**In one line:** re-decide 30 fibres cold, accept only when sure, prefer Ambiguous/Reject when in doubt, fix obvious mask errors, then Save → Copy all → send it back.
