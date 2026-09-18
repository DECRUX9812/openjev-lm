# Provenance and ethics

Plain version, because this project distills a commercial model's judgments and that deserves a
straight paragraph rather than a footnote.

**What we did.** Every label in `data/` and every training target is an answer that the hosted Jev
API returned to a request we sent, recorded verbatim. We did not obtain Jev's weights, prompts,
or internal data, and we redistribute none of those. The models here were trained *on answers*:
the adapter and the classifier are workalikes that approximate Jev's judgment on this task. They
are not copies, and neither is claimed to be.

**What we published.** Our training corpora (postings + Jev's answers to them), our code, and our
weights, under MIT. The postings are public job advertisements. The answers are outputs of a
paid API, published as a research record of what the model said on specific inputs.

**The honest caveat.** Whether training and publishing on a vendor's API outputs is within that
vendor's terms is a question between you and the vendor; we flag it rather than bury it. We are
not affiliated with TypeSafe and this work is not endorsed by them. If TypeSafe objects to any
of this, the right response is to take it down on request — the point of the project was the
measurement and the recipe, not the artifact.

**What we would tell a reader.** Two things are true at once: reproducing a hosted model's
judgment locally is now cheap enough to do overnight, and the reproduction inherits both the
vendor's judgment *and* the vendor's mistakes, frozen at the moment of distillation. If you
deploy a workalike, you are deploying a snapshot of someone else's opinions. Say so, version it,
and re-distill when the upstream moves.
