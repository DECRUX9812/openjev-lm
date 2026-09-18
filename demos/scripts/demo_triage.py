"""openjev computer-use demo — LM-arm-driven inbox triage on this desktop.

A folder of mixed inbound .txt files (lead requests, job ads, spam, notes).
For each file the 0.5B LoRA adapter (trained on THIS box) scores the seven
questions via staged likelihood; the script performs the action it decided:
file moves into ACTION folders, lead drafts get written. No cloud call.
"""
import re, shutil, sys, time
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/repos/openjev-lm/openjev')
from dataset import prompt_for
import eval_likelihood as E

BASE = Path('/home/ubuntu/openjev_inbox')
INBOX = BASE / 'INBOX'
ACT = {
    'service_lead': BASE / 'ACT_OUTREACH',
    'staff_role': BASE / 'ACT_FORWARD',
    'generic_job': BASE / 'ACT_JOB_ADS',
    'junk': BASE / 'ACT_TRASH',
}

SEED = [
    ("req_plumbing_site.txt", "From: mike@mikesplumbing.com\nSubject: need a website for my plumbing business\n\nHi - we're a 3-truck plumbing outfit in Austin. Need a simple site + google business setup. Budget around $2,000. Can you start this month?"),
    ("req_redesign.txt", "From: dana@bloombakery.co\nSubject: our bakery site looks 2009\n\nWe need a refresh before the holiday rush. Shopify-based. ~$1,500 flexible. Photos ready."),
    ("req_landscaping.txt", "From: jim@greenthumb\nSubject: lawn care co needs landing page + booking\n\nSmall landscaping crew, want online booking + quote form. $800-1200 range."),
    ("req_dashboard.txt", "From: ops@fleetco.io\nSubject: internal dashboard for dispatch\n\nWe run 12 vans. Need a simple dispatcher dashboard. Have API. Budget $5k."),
    ("req_logo.txt", "From: kat@katcrafts\nSubject: logo + tiny site for etsy seller\n\nSelling candles, need logo and one-page site. Can pay $400."),
    ("req_crm.txt", "From: pavel@quickmove\nSubject: moving company needs quote calculator\n\nCustomers want instant quotes. Need form + pricing logic. $3,000."),
    ("job_swe.txt", "From: jobs@bigcorp.example\nSubject: Senior Software Engineer — Platform\n\nBigCorp is hiring a Senior SWE for platform team. $180-220k + equity. Apply via portal."),
    ("job_nurse.txt", "From: hr@stmarys.org\nSubject: RN — ICU nights\n\nSt. Mary's Hospital seeks RN for ICU night shift. $78k-95k. Benefits included."),
    ("job_cook.txt", "From: hiring@olivegarden.example\nSubject: Line Cook — full time\n\nOlive Garden hiring line cooks, $17.50/hr, evenings. Immediate start."),
    ("job_data.txt", "From: talent@datafirm.example\nSubject: Data Analyst — remote\n\nDataFirm seeks analyst, SQL + Python. $110k remote. 3 rounds of interviews."),
    ("spam_prince.txt", "From: prince@totallylegit.biz\nSubject: URGENT business proposal\n\nDear friend, I have $14,000,000 to transfer and need your assistance. Reply with bank details."),
    ("spam_crypto.txt", "From: moon@cryptogains.gg\nSubject: 100x guaranteed!!!\n\nThis token WILL moon. Buy now, thank me later. Not financial advice lol."),
    ("spam_seo.txt", "From: rank#1@seomagic.example\nSubject: I found errors on your website!\n\nYour site has 47 critical SEO errors. Buy our audit package $299. Limited time!"),
    ("spam_lottery.txt", "From: winner@eurolotto.example\nSubject: You won 2.5M EUR\n\nCongratulations! Your email won our lottery. Claim now, pay only 50 EUR processing fee."),
    ("note_mom.txt", "From: mom@family.example\nSubject: dinner sunday\n\nDon't forget dinner Sunday at 6. Bring dessert. Love, mom"),
    ("note_self.txt", "From: me@self.example\nSubject: pickup dry cleaning\n\nReminder: dry cleaning ready Thursday. Also dentist Tue 9am."),
    ("newsletter.txt", "From: digest@dailynews.example\nSubject: Your morning digest\n\nTop stories: markets up, weather fine, local team wins. Read more..."),
    ("req_wedding.txt", "From: sarah@bride2026.example\nSubject: wedding photobooth site\n\nGetting married in June — need a booking page for my photobooth side hustle. ~$600."),
    ("job_intern.txt", "From: campus@techuni.example\nSubject: Summer internship — CS students\n\nTechUni partner companies seeking CS interns. $25/hr, remote OK. Deadline May 1."),
    ("req_invoice.txt", "From: accounting@blueprintbuilders.example\nSubject: need invoice automation\n\nWe hand-write invoices. Want a simple tool that generates PDF invoices. $1,200."),
]

def seed():
    shutil.rmtree(BASE, ignore_errors=True)
    INBOX.mkdir(parents=True)
    for d in ACT.values():
        d.mkdir(parents=True)
    for name, body in SEED:
        (INBOX / name).write_text(body)

def main():
    if '--seed' in sys.argv:
        seed()
        print(f'seeded {len(SEED)} files in {INBOX}')
        return
    print('loading 0.5B + adapter trained on this box …')
    model, tok, _ = E.load_model('/home/ubuntu/repos/openjev-lm/models/Qwen2.5-0.5B-Instruct', 8, want_lora=True)
    state, step = E.load_lora_state('/home/ubuntu/repos/openjev-lm/openjev/runs/retrain-check/adapter.pt')
    print(E.apply_adapter(model, state), f'(adapter step {step})')

    print(f'\n{len(list(INBOX.iterdir()))} files in ~/openjev_inbox/INBOX — the model decides, the OS obeys\n')
    time.sleep(2)

    stats = {"boundary_fallbacks": 0}
    drafts, correct = [], 0
    for i, f in enumerate(sorted(INBOX.iterdir())):
        body = f.read_text()
        m = re.search(r'Subject: (.*)', body)
        subj = m.group(1) if m else f.stem
        frm = re.search(r'From: (\S+)', body)
        pay = re.search(r'\$[\d,]+(?:k)?(?:\s*(?:-|–)\s*\$?[\d,]+k?)?', body)
        title = (subj + ' — ' + body.split('\n\n', 1)[-1][:120])[:140]
        row = {"id": f.stem, "title": title,
               "prompt": prompt_for(title, frm.group(1) if frm else '', pay.group(0) if pay else '')}
        t = time.time()
        out = E.score_row(model, tok, row, stats, use_enum=False)
        s = out['staged']
        sec = time.time() - t
        dest = ACT[s['bucket']]
        shutil.move(str(f), dest / f.name)
        truth = 'service_lead' if f.name.startswith('req_') else \
                ('junk' if f.name.startswith(('spam_', 'note_', 'newsletter')) else 'job')
        ok = (s['bucket'] == truth) or (truth == 'job' and s['bucket'] in ('generic_job', 'staff_role'))
        correct += ok
        if s['bucket'] == 'service_lead':
            drafts.append((f.stem, body))
        flag = '' if ok else '   ← miss'
        print(f'[{i+1:02d}] {f.name:<24} → {s["bucket"]:<13} fit={s["fit"]} {sec:4.1f}s  ▸ {dest.name}/{flag}')
        time.sleep(0.6)

    ddir = BASE / 'ACT_OUTREACH' / 'drafts'
    ddir.mkdir(exist_ok=True)
    for stem, body in drafts:
        m = re.search(r'Subject: (.*)', body)
        (ddir / f'{stem}_reply.txt').write_text(
            f'Re: {m.group(1) if m else stem}\n\n'
            'Hi — thanks for reaching out. This looks doable; sending a short proposal shortly.\n'
            '— auto-drafted by openjev-lm (service_lead only)\n')
    print(f'\n{len(drafts)} service_lead(s) → draft replies written to ACT_OUTREACH/drafts/')
    print(f'triage complete — {correct}/{i+1} landed right, zero API calls, $0\n')

if __name__ == '__main__':
    main()
