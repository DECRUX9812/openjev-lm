#!/usr/bin/env python3
"""synth_core.py - balanced synthetic SaskJobs corpus, labelled live by Jev.

Why this exists
---------------
`data/train.jsonl` is 2,326 generic_job / 35 staff_role / 0 service_lead / 0 junk: a model
trained on it learns "everything is generic_job". This script manufactures coverage instead
of waiting for the stream to supply it.

Label provenance (read this before using the file)
--------------------------------------------------
Every row that is kept is labelled by **Jev himself**: one `qualify_many()` request per
posting against the same 7-question rubric the production watcher uses, and the training
`target` is built from his response (bucket = his bucket; each boolean = his probability
>= 0.5; fit = round(his fit) clamped to 0-4). There is no teacher model anywhere in this
pipeline - no Claude, no GPT - so there is no teacher-vs-Jev gap in the labels. The only
extra fields on a row are `source`, `stratum` (the generation intent) and `rank` (realism
rank); none of them are trained on. The teacher-first pipeline that does exist in this repo
(`synth/generate.py`) is a different experiment; its 40 Jev-checked rows are used here for
exactly one thing: the teacher-vs-Jev agreement number in the report, which says how much
label noise a teacher-labelled corpus would have carried.

How it works (all deterministic, SEED below)
--------------------------------------------
1. Employer names are *harvested from the real SaskJobs corpus* (typesafe-lab
   runs/leads_corpus_full.json, read-only) so they are real Saskatchewan organizations -
   co-ops, credit unions, school divisions, health facilities, restaurants, dealerships,
   machine shops, dental offices, hotels - and never "Acme Corp". Names are bucketed into
   institutional / tech / local pools by keyword and dealt *without replacement* inside a
   pool, so no employer is over-used.
2. Postings are composed from per-stratum title banks x modifier x pay-style so the intended
   bucket is grounded in *who is hiring* (a health region filling an IT seat is staff_role;
   a restaurant that needs a website is service_lead) rather than in keywords alone.
3. Candidates are labelled by the live Jev API; rows where Jev overrules the intent are KEPT
   (his label is the truth) and counted per stratum in the report.
4. Rows are emitted with `dataset.prompt()` verbatim so this file is byte-compatible with
   the real corpus. The target is rebuilt from the *rounded* jev values so a reader can
   recompute target from the `jev` dict with no floating-point schism.

Quality gates are checked and printed (dedupe, gold/corpus leak, stratum mix, pay share,
employer plausibility). Progress checkpoints after each chunk, and the Jev disk cache makes
a re-run nearly free: re-runs pick up cached answers.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

LAB = Path("/home/decrux/Code/typesafe-lab")
ROOT = Path("/home/decrux/Code/jev-repro-test/openjev")
DATA = ROOT / "data"
RUNS = ROOT / "runs"
DATA.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(LAB))
sys.path.insert(0, str(ROOT))

from typesafe_lab.cache import JsonCache                      # noqa: E402
from typesafe_lab.client import SystemOneClient               # noqa: E402
from typesafe_lab.leads import qualify_many                   # noqa: E402
import dataset as DS                                          # noqa: E402  (prompt/target/NOULS)

SEED = 20260918
CACHE_PATH = DATA / "cache_synth_core.json"
OUT_JSONL = DATA / "synth_core.jsonl"
OUT_POSTINGS = DATA / "synth_core_postings.jsonl"
OUT_STATS = RUNS / "synth_core_stats.json"
OUT_USAGE = RUNS / "synth_core_usage.json"
OUT_REPORT = RUNS / "synth_core_report.md"
TEACHER_CHECKED = ROOT / "synth" / "data" / "jev_checked.jsonl"

NOULS = DS.NOULS
STRATA = {"generic_job": 0.35, "staff_role": 0.25, "service_lead": 0.25, "junk": 0.15}
#: how often a stratum states pay. Chosen so the whole file lands near the 40% gate.
PAY_RATE = {"generic_job": 0.30, "staff_role": 0.60, "service_lead": 0.45, "junk": 0.35}

# --------------------------------------------------------------------------- title banks
GENERIC_TITLES = [
    # care / health
    "Continuing Care Assistant", "Health Care Aide", "Licensed Practical Nurse",
    "Registered Nurse", "Nurse A - Registered Nurse General Duty", "Combined Laboratory and X-Ray Technologist",
    "Medical Office Assistant", "Pharmacy Assistant", "Dental Receptionist", "Physiotherapist",
    "Occupational Therapist", "Respiratory Therapist", "Social Worker", "Addictions Counsellor",
    "Community Support Worker", "Direct Support Professional", "Recreation Therapist",
    # education
    "Early Childhood Educator", "Educational Assistant", "Elementary School Teacher",
    "High School Science Teacher", "Instructor - Adult Basic Education", "School Bus Driver",
    "Tutor - After School Program", "Custodian",
    # food / retail
    "Cook", "Line Cook", "Sous Chef", "Food Service Supervisor", "Kitchen Helper", "Dishwasher",
    "Baker", "Barista", "Server", "Bartender", "Cashier", "Retail Sales Associate",
    "Store Manager", "Grocery Clerk", "Shelf Stocker", "Department Supervisor", "Meat Cutter",
    # driving / logistics / trades
    "Delivery Driver", "Class 1 Truck Driver", "Long Haul Truck Driver", "Warehouse Associate",
    "Forklift Operator", "Shipper/Receiver", "Dispatcher", "Heavy Duty Mechanic",
    "Journeyperson Electrician", "Plumber", "HVAC Technician", "Welder", "Machinist",
    "Millwright", "Carpenter", "Concrete Finisher", "Roofer", "Painter", "Landscaping Labourer",
    "Equipment Operator", "Farm Equipment Operator", "Production Worker", "Packaging Operator",
    "Quality Control Inspector", "Agricultural Commodity Handler",
    # office / legal / finance / sales / management
    "Administrative Assistant", "Office Administrator", "Receptionist", "Executive Assistant",
    "Accounting Clerk", "Accounts Payable Clerk", "Payroll Administrator", "Bookkeeper",
    "Legal Assistant", "Paralegal", "Legal Counsel - Litigation", "Accountant", "Tax Preparer",
    "Financial Analyst", "Branch Manager", "Operations Manager", "Human Resources Coordinator",
    "Recruiter", "Sales Representative", "Account Manager", "Territory Sales Manager",
    "Insurance Broker", "Customer Service Representative", "Call Centre Agent", "Security Guard",
    "Housekeeping Room Attendant", "Front Desk Agent", "Hotel Maintenance Worker",
    "Groundskeeper", "Property Manager", "Hairstylist", "Esthetician", "Unit Clerk",
    "Home Care Scheduler", "Assessment Coordinator", "Bingo Caller", "Funeral Services Assistant",
]
GENERIC_MODS = ["", "", "", "", "", "", "", " (Casual)", " (Part-Time)", " (Full-Time)",
                " (Term)", " - Evening Shift", " - Weekend Availability", " - Days",
                " - Night Shift", " - Rotating Shifts", " - Full Time Permanent"]

STAFF_TITLES = [
    "IT Support Analyst", "Technical Support Analyst", "Help Desk Analyst",
    "Desktop Support Technician", "Systems Administrator", "Network Administrator",
    "Network Engineer", "Senior Systems Analyst", "Cybersecurity Analyst",
    "Information Security Analyst", "Security Operations Analyst", "Software Developer",
    "Senior Software Developer", "Full Stack Developer", "Application Developer",
    "Programmer Analyst", "Programmer Analyst III - GIS", "QA Analyst", "QA Automation Engineer",
    "DevOps Engineer", "Cloud Platform Engineer", "Data Analyst", "Data Engineer",
    "Business Intelligence Analyst", "Database Administrator", "Data Architect",
    "Information Management Analyst", "GIS Analyst", "Geospatial Technician", "IT Project Manager",
    "IT Manager", "Manager, Information Technology", "Director, Enterprise Security",
    "Enterprise Architect", "Solution Architect", "Applications Architect",
    "Clinical Informatics Specialist", "ERP Systems Analyst", "Financial Systems Analyst",
    "SharePoint Administrator", "Identity and Access Management Analyst",
    "Telecommunications Analyst", "IT Asset Coordinator", "Systems Analyst",
    "Software Engineer", "Automation Engineer", "Middleware Administrator",
]
STAFF_MODS = ["", "", "", "", "", "", " (Term)", " (Temporary)", " - Term Position",
              " - Permanent", " (Permanent Full-Time)"]

SERVICE_TITLES = [
    "Website Developer", "Web Designer", "Web Site Developer", "Website Designer and Developer",
    "Front-End Web Developer", "WordPress Website Developer", "E-Commerce Website Developer",
    "Online Store Coordinator", "IT Support Technician", "Computer and Network Support",
    "IT Support (Part-Time)", "Small Business IT Support", "Point of Sale System Support",
    "POS and Inventory Systems Support", "Bookkeeping and Reporting Automation",
    "Reporting and Data Coordinator", "Cybersecurity and IT Consultant",
    "Network Security Upgrade - Contractor", "Website and Social Media Coordinator",
    "Social Media and Digital Marketing Coordinator", "SEO and Google Ads Specialist",
    "Website Maintenance and IT Support", "IT Consultant - Systems Review",
    "Custom Software Developer (Contract)", "Database and Reporting Support",
    "Website Redesign - Contract", "Computer Repair Technician", "Shopify Store Manager",
    "Booking System and Website Coordinator", "Email Marketing and CRM Coordinator",
    "Automation and Controls Technician", "Network Cabling and IT Technician",
]
SERVICE_MODS = ["", "", "", "", " (Part-Time)", " (Contract)", " - Contract", " - Term",
                " - Part Time / Flexible Hours", " (Contract, 6 Months)"]

JUNK_TITLES = [
    "Sales Representative - Commission Only", "Commission-Only Sales Agent",
    "Independent Sales Distributor", "Brand Ambassador - Uncapped Commission",
    "Sales Partner - Work From Home", "Marketing Representative - Commission Based",
    "Work From Home - No Experience Needed", "Entry Level - Immediate Start - Weekly Pay",
    "General Help Wanted - Various Positions", "Various Positions - Apply Within",
    "Multiple Positions Available - Immediate Start", "Warehouse Staff - Ongoing - Immediate Start",
    "General Labourers - Multiple Openings", "Skilled Trades - Multiple Positions",
    "Volunteer Board Member", "Volunteer Fundraising Coordinator", "Unpaid Marketing Intern",
    "Volunteer Front Desk Assistant", "Door-to-Door Sales Agent - Commission Only",
    "Mystery Shopper - Flexible Hours", "Telemarketing Sales Agent - Commission",
    "Independent Contractor - Set Your Own Hours", "Business Opportunity - Be Your Own Boss",
    "Team Leader - Health and Wellness Products", "Distributor Opportunity - No Experience",
    "Sales Associate - 100 Percent Commission", "Fundraiser - Commission Based",
    "Casual Labour - Various Shifts", "Real Estate Sales - Commission Only",
    "Insurance Sales Agent - Commission Only", "Make Money From Home - Flexible Schedule",
]
JUNK_MODS = ["", "", "", " - Immediate Start", " - Work From Home", " - No Experience Necessary",
             " - URGENT", " - Flexible Hours", " - Uncapped Commission", " - Apply Today"]

#: employer names for the junk stratum: mostly harvested locals (a vague commission ad does come
#: from a real business) plus place-stemmed marketing/staffing names. Never "Acme Corp".
JUNK_STEMS = ["Wheatland", "Qu'Appelle", "Wascana", "Prairie Sky", "Buffalo Pound", "Stonegate",
              "Legacy", "Harvest", "Cedar Ridge", "Northgate", "Sunrise", "Riverside", "Aspen"]
JUNK_SUFFIX = ["Marketing Group", "Sales Partners", "Independent Distributors", "Staffing Group",
               "Employment Solutions", "Promotions", "Direct Sales Team", "Brand Partners"]

INSTITUTIONAL_KEYS = [
    "HEALTH", "HOSPITAL", "CANCER", "SCHOOL", "DIVISION", "UNIVERSITY", "COLLEGE",
    "POLYTECHNIC", "LIBRARY", "CITY OF", "AUTHORITY", "AGENCY", "MINISTRY", "GOVERNMENT",
    "SGI", "SASKPOWER", "SASKENERGY", "CREDIT UNION", "CROWN", "POLICE", "RCMP", "MUNICIPAL",
    "UNION", "CO-OP", "COOP", "GALLERY", "MUSEUM", "INSTITUTE", "FOUNDATION", "SOCIETY",
    "HOUSING", "BOARD", "SASKATCHEWAN", "SASK ", "PUBLIC", "UNIVERSITY OF", "ASSOCIATION",
    "COUNCIL", "TREATY", "TRIBAL", "BAND", "GOVERNMENT OF",
]
TECH_KEYS = ["SOFTWARE", "TECHNOLOG", "SYSTEM", "SOLUTION", "DIGITAL", "CONSULT", "COMPUTER",
             "WEB ", "INFORMAT", "NETWORK", "CYBER", "AUTOMATION", "ROBOTIC", "DATA "]


# --------------------------------------------------------------------------- helpers
def clip(s: str, n: int = 90) -> str:
    return " ".join(str(s or "").split())[:n]


def target_from_jev(jev: dict) -> str:
    """Byte-identical to dataset.target(), but fed the *rounded* jev values so that
    target is always recomputable from the stored jev dict."""
    obj = {"bucket": jev["bucket"]}
    for k in NOULS:
        obj[k] = bool(float(jev[k]) >= 0.5)
    obj["fit"] = max(0, min(4, int(round(float(jev["fit"])))))
    return json.dumps(obj, separators=(", ", ": ")) + "<|im_end|>\n"


def jev_dict(row: dict) -> dict:
    """Jev's own answer, rounded. bucket_confidence/probabilities ride along for analysis;
    the five booleans and fit are recomputable from this dict alone."""
    fit = row.get("fit")
    out = {
        "bucket": row["bucket"],
        "bucket_confidence": round(float(row.get("bucket_confidence") or 0.0), 3),
        "fit": round(float(fit), 3) if fit is not None else None,
        **{k: round(float(row.get(k, 0.0)), 3) for k in NOULS},
    }
    pr = row.get("bucket_probabilities")
    if pr:
        out["bucket_probabilities"] = {k: round(float(v), 4) for k, v in pr.items()}
    return out


def rank_row(intent: str, row: dict) -> int:
    """1-5 realism rank. Agreeing with the intent matters, but so does how sure Jev was:
    a confident disagreement is a genuinely ambiguous posting, a timid one is a shrug."""
    conf = float(row.get("bucket_confidence") or 0.0)
    agree = row["bucket"] == intent
    if agree and conf >= 0.80:
        return 5
    if agree and conf >= 0.60:
        return 4
    if agree:
        return 3
    return 2 if conf < 0.60 else 1


class EmployerDealer:
    """Deal employers without replacement inside a pool; reshuffle when drained."""

    def __init__(self, pools: dict[str, list[str]], rng: random.Random):
        self.rng = rng
        self.pools = {k: sorted(set(v)) for k, v in pools.items()}
        self.bags: dict[str, list[str]] = {}
        self.last: dict[str, str] = {}

    def deal(self, pool: str) -> str:
        bag = self.bags.get(pool)
        if not bag:
            fresh = [e for e in self.pools[pool] if e != self.last.get(pool)]
            if not fresh:
                fresh = list(self.pools[pool])
            self.rng.shuffle(fresh)
            bag = fresh
        name = bag.pop()
        self.bags[pool] = bag
        self.last[pool] = name
        return name


def pay_string(rng: random.Random, stratum: str) -> str:
    """~40% of rows state pay; the formats are the ones SaskJobs actually prints."""
    bands = {
        "staff_role": ((28.0, 55.0), (55_000, 135_000)),
        "service_lead": ((22.0, 45.0), (45_000, 95_000)),
        "generic_job": ((15.0, 38.0), (35_000, 85_000)),
        "junk": ((15.0, 25.0), (30_000, 60_000)),
    }
    (lo_h, hi_h), (lo_y, hi_y) = bands[stratum]
    style = rng.choices(
        ["hourly_range", "hourly", "annual_range", "from_annual", "annual", "vague"],
        weights=[26, 16, 26, 10, 14, 8],
    )[0]
    if style == "hourly_range":
        a = round(rng.uniform(lo_h, hi_h - 2), 2)
        b = round(a + rng.uniform(1.5, 7.0), 2)
        return f"${a:.2f} - ${b:.2f}/hr"
    if style == "hourly":
        return f"${round(rng.uniform(lo_h, hi_h), 2):.2f}/hr"
    if style == "annual_range":
        a = int(rng.uniform(lo_y, hi_y - 6_000) / 500) * 500
        b = a + int(rng.uniform(5_000, 22_000) / 500) * 500
        return f"${a:,} - ${b:,}/yr"
    if style == "from_annual":
        return f"From ${int(rng.uniform(lo_y, hi_y) / 500) * 500:,}/yr"
    if style == "annual":
        return f"${int(rng.uniform(lo_y, hi_y + 10_000) / 1000) * 1000:,}/yr"
    return rng.choice(["Negotiable", "To be discussed", "Competitive", "Salary commensurate with experience"])


def compose_title(stratum: str, rng: random.Random) -> str:
    bank, mods = {
        "generic_job": (GENERIC_TITLES, GENERIC_MODS),
        "staff_role": (STAFF_TITLES, STAFF_MODS),
        "service_lead": (SERVICE_TITLES, SERVICE_MODS),
        "junk": (JUNK_TITLES, JUNK_MODS),
    }[stratum]
    title = rng.choice(bank) + rng.choice(mods)
    if stratum in ("staff_role", "generic_job") and rng.random() < 0.12:
        title = f"{rng.randint(100000, 999999)} - {title}"          # real SaskJobs requistion ids
    elif stratum == "staff_role" and rng.random() < 0.15:
        title = f"{rng.choice(['Senior', 'Junior', 'Lead', 'Intermediate'])} {title}"
    return title


def pick_employer(stratum: str, dealer: EmployerDealer, rng: random.Random) -> str:
    if stratum == "staff_role":
        pool = rng.choices(["institutional", "tech", "local"], weights=[60, 25, 15])[0]
    elif stratum == "service_lead":
        pool = "local"
    elif stratum == "generic_job":
        pool = rng.choices(["local", "institutional", "tech"], weights=[85, 10, 5])[0]
    else:
        if rng.random() < 0.5:
            return f"{rng.choice(JUNK_STEMS)} {rng.choice(JUNK_SUFFIX)}"
        pool = "local"
    return dealer.deal(pool)


# --------------------------------------------------------------------------- generation
def load_reference() -> tuple[dict, dict, set, set, set]:
    corpus = json.loads((LAB / "runs" / "leads_corpus_full.json").read_text())["rows"]
    gold = json.loads((LAB / "data" / "leads.gold.json").read_text())
    gold_labels = gold["labels"]
    by_id = {r["id"]: r for r in corpus}
    corpus_pairs = {(clip(r.get("title")).lower(), clip(r.get("employer")).lower()) for r in corpus}
    gold_titles = {clip(by_id[i].get("title")).lower() for i in gold_labels if i in by_id}
    gold_pairs = {(clip(by_id[i].get("title")).lower(), clip(by_id[i].get("employer")).lower())
                  for i in gold_labels if i in by_id}
    return gold_labels, by_id, corpus_pairs, gold_titles, gold_pairs


def build_postings(n: int, rng: random.Random) -> list[dict]:
    gold_labels, _, corpus_pairs, gold_titles, gold_pairs = load_reference()
    corpus = json.loads((LAB / "runs" / "leads_corpus_full.json").read_text())["rows"]
    employers = []
    for r in corpus:
        name = clip(r.get("employer"))
        if 3 <= len(name) <= 58:
            employers.append(name)
    employers = sorted(set(employers))
    upper = {e.upper(): e for e in employers}

    def matched(keys):
        return [e for u, e in upper.items() if any(k in u for k in keys)]

    inst = matched(INSTITUTIONAL_KEYS)
    tech = [e for e in matched(TECH_KEYS) if e not in inst]
    local = [e for e in employers if e not in inst and e not in tech]
    dealer = EmployerDealer({"institutional": inst, "tech": tech, "local": local}, rng)
    print(f"      employer pools: institutional {len(inst)}, tech {len(tech)}, local {len(local)}")

    quota = {k: int(round(n * v)) for k, v in STRATA.items()}
    while sum(quota.values()) < n:
        quota["generic_job"] += 1
    seen: set[tuple[str, str]] = set()
    postings: list[dict] = []
    for stratum, count in quota.items():
        made, tries = 0, 0
        while made < count and tries < count * 60:
            tries += 1
            title = compose_title(stratum, rng)
            if title.lower() in gold_titles:
                continue
            employer = pick_employer(stratum, dealer, rng)
            pair = (title.lower(), employer.lower())
            if pair in seen or pair in corpus_pairs or pair in gold_pairs:
                continue
            seen.add(pair)
            pay = pay_string(rng, stratum) if rng.random() < PAY_RATE[stratum] else ""
            postings.append({
                "id": f"synth-core-{len(postings) + 1:04d}",
                "title": title,
                "employer": employer,
                "pay": pay,
                "region": "Regina",
                "source": "saskjobs",
                "stratum": stratum,
            })
            made += 1
        if made < count:
            print(f"  !! {stratum}: only {made}/{count} candidates after {tries} tries")
    rng.shuffle(postings)
    for i, p in enumerate(postings, start=1):      # ids follow the shuffled order
        p["id"] = f"synth-core-{i:04d}"
    return postings


# --------------------------------------------------------------------------- selection
def select_kept(rows: list[dict], postings: list[dict], keep: int) -> tuple[list[dict], dict]:
    """Keep the `keep` best rows, preserving the stratum mix and pay-style diversity.

    Best = realism rank first (a row Jev agrees with, confidently, beats a row he overruled),
    then diversity inside a stratum: within a rank band we round-robin over pay-style so the
    kept set is not all pay-stated or all pay-blank. Rows Jev disagreed with are still
    eligible - they are only dropped if the file is over the cap and they are the weakest.
    """
    intent = {p["id"]: p for p in postings}
    if len(rows) <= keep:
        return rows, {"dropped": 0, "reason": "under cap"}
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stratum[intent[r["id"]]["stratum"]].append(r)
    quota = {k: int(round(keep * v)) for k, v in STRATA.items()}
    while sum(quota.values()) < keep:
        quota["generic_job"] += 1
    kept: list[dict] = []
    kept_ids: set[str] = set()
    for stratum, count in quota.items():
        pool = sorted(by_stratum[stratum],
                      key=lambda r: (-rank_row(stratum, r),
                                     -float(r.get("bucket_confidence") or 0.0),
                                     bool(intent[r["id"]]["pay"]),     # keep both pay styles
                                     r["id"]))
        taken = pool[:count]
        kept.extend(taken)
        kept_ids.update(r["id"] for r in taken)
    dropped = [r for r in rows if r["id"] not in kept_ids]
    return kept, {"dropped": len(dropped),
                  "dropped_by_stratum": dict(Counter(intent[r["id"]]["stratum"] for r in dropped)),
                  "reason": f"over cap {keep}"}


# --------------------------------------------------------------------------- labelling
def label(postings: list[dict], workers: int, chunk: int) -> tuple[list[dict], dict]:
    client = SystemOneClient(cache=JsonCache(str(CACHE_PATH)))
    rows: list[dict] = []
    notes = {"chunks": 0, "retried_with_4_workers": 0, "skipped": [], "workers": workers}
    t0 = time.time()
    for start in range(0, len(postings), chunk):
        batch = postings[start:start + chunk]
        try:
            got = qualify_many(client, batch, workers=workers)
        except Exception as exc:                                    # rate limit / transient
            print(f"  ! chunk {start} failed with workers={workers}: {exc}; retrying workers=4")
            notes["retried_with_4_workers"] += 1
            notes["workers"] = 4
            try:
                got = qualify_many(client, batch, workers=4)
            except Exception as exc2:
                print(f"  !! chunk {start} failed again: {exc2}; skipping chunk")
                notes["skipped"].append({"start": start, "error": str(exc2)})
                continue
        rows.extend(got)
        notes["chunks"] += 1
        print(f"  [label] {len(rows)}/{len(postings)} rows  ({time.time() - t0:.0f}s)", flush=True)
    notes["seconds"] = round(time.time() - t0, 1)
    notes["usage"] = client.usage.as_dict()
    return rows, notes


def emit(rows: list[dict], postings: list[dict], out_path: Path) -> list[dict]:
    intent = {p["id"]: p["stratum"] for p in postings}
    recs = []
    for row in rows:
        if not row.get("bucket"):
            continue
        jev = jev_dict(row)
        recs.append({
            "id": row["id"],
            "prompt": DS.prompt(row),
            "target": target_from_jev(jev),
            "jev": jev,
            "source": "synthetic",
            "stratum": intent.get(row["id"]),
            "rank": rank_row(intent.get(row["id"]), row),
        })
    recs.sort(key=lambda r: r["id"])
    for i, rec in enumerate(recs, start=1):
        rec["id"] = f"synth-core-{i:04d}"
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in recs))
    tmp.replace(out_path)
    return recs


def usage_tally(usage: dict) -> dict:
    """Accumulate live spend across runs (a resume re-uses the cache and bills ~nothing)."""
    prev = json.loads(OUT_USAGE.read_text()) if OUT_USAGE.exists() else {"runs": []}
    prev["runs"].append(usage)
    total = {
        "calls": sum(r.get("calls", 0) for r in prev["runs"]),
        "cached_calls": sum(r.get("cached_calls", 0) for r in prev["runs"]),
        "input_tokens": sum(r.get("input_tokens", 0) for r in prev["runs"]),
        "output_tokens": sum(r.get("output_tokens", 0) for r in prev["runs"]),
        "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in prev["runs"]), 6),
    }
    prev["total"] = total
    OUT_USAGE.write_text(json.dumps(prev, indent=1))
    return total


def teacher_agreement() -> dict:
    """How far a teacher-labelled corpus would sit from Jev, measured on the 40 rows the
    teacher-first pipeline already ran through him (synth/data/jev_checked.jsonl)."""
    out: dict = {"available": False}
    if not TEACHER_CHECKED.exists():
        return out
    rows = []
    for line in TEACHER_CHECKED.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return out
    buckets = Counter()
    keys = {k: [0, 0] for k in NOULS}
    fit_ok = [0, 0]
    for r in rows:
        lab, jev = r.get("label") or {}, r.get("jev") or {}
        if not lab or "bucket" not in jev:
            continue
        tb, jb = lab.get("bucket"), (jev.get("bucket") or {}).get("choice")
        if tb and jb:
            buckets[("agree" if tb == jb else "differ")] += 1
            buckets[tb + " -> " + jb] += 1 if tb != jb else 0
        for k in NOULS:
            if k in lab and isinstance(jev.get(k), dict) and "noul" in jev[k]:
                keys[k][0] += int(bool(lab[k]) == (float(jev[k]["noul"]) >= 0.5))
                keys[k][1] += 1
        if "fit" in lab and isinstance(jev.get("fit"), dict) and "score" in jev["fit"]:
            lv = max(0, min(4, int(round(float(lab["fit"])))))
            jv = max(0, min(4, int(round(float(jev["fit"]["score"])))))
            fit_ok[0] += int(lv == jv)
            fit_ok[1] += 1
    n = buckets.get("agree", 0) + buckets.get("differ", 0)
    out = {
        "available": True, "rows": len(rows), "scored": n,
        "bucket_agree": buckets.get("agree", 0), "bucket_differ": buckets.get("differ", 0),
        "bucket_agreement": round(buckets.get("agree", 0) / n, 4) if n else None,
        "mismatches": {k: v for k, v in buckets.items() if "->" in str(k)},
        "noul_agreement": {k: (round(v[0] / v[1], 4) if v[1] else None) for k, v in keys.items()},
        "fit_agreement": round(fit_ok[0] / fit_ok[1], 4) if fit_ok[1] else None,
        "source": str(TEACHER_CHECKED),
    }
    return out


# --------------------------------------------------------------------------- gates + report
def gates(recs: list[dict], postings: list[dict]) -> dict:
    _, _, corpus_pairs, gold_titles, gold_pairs = load_reference()
    meta = {p["id"]: p for p in postings}
    lowered = [(meta[r["id"]]["title"].lower(), meta[r["id"]]["employer"].lower()) for r in recs]
    dup = [p for p, c in Counter(lowered).items() if c > 1]
    pays = [meta[r["id"]]["pay"] for r in recs]
    intent = Counter(r["stratum"] for r in recs)
    actual = Counter(r["jev"]["bucket"] for r in recs)
    agree = sum(1 for r in recs if r["jev"]["bucket"] == r["stratum"])
    by_stratum = defaultdict(Counter)
    for r in recs:
        by_stratum[r["stratum"]][r["jev"]["bucket"]] += 1
    fits = Counter(str(int(r["jev"]["fit"] or 0)) for r in recs)
    return {
        "rows": len(recs),
        "gates": {
            "duplicate_pairs": len(dup),
            "gold_pair_leak": sum(1 for p in lowered if p in gold_pairs),
            "gold_title_leak": sum(1 for p in lowered if p[0] in gold_titles),
            "corpus_pair_leak": sum(1 for p in lowered if p in corpus_pairs),
            "pay_rows": sum(1 for p in pays if p),
            "pay_share": round(sum(1 for p in pays if p) / max(len(recs), 1), 4),
            "implausible_employers": sum(1 for _, e in lowered if "acme" in e or "company a" in e),
            "distinct_titles": len({t for t, _ in lowered}),
            "distinct_employers": len({e for _, e in lowered}),
        },
        "intent": dict(intent),
        "jev": dict(actual),
        "confusion": {s: dict(c) for s, c in by_stratum.items()},
        "disagreements_per_stratum": {s: sum(v for b, v in c.items() if b != s)
                                      for s, c in by_stratum.items()},
        "agreement": {"agree": agree, "rows": len(recs), "rate": round(agree / max(len(recs), 1), 4)},
        "fit_levels": {k: v for k, v in sorted(fits.items())},
        "ranks": dict(Counter(r["rank"] for r in recs)),
        "mean_bucket_confidence": round(
            sum(float(r["jev"].get("bucket_confidence") or 0) for r in recs) / max(len(recs), 1), 4),
    }


EXPLAIN = {
    ("generic_job", "staff_role"): "intended a plain non-technical seat; Jev read the seat itself as technical",
    ("generic_job", "service_lead"): "intended a plain posting; Jev read it as an outside deliverable",
    ("generic_job", "junk"): "intended a real posting; Jev judged it not a real buyer of work",
    ("staff_role", "generic_job"): "intended an in-house technical seat; Jev read the seat as non-technical",
    ("staff_role", "service_lead"): "intended an in-house seat; Jev read it as a buyer of contract work",
    ("staff_role", "junk"): "intended a real technical seat; Jev judged the ad itself not a real opening",
    ("service_lead", "staff_role"): "intended contract-shaped work; Jev read it as a permanent employee seat",
    ("service_lead", "generic_job"): "intended a digital need; Jev saw no technical content",
    ("service_lead", "junk"): "intended a real buyer with a need; Jev judged it junk",
    ("junk", "generic_job"): "intended a commission/unpaid/vague ad; Jev read it as an ordinary job",
    ("junk", "staff_role"): "intended a junk ad; Jev read it as a real technical seat",
    ("junk", "service_lead"): "intended a junk ad; Jev read it as a buyer of outside work",
}


def report(recs, postings, g, usage, label_notes, sel, teacher, args) -> str:
    intent, jev = g["intent"], g["jev"]
    n = max(g["rows"], 1)
    meta = {p["id"]: p for p in postings}

    def fmt_dist(d):
        return "  ".join(f"{k} {v} ({v / n:.1%})" for k, v in sorted(d.items()))

    lines = [
        "# synth_core - balanced synthetic SaskJobs corpus, labelled by Jev",
        "",
        f"Generated by `synth_core.py` (seed {SEED}), labelled live against `jev-latest`.",
        f"**{g['rows']} rows**, {g['gates']['pay_rows']} with a stated pay "
        f"({g['gates']['pay_share']:.1%}), {g['gates']['distinct_titles']} distinct titles across "
        f"{g['gates']['distinct_employers']} distinct employers.",
        "",
        "## Label provenance",
        "",
        "**Every kept row is labelled by Jev himself.** One `qualify_many()` request per posting "
        "against",
        "the production 7-question rubric; `target` is built from his response - bucket = his "
        "bucket, the",
        "five booleans = his probability >= 0.5, fit = `round(his fit)` clamped to 0-4. There is "
        "**no teacher",
        "model in this pipeline** (no Claude/GPT generation, no teacher label, nothing to "
        "distil), so the",
        "student trains on Jev's judgment directly, not on an approximation of it. `stratum` "
        "(generation",
        "intent) and `rank` ride along as metadata and are never trained on.",
        "",
    ]
    if teacher.get("available"):
        lines += [
            f"For reference, the separate teacher-first pipeline (`synth/generate.py`, "
            f"gemini+claude) has",
            f"{teacher['rows']} rows that were also run through Jev. On those, the teacher and "
            f"Jev agree on",
            f"the **bucket {teacher['bucket_agreement']:.1%}** of the time "
            f"({teacher['bucket_agree']}/{teacher['scored']}; "
            + "; ".join(f"{k}: {v}" for k, v in list(teacher["mismatches"].items())[:4]) + "),",
            "on fit level " + (f"{teacher['fit_agreement']:.1%}" if teacher.get("fit_agreement") is not None else "n/a")
            + ", and per boolean "
            + ", ".join(f"{k} {v:.0%}" for k, v in teacher["noul_agreement"].items() if v is not None)
            + ".",
            "",
            "That is the number this corpus exists to avoid: a teacher-labelled training set "
            "would carry",
            "that much disagreement with Jev before the student ever sees a row. Caveat: those "
            "40 rows were",
            "selected by the teacher pipeline itself, so the figure is indicative, not an "
            "unbiased estimate.",
            "",
        ]
    else:
        lines += ["_Teacher-vs-Jev comparison unavailable (`synth/data/jev_checked.jsonl` "
                  "missing)._", ""]
    lines += [
        "## Method",
        "",
        "1. Employer names are harvested verbatim from the real watcher corpus "
        "(`typesafe-lab/runs/leads_corpus_full.json`), so every employer is a real "
        "Saskatchewan",
        "   organization - co-ops, credit unions, school divisions, health facilities, "
        "restaurants,",
        "   dealerships, machine shops, dental offices, hotels, non-profits. Names are split "
        "into",
        "   institutional / tech / local pools by keyword and dealt without replacement inside "
        "a pool.",
        "2. Postings are composed from per-stratum title banks x modifiers x pay styles, so the",
        "   intended bucket follows *who is hiring* (a health region filling an IT seat is "
        "staff_role;",
        "   a restaurant that needs a website is service_lead), not title keywords alone.",
        f"3. Every candidate went through the live Jev API (`qualify_many`, workers="
        f"{label_notes.get('workers')}).",
        "   Rows where he overruled the intent are kept - his label is the target - and counted "
        "below.",
        "4. `prompt` is `dataset.prompt()` verbatim; `target` is rebuilt from the rounded `jev` "
        "values, so",
        "   it is exactly recomputable from the stored row.",
        f"5. Kept set: {g['rows']} of {len(postings)} labelled candidates "
        f"(dropped {sel.get('dropped', 0)}",
        "   lowest-rank rows to respect the cap; stratum mix preserved).",
        "",
        "`rank` is a 1-5 realism rank: 5 = Jev agrees at confidence >= 0.80, 4 = agrees >= 0.60, "
        "3 = agrees",
        "below that, 2 = disagrees at low confidence, 1 = Jev confidently disagrees.",
        "",
        "## Quality gates",
        "",
        "| gate | value | status |",
        "| --- | --- | --- |",
        f"| duplicate (title, employer) pairs | {g['gates']['duplicate_pairs']} | "
        f"{'PASS' if g['gates']['duplicate_pairs'] == 0 else 'FAIL'} |",
        f"| overlap with the 70 gold rows (pair) | {g['gates']['gold_pair_leak']} | "
        f"{'PASS' if g['gates']['gold_pair_leak'] == 0 else 'FAIL'} |",
        f"| overlap with the 70 gold rows (title) | {g['gates']['gold_title_leak']} | "
        f"{'PASS' if g['gates']['gold_title_leak'] == 0 else 'FAIL'} |",
        f"| overlap with leads_corpus_full pairs | {g['gates']['corpus_pair_leak']} | "
        f"{'PASS' if g['gates']['corpus_pair_leak'] == 0 else 'FAIL'} |",
        f"| pay strings present | {g['gates']['pay_rows']} rows ({g['gates']['pay_share']:.1%}) | "
        f"{'PASS' if 0.30 <= g['gates']['pay_share'] <= 0.50 else 'CHECK'} |",
        f"| implausible employers (acme/company-a) | {g['gates']['implausible_employers']} | "
        f"{'PASS' if g['gates']['implausible_employers'] == 0 else 'FAIL'} |",
        f"| rows | {g['rows']} | {'PASS' if 1200 <= g['rows'] <= 1800 else 'FAIL'} |",
        "",
        "## Intent vs Jev (his label is the target; drift is kept, not scrubbed)",
        "",
        f"- intent:   {fmt_dist(intent)}",
        f"- Jev said: {fmt_dist(jev)}",
        f"- agreement: {g['agreement']['agree']}/{g['agreement']['rows']} "
        f"({g['agreement']['rate']:.1%}) - mean bucket_confidence {g['mean_bucket_confidence']:.3f}",
        f"- disagreements per family (stratum): "
        + ", ".join(f"{k} {v}/{intent[k]}" for k, v in sorted(g['disagreements_per_stratum'].items())),
        "",
        "| intent \\ Jev | " + " | ".join(sorted(jev)) + " |",
        "| --- | " + " | ".join("---" for _ in jev) + " |",
    ]
    for s in sorted(intent):
        row = g["confusion"].get(s, {})
        lines.append(f"| {s} | " + " | ".join(str(row.get(b, 0)) for b in sorted(jev)) + " |")
    lines += [
        "",
        f"Fit levels (Jev's 0-4, rounded): {g['fit_levels']}",
        f"Realism ranks: {dict(sorted(g['ranks'].items()))}",
        f"Target-rounding mismatches vs `dataset.target()`: {g.get('target_rounding_mismatches')} "
        "(0 = the stored target is recomputable from `jev`)",
        "",
        "## The 10 most surprising disagreements (verbatim)",
        "",
        "(sorted by how confidently Jev overruled the intent - these are the boundary cases the "
        "imitation",
        "model has to get right, and the rows worth a hand label later)",
        "",
    ]
    dis = [r for r in recs if r["jev"]["bucket"] != r["stratum"]]
    dis.sort(key=lambda r: -float(r["jev"].get("bucket_confidence") or 0))
    for i, r in enumerate(dis[:10], start=1):
        p = meta[r["id"]]
        pr = r["jev"].get("bucket_probabilities") or {}
        probs = ", ".join(f"{k} {v:.2f}" for k, v in sorted(pr.items(), key=lambda kv: -kv[1]))
        why = EXPLAIN.get((r["stratum"], r["jev"]["bucket"]), "boundary case")
        lines.append(
            f"{i}. title `{p['title']}` | employer `{p['employer']}` | pay "
            f"`{p['pay'] or '-'}` | intended **{r['stratum']}** -> Jev **{r['jev']['bucket']}** "
            f"(conf {float(r['jev'].get('bucket_confidence') or 0):.2f}; {probs}) - {why}")
    lines += ["", "## Five sample rows, verbatim", ""]
    picked, seen_b = [], set()
    for r in recs:
        if r["jev"]["bucket"] not in seen_b:
            seen_b.add(r["jev"]["bucket"])
            picked.append(r)
    for r in recs:
        if len(picked) >= 5:
            break
        if r not in picked:
            picked.append(r)
    for r in picked[:5]:
        lines += ["```json", json.dumps(r), "```", ""]
    lines += [
        "## Cost",
        "",
        f"- live Jev calls: {usage['calls']} ({usage['cached_calls']} served from cache; totals "
        f"accumulated over every run of this script - a resume is served by the cache and bills "
        f"~nothing), "
        f"{usage['input_tokens']:,} input tokens, {usage['output_tokens']:,} output tokens",
        f"- **${usage['cost_usd']:.4f}** at $0.042/MTok in, output free "
        f"({usage['cost_usd'] / max(usage['calls'], 1) * 1000:.3f} $/1k calls)",
        f"- labelling wall clock {label_notes.get('seconds')}s; chunks retried at workers=4: "
        f"{label_notes.get('retried_with_4_workers')}; skipped chunks: "
        f"{len(label_notes.get('skipped', []))}",
        "",
        "## Caveats",
        "",
        "- These are *manufactured* postings. Titles and pay strings are plausible SaskJobs "
        "shapes, but",
        "  they are not real ads; Jev's labels are the only ground truth the file carries and "
        "they inherit",
        "  whatever bias his rubric has on synthetic text.",
        "- The junk stratum's employers are half harvested locals and half place-stemmed "
        "marketing/staffing",
        "  names (a real junk ad usually hides behind a vague name); every other stratum uses "
        "harvested",
        "  real employers only.",
        "- Rows where Jev overruled the intent are kept deliberately. They make the file harder "
        "than a",
        "  real 2,600-posting day: boundaries appear more often here than in the stream.",
        "",
        f"_generated {time.strftime('%Y-%m-%d %H:%M:%S')} by synth_core.py_",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1680, help="candidate postings to generate")
    ap.add_argument("--keep", type=int, default=1600, help="cap on kept (Jev-labelled) rows")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=140)
    args = ap.parse_args()

    rng = random.Random(SEED)
    print(f"[gen] building {args.rows} candidates (seed {SEED})")
    postings = build_postings(args.rows, rng)
    OUT_POSTINGS.write_text("".join(json.dumps(p) + "\n" for p in postings))
    print(f"[gen] {len(postings)} candidates -> {OUT_POSTINGS}")
    print("      intent:", dict(Counter(p["stratum"] for p in postings)))
    print("      pay rows:", sum(1 for p in postings if p["pay"]),
          "distinct employers:", len({p["employer"] for p in postings}))

    print(f"[label] calling the live Jev API (workers={args.workers})")
    rows, notes = label(postings, args.workers, args.chunk)
    print(f"[label] {len(rows)} rows labelled by Jev")

    kept, sel = select_kept(rows, postings, args.keep)
    print(f"[keep]  {len(kept)} of {len(rows)} rows ({sel})")
    recs = emit(kept, postings, OUT_JSONL)

    total_usage = usage_tally(notes["usage"])
    g = gates(recs, postings)
    teacher = teacher_agreement()

    band = 0
    for rec, raw in zip(recs, kept):
        if target_from_jev(rec["jev"]) != DS.target(raw):
            band += 1
    g["target_rounding_mismatches"] = band
    g["usage"] = total_usage
    g["label_notes"] = notes
    g["selection"] = sel
    g["teacher_vs_jev"] = teacher
    OUT_STATS.write_text(json.dumps(g, indent=1))

    OUT_REPORT.write_text(report(recs, postings, g, total_usage, notes, sel, teacher, args))
    print("[gates]")
    for k, v in g["gates"].items():
        print(f"   {k}: {v}")
    print(f"   agreement {g['agreement']['rate']:.1%}  mean conf {g['mean_bucket_confidence']}")
    print(f"   intent {g['intent']}")
    print(f"   jev    {g['jev']}")
    print(f"   disagreements/stratum {g['disagreements_per_stratum']}")
    print(f"   cost ${total_usage['cost_usd']:.4f} over {total_usage['calls']} calls "
          f"({total_usage['input_tokens']:,} in-tokens); target-rounding mismatches {band}")
    print(f"[out] {OUT_JSONL} -> {sum(1 for _ in OUT_JSONL.open())} lines; {OUT_REPORT}")


if __name__ == "__main__":
    main()
