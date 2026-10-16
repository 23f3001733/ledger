"""Acme Ledger Agent: LLM understands the question, Python does the maths."""
import os, json, datetime as dt
from decimal import Decimal, ROUND_HALF_UP
from zoneinfo import ZoneInfo
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

IST = ZoneInfo("Asia/Kolkata")
EXPORT_URL = os.getenv("LEDGER_EXPORT_URL", "")   # 3rd link (the file download)
RATES_URL = os.getenv("LEDGER_RATES_URL", "")     # 2nd link (rates JSON)
LLM_URL = os.getenv("LLM_BASE_URL", "https://aipipe.org/openai/v1") + "/chat/completions"
LLM_KEY = os.getenv("AIPIPE_TOKEN") or os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")

# ---------- STEP 1: load + clean data once, at startup ----------
def parse_ts(s):                      # handles both "...Z" and "...+05:30"
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))

def load_rows():
    text = ""
    path = os.path.join(os.path.dirname(__file__), "data.ndjson")
    if os.path.exists(path):                      # fast + reliable: file shipped in repo
        text = open(path).read()
    if not text.strip() and EXPORT_URL:           # otherwise download the export
        text = httpx.get(EXPORT_URL, timeout=30).text
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    latest = {}
    for r in rows:                                # same id twice -> keep latest updated_at
        t = parse_ts(r["updated_at"])
        if r["id"] not in latest or t >= latest[r["id"]][0]:
            latest[r["id"]] = (t, r)
    out = []
    for _, r in latest.values():
        r["ist_date"] = parse_ts(r["created_at"]).astimezone(IST).date()
        r["ist_upd"] = parse_ts(r["updated_at"]).astimezone(IST).date()
        out.append(r)
    return out

def load_rates():
    # Your /rates API gives: {"usd_per_unit": {"USD": 1, "EUR": 1.05, "INR": 0.01111}}
    rates = {"USD": 1.0, "EUR": 1.05, "INR": 0.01111}   # safe fallback
    if RATES_URL:
        try:
            j = httpx.get(RATES_URL, timeout=5).json()
            for k, v in j["usd_per_unit"].items():
                rates[k.upper()] = float(v)
        except Exception:
            pass
    return rates

ROWS, RATES = load_rows(), load_rates()
PRODUCTS = sorted({r["product"] for r in ROWS})
REGIONS = sorted({r["region"] for r in ROWS})

def usd(r): return Decimal(str(r["amount"])) * Decimal(str(RATES[r["currency"]]))

def cents(x): return float(Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# ---------- STEP 2: LLM turns English into a small JSON "spec" ----------
SYSTEM = f"""Convert the question into JSON only (no prose). Today is {dt.date.today()}.
Fields (use null if not mentioned):
metric: revenue | refund_amount | order_count | refund_count | units | avg_order_value | unique_customers
region: one of {REGIONS}
product: one of {PRODUCTS} (spelled exactly)
customer: id like C1011
start_date, end_date: YYYY-MM-DD inclusive (a month = first..last day; a quarter likewise)
group_by: null | region | product | customer | month   (use when asked "which"/"top"/"per")
top_n: integer or null ; order: "desc" or "asc"
revenue = paid orders. refund_amount/refund_count = refunded orders."""

def ask_llm(q):
    r = httpx.post(LLM_URL, timeout=9,
        headers={"Authorization": f"Bearer {LLM_KEY}"},
        json={"model": MODEL, "temperature": 0,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": q}]})
    return json.loads(r.json()["choices"][0]["message"]["content"])


# ---------- STEP 2b: rule-based reader (instant, no LLM) ----------
import re, calendar
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
LLM_ONLY = ("percent", "%", "ratio", "compare", "difference", "more than", "less than",
            "exceed", "above", "below", "list", "who ", "growth", "change", "increase", "decrease")

def month_range(y, m):
    return f"{y}-{m:02d}-01", f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"

def rule_parse(q):
    t = q.lower()
    if any(w in t for w in LLM_ONLY): return None
    spec = {"metric": None, "region": None, "product": None, "customer": None,
            "start_date": None, "end_date": None, "group_by": None, "top_n": None, "order": "desc"}
    for r in REGIONS:
        if re.search(rf"\b{r.lower()}\b", t): spec["region"] = r
    for p in PRODUCTS:
        pl = p.lower()
        if re.search(rf"\b{pl}s?\b", t): spec["product"] = p
    m = re.search(r"\bc\d{4}\b", t)
    if m: spec["customer"] = m.group(0).upper()
    # ---- dates ----
    mon = "|".join(sorted(MONTHS, key=len, reverse=True))
    iso = re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    day1 = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({mon})[a-z]*,?\s+(\d{{4}})\b", t)
    day2 = re.search(rf"\b({mon})[a-z]*\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", t)
    mon_y = re.search(rf"\b({mon})[a-z]*,?\s+(\d{{4}})\b", t)
    qtr = re.search(r"\bq([1-4])\s*(?:of\s*)?(\d{4})\b", t)
    year = re.search(r"\b(20\d{2})\b", t)
    if len(iso) >= 2:
        spec["start_date"] = "-".join(iso[0]); spec["end_date"] = "-".join(iso[1])
    elif len(iso) == 1:
        spec["start_date"] = spec["end_date"] = "-".join(iso[0])
    elif day1:
        d, mo, y = int(day1.group(1)), MONTHS[day1.group(2)], int(day1.group(3))
        spec["start_date"] = spec["end_date"] = f"{y}-{mo:02d}-{d:02d}"
    elif day2:
        mo, d, y = MONTHS[day2.group(1)], int(day2.group(2)), int(day2.group(3))
        spec["start_date"] = spec["end_date"] = f"{y}-{mo:02d}-{d:02d}"
    elif qtr:
        n, y = int(qtr.group(1)), int(qtr.group(2))
        spec["start_date"] = month_range(y, 3 * n - 2)[0]; spec["end_date"] = month_range(y, 3 * n)[1]
    elif mon_y:
        spec["start_date"], spec["end_date"] = month_range(int(mon_y.group(2)), MONTHS[mon_y.group(1)])
    elif year:
        spec["start_date"], spec["end_date"] = f"{year.group(1)}-01-01", f"{year.group(1)}-12-31"
    elif re.search(rf"\b({mon})\b", t):          # month without year -> data year
        mo = MONTHS[re.search(rf"\b({mon})\b", t).group(1)]
        spec["start_date"], spec["end_date"] = month_range(2026, mo)
    if re.search(r"\b(between|from)\b.*\b(and|to)\b", t) and len(iso) < 2 and not (spec["start_date"] and mon_y):
        pass
    # ---- metric ----
    if "refund" in t:
        spec["metric"] = "refund_count" if re.search(r"how many|number of|count", t) else "refund_amount"
    elif re.search(r"average|avg|mean", t): spec["metric"] = "avg_order_value"
    elif re.search(r"unique|distinct|different", t) and "customer" in t: spec["metric"] = "unique_customers"
    elif re.search(r"how many customers|number of customers", t): spec["metric"] = "unique_customers"
    elif re.search(r"units|quantity|pieces|items", t): spec["metric"] = "units"
    elif re.search(r"how many|number of|count", t) and "order" in t: spec["metric"] = "order_count"
    elif re.search(r"revenue|earn|sales|sale|income|sold|spent|spend|money|made|total|usd|dollar|\$|worth|gross|turnover", t):
        spec["metric"] = "revenue"
    else:
        return None
    # ---- which / top ----
    g = re.search(r"\b(?:which|top\s*\d*|best|highest|largest|biggest|most|lowest|least|fewest|worst)\s+(?:\w+\s+){0,2}?(region|product|customer|month)s?\b", t) \
        or re.search(r"\bwhat\s+(region|product|customer|month)s?\b", t)
    if g and g.group(1) == "customer" and spec["customer"]: g = None   # a specific customer was named
    if g:
        spec["group_by"] = g.group(1)
        n = re.search(r"top\s*(\d+)", t)
        spec["top_n"] = int(n.group(1)) if n else 1
        if re.search(r"lowest|least|fewest|worst|smallest|bottom", t): spec["order"] = "asc"
        if spec["group_by"] == "region": spec["region"] = None
        if spec["group_by"] == "product": spec["product"] = None
        if spec["group_by"] == "customer": spec["customer"] = None
    return spec

def solve(q):
    spec = rule_parse(q)
    if spec is None: spec = ask_llm(q)
    return compute(spec)

# ---------- STEP 3: plain Python does the exact maths ----------
def compute(s):
    metric = s.get("metric") or "revenue"
    status = "refunded" if metric.startswith("refund") else "paid"
    a = dt.date.fromisoformat(s["start_date"]) if s.get("start_date") else dt.date.min
    b = dt.date.fromisoformat(s["end_date"]) if s.get("end_date") else dt.date.max
    sel = [r for r in ROWS if r["status"] == status
           and a <= r["ist_date"] <= b
           and (not s.get("region") or r["region"].lower() == s["region"].lower())
           and (not s.get("product") or r["product"].lower() == s["product"].lower())
           and (not s.get("customer") or r["customer"].upper() == s["customer"].upper())]

    def agg(rs):
        if metric in ("revenue", "refund_amount"): return cents(sum((usd(r) for r in rs), Decimal(0)))
        if metric in ("order_count", "refund_count"): return len(rs)
        if metric == "units": return sum(r["qty"] for r in rs)
        if metric == "unique_customers": return len({r["customer"] for r in rs})
        if metric == "avg_order_value": return cents(sum((usd(r) for r in rs), Decimal(0)) / len(rs)) if rs else 0
    g = s.get("group_by")
    if not g: return agg(sel)
    groups = {}
    for r in sel:
        k = r["ist_date"].strftime("%Y-%m") if g == "month" else r[g]
        groups.setdefault(k, []).append(r)
    res = sorted(((k, agg(v)) for k, v in groups.items()),
                 key=lambda x: x[1], reverse=(s.get("order") != "asc"))
    n = s.get("top_n") or 1
    return res[0][0] if n == 1 else [k for k, _ in res[:n]]

# ---------- STEP 4: the web endpoint ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health(): return {"ok": True, "rows": len(ROWS)}

@app.post("/")
@app.post("/ask")
async def ask(req: Request):
    q = (await req.json()).get("question", "")
    try:
        return {"answer": solve(q)}
    except Exception as e:
        return {"answer": None, "error": str(e)}
