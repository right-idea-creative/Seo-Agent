#!/usr/bin/env python3
"""
wp_compare.py — Compare two WordPress posts side by side.

Compares:
  1. All wp_posts columns (except naturally-varying ones)
  2. All wp_postmeta rows
  3. All taxonomies and term relationships

Produces:
  ✓  identical
  ✗  different value
  △  key/column exists only in one post

No fixes are applied. The analysis section lists differences and
classifies each one; the decision of what to change stays with you.

Usage:
    python3 wp_compare.py MANUAL_ID AGENT_ID --wpconfig /path/to/wp-config.php
    python3 wp_compare.py MANUAL_ID AGENT_ID --host localhost --user u --passwd p --db dbname

Requirements:
    pip install pymysql
"""

import sys
import re
import argparse
from pathlib import Path

# ── colors ─────────────────────────────────────────────────────────────────────
G = "\033[32m"   # green
R = "\033[31m"   # red
Y = "\033[33m"   # yellow
C = "\033[36m"   # cyan
B = "\033[1m"    # bold
DIM = "\033[2m"  # dim
X = "\033[0m"    # reset


def trunc(v, n=70):
    s = str(v) if v is not None else "NULL"
    return (s[:n] + "…") if len(s) > n else s


def header(title, num, total):
    print(f"\n{B}{C}{'═' * 65}{X}")
    print(f"{B}{C}  [{num}/{total}]  {title}{X}")
    print(f"{B}{C}{'═' * 65}{X}")


def row_ok(label, value=""):
    print(f"  {G}✓{X}  {label:<48} {DIM}{trunc(value, 30)}{X}")


def row_diff(label, manual_val, agent_val):
    print(f"  {R}✗  {B}{label}{X}")
    print(f"       {Y}manual:{X} {trunc(manual_val, 58)}")
    print(f"       {R}agent: {X} {trunc(agent_val,  58)}")


def row_only(side, label, value):
    color = Y if side == "manual" else R
    arrow = "△" if side == "manual" else "▽"
    print(f"  {color}{arrow}  {B}{label}{X}  {DIM}→ only in {side}{X}")
    print(f"       value: {trunc(value, 62)}")


# ── wp-config.php parser ───────────────────────────────────────────────────────
def parse_wpconfig(path: Path) -> dict:
    txt = path.read_text(errors="replace")

    def get(key):
        m = re.search(
            rf"define\(\s*['\"]DB_{key}['\"]\s*,\s*['\"]([^'\"]*)['\"]", txt
        )
        return m.group(1) if m else None

    m = re.search(r"\$table_prefix\s*=\s*['\"]([^'\"]+)['\"]", txt)
    prefix = m.group(1) if m else "wp_"
    return {
        "host":   get("HOST") or "localhost",
        "user":   get("USER"),
        "passwd": get("PASSWORD"),
        "db":     get("NAME"),
        "prefix": prefix,
    }


# ── DB connection ──────────────────────────────────────────────────────────────
def connect(creds: dict):
    try:
        import pymysql
        import pymysql.cursors
        return pymysql.connect(
            host=creds["host"],
            user=creds["user"],
            password=creds["passwd"],
            database=creds["db"],
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
    except ImportError:
        sys.exit(
            f"{R}ERROR:{X} pymysql not installed.\n"
            "Run:  pip install pymysql"
        )


# ── collector: accumulates all differences for final analysis ──────────────────
class Diff:
    """Lightweight difference record."""
    __slots__ = ("section", "key", "manual_val", "agent_val", "kind")

    def __init__(self, section, key, manual_val, agent_val, kind):
        # kind: "diff" | "only_manual" | "only_agent"
        self.section    = section
        self.key        = key
        self.manual_val = manual_val
        self.agent_val  = agent_val
        self.kind       = kind


diffs: list[Diff] = []


# ── Section 1: wp_posts ────────────────────────────────────────────────────────
# Columns excluded because they change naturally and carry no structural meaning.
POSTS_SKIP = {
    "ID",
    "post_date", "post_date_gmt",
    "post_modified", "post_modified_gmt",
    "guid",
    "post_name",   # slug — intentionally different per article
    "post_title",  # intentionally different per article
    "post_content",
    "post_excerpt",
}


def compare_posts(cur, pfx: str, mid: int, aid: int) -> None:
    header("wp_posts  (excluding: content, dates, ID, title, slug, guid)", 1, 3)

    cur.execute(
        f"SELECT * FROM {pfx}posts WHERE ID IN (%s, %s) ORDER BY ID",
        (mid, aid),
    )
    rows = {int(r["ID"]): r for r in cur.fetchall()}
    mr = rows.get(mid)
    ar = rows.get(aid)
    if mr is None:
        sys.exit(f"{R}ERROR:{X} Post {mid} not found.")
    if ar is None:
        sys.exit(f"{R}ERROR:{X} Post {aid} not found.")

    count_diff = 0
    for col in mr:
        if col in POSTS_SKIP:
            continue
        mv, av = mr[col], ar[col]
        if mv == av:
            row_ok(col, mv)
        else:
            count_diff += 1
            row_diff(col, mv, av)
            diffs.append(Diff("wp_posts", col, mv, av, "diff"))

    status = f"{R}{count_diff} difference(s){X}" if count_diff else f"{G}all compared columns identical{X}"
    print(f"\n  → {status}")


# ── Section 2: wp_postmeta ─────────────────────────────────────────────────────
def compare_postmeta(cur, pfx: str, mid: int, aid: int) -> None:
    header("wp_postmeta  (all rows, sorted by meta_key)", 2, 3)

    cur.execute(
        f"SELECT meta_key, meta_value FROM {pfx}postmeta "
        f"WHERE post_id = %s ORDER BY meta_key",
        (mid,),
    )
    mm = {r["meta_key"]: r["meta_value"] for r in cur.fetchall()}

    cur.execute(
        f"SELECT meta_key, meta_value FROM {pfx}postmeta "
        f"WHERE post_id = %s ORDER BY meta_key",
        (aid,),
    )
    am = {r["meta_key"]: r["meta_value"] for r in cur.fetchall()}

    all_keys = sorted(set(mm) | set(am))
    count_diff = only_m = only_a = 0

    for k in all_keys:
        in_m, in_a = k in mm, k in am

        if in_m and not in_a:
            only_m += 1
            count_diff += 1
            row_only("manual", k, mm[k])
            diffs.append(Diff("wp_postmeta", k, mm[k], None, "only_manual"))

        elif in_a and not in_m:
            only_a += 1
            count_diff += 1
            row_only("agent", k, am[k])
            diffs.append(Diff("wp_postmeta", k, None, am[k], "only_agent"))

        else:
            mv, av = mm[k], am[k]
            if mv == av:
                row_ok(k, mv)
            else:
                count_diff += 1
                row_diff(k, mv, av)
                diffs.append(Diff("wp_postmeta", k, mv, av, "diff"))

    if count_diff == 0:
        print(f"\n  {G}→ wp_postmeta: all keys identical{X}")
    else:
        print(
            f"\n  {R}→ wp_postmeta: {count_diff} difference(s) "
            f"({only_m} only in manual, {only_a} only in agent){X}"
        )


# ── Section 3: taxonomies and term relationships ───────────────────────────────
def compare_taxonomies(cur, pfx: str, mid: int, aid: int) -> None:
    header("Taxonomies  (wp_term_relationships + wp_term_taxonomy + wp_terms)", 3, 3)

    cur.execute(
        f"""
        SELECT
            p.ID                AS post_id,
            tt.taxonomy         AS taxonomy,
            t.slug              AS term_slug,
            t.name              AS term_name,
            t.term_id           AS term_id,
            tt.term_taxonomy_id AS term_taxonomy_id
        FROM {pfx}posts p
        JOIN {pfx}term_relationships tr ON p.ID = tr.object_id
        JOIN {pfx}term_taxonomy tt ON tr.term_taxonomy_id = tt.term_taxonomy_id
        JOIN {pfx}terms t ON tt.term_id = t.term_id
        WHERE p.ID IN (%s, %s)
        ORDER BY tt.taxonomy, t.slug
        """,
        (mid, aid),
    )
    rows = cur.fetchall()

    # Build sets of (taxonomy, term_slug) per post
    manual_terms: dict[str, set] = {}   # taxonomy → set of slugs
    agent_terms:  dict[str, set] = {}

    for r in rows:
        pid = int(r["post_id"])
        tax = r["taxonomy"]
        slug = r["term_slug"]
        target = manual_terms if pid == mid else agent_terms
        target.setdefault(tax, set()).add(slug)

    all_taxonomies = sorted(set(manual_terms) | set(agent_terms))

    if not all_taxonomies:
        print(f"  {DIM}No term relationships found for either post.{X}")
        return

    count_diff = 0
    for tax in all_taxonomies:
        m_slugs = manual_terms.get(tax, set())
        a_slugs = agent_terms.get(tax, set())

        if m_slugs == a_slugs:
            row_ok(f"[{tax}]", ", ".join(sorted(m_slugs)))
        else:
            count_diff += 1
            only_in_manual = m_slugs - a_slugs
            only_in_agent  = a_slugs - m_slugs
            shared         = m_slugs & a_slugs

            print(f"  {R}✗  {B}[{tax}]{X}")
            if shared:
                print(f"       {G}both:{X}          {', '.join(sorted(shared))}")
            if only_in_manual:
                print(f"       {Y}only manual:{X}   {', '.join(sorted(only_in_manual))}")
                for slug in sorted(only_in_manual):
                    diffs.append(Diff("taxonomy", f"{tax}:{slug}", slug, None, "only_manual"))
            if only_in_agent:
                print(f"       {R}only agent:{X}    {', '.join(sorted(only_in_agent))}")
                for slug in sorted(only_in_agent):
                    diffs.append(Diff("taxonomy", f"{tax}:{slug}", None, slug, "only_agent"))

    status = f"{R}{count_diff} taxonomy/taxonomies differ{X}" if count_diff else f"{G}all taxonomies identical{X}"
    print(f"\n  → {status}")


# ── Analysis: classify every difference found ──────────────────────────────────
#
# Classification categories:
#   ELEMENTOR  — could directly affect whether Elementor Theme Builder applies
#   WORDPRESS  — normal WordPress behavior, no structural impact on templates
#   NEUTRAL    — expected to differ (content, IDs, SEO slugs, etc.)
#   UNKNOWN    — no clear classification without more context
#
# This section does NOT suggest fixes. It only explains what each difference is.

CLASSIFICATION: dict[str, tuple[str, str]] = {
    # (category, explanation)

    # ── wp_posts columns ───────────────────────────────────────────────────────
    "post_status":        ("WORDPRESS", "Both should be the same if published; may differ during testing."),
    "post_type":          ("ELEMENTOR", "Rare. If it differs, template conditions by post type would break."),
    "post_author":        ("WORDPRESS", "Different user IDs. No impact on Elementor Theme Builder conditions."),
    "post_parent":        ("WORDPRESS", "Used for page hierarchies, not posts. Normally 0 for both."),
    "menu_order":         ("WORDPRESS", "Ordering field. No template impact."),
    "post_mime_type":     ("WORDPRESS", "Empty for posts. No template impact."),
    "comment_count":      ("NEUTRAL",   "Count only; changes over time."),
    "comment_status":     ("WORDPRESS", "open/closed. WordPress admin defaults differ from REST API defaults."),
    "ping_status":        ("WORDPRESS", "open/closed. Same as comment_status — REST API may differ."),
    "post_password":      ("WORDPRESS", "Password-protected posts. Normally empty for both."),
    "to_ping":            ("WORDPRESS", "Pingback URLs queue. Normally empty."),
    "pinged":             ("WORDPRESS", "Already-pinged URLs. Normally empty."),
    "post_content_filtered": ("WORDPRESS", "Filtered content cache. Normally empty."),

    # ── wp_postmeta keys ───────────────────────────────────────────────────────
    "_elementor_edit_mode":       ("ELEMENTOR", "Elementor uses this internally. Its presence or value may affect template routing."),
    "_elementor_data":            ("ELEMENTOR", "Elementor page layout JSON. Its presence marks the post as processed by Elementor."),
    "_elementor_version":         ("ELEMENTOR", "Elementor version that last saved this post."),
    "_elementor_template_type":   ("ELEMENTOR", "Elementor document type (e.g. 'post', 'page'). Affects rendering path."),
    "_elementor_page_settings":   ("ELEMENTOR", "Per-page settings override in Elementor."),
    "_elementor_page_assets":     ("ELEMENTOR", "Cached CSS/JS assets for this post. Rebuilt on next view."),
    "_elementor_controls_usage":  ("ELEMENTOR", "Internal usage tracking. Low probability of template impact."),
    "_elementor_css":             ("ELEMENTOR", "Per-post Elementor CSS cache. Rebuilt automatically."),
    "_wp_page_template":          ("ELEMENTOR", "WordPress page template meta. Admin sets 'default'; REST API may set empty string. Some themes check this before delegating to Elementor."),
    "_edit_last":                 ("WORDPRESS", "ID of the last user who saved from the admin editor. Expected to differ."),
    "_edit_lock":                 ("WORDPRESS", "Temporary editor lock. Not persistent. No template impact."),
    "_thumbnail_id":              ("WORDPRESS", "Featured image ID. Expected to differ (different media IDs per post)."),
    "_wp_trash_meta_status":      ("WORDPRESS", "Set when a post is trashed. Should not appear on published posts."),
    "_wp_trash_meta_time":        ("WORDPRESS", "Trashed timestamp. Should not appear on published posts."),
    "_pingme":                    ("WORDPRESS", "Triggers ping processing. Transient; no template impact."),
    "_encloseme":                 ("WORDPRESS", "Triggers enclosure processing. Transient; no template impact."),

    # ── Taxonomies ─────────────────────────────────────────────────────────────
    "category:*":     ("ELEMENTOR", "Elementor Theme Builder conditions use 'has_category()'. Both posts must share the same categories for the same conditions to apply."),
    "post_tag:*":     ("WORDPRESS", "Tags. Elementor Theme Builder can use tag conditions but they are optional."),
    "post_format:*":  ("ELEMENTOR", "Post format. WordPress admin always assigns 'standard'. If REST API skips it, the behavior may differ depending on the theme."),
}

# Keys that start with these prefixes get a default classification
PREFIX_RULES: list[tuple[str, tuple[str, str]]] = [
    ("_elementor_",       ("ELEMENTOR", "Elementor internal meta. Any difference here is relevant to investigate.")),
    ("rank_math_",        ("WORDPRESS", "Rank Math SEO meta. No template impact.")),
    ("_yoast_",           ("WORDPRESS", "Yoast SEO meta. No template impact.")),
    ("_aioseop_",         ("WORDPRESS", "All in One SEO meta. No template impact.")),
    ("wpseo_",            ("WORDPRESS", "Yoast SEO meta. No template impact.")),
    ("_wp_",              ("WORDPRESS", "Core WordPress meta. Likely no template impact unless it is _wp_page_template.")),
]


def classify(d: Diff) -> tuple[str, str]:
    key = d.key

    # Exact match
    if key in CLASSIFICATION:
        return CLASSIFICATION[key]

    # Taxonomy wildcards
    if d.section == "taxonomy":
        tax = key.split(":")[0]
        wildcard = f"{tax}:*"
        if wildcard in CLASSIFICATION:
            return CLASSIFICATION[wildcard]
        return ("UNKNOWN", "Taxonomy not in classification table. Review manually.")

    # Prefix match
    for prefix, result in PREFIX_RULES:
        if key.startswith(prefix):
            return result

    return ("UNKNOWN", "No classification rule found. Review manually.")


CATEGORY_COLORS = {
    "ELEMENTOR": R,
    "WORDPRESS":  Y,
    "NEUTRAL":    DIM,
    "UNKNOWN":    C,
}

CATEGORY_LABELS = {
    "ELEMENTOR": "could affect Elementor Template Builder",
    "WORDPRESS":  "normal WordPress behavior — no template impact expected",
    "NEUTRAL":    "naturally different (content, IDs, etc.)",
    "UNKNOWN":    "unclassified — review manually",
}


def analysis():
    print(f"\n{B}{'═' * 65}")
    print(f"  ANALYSIS{X}")
    print(f"{B}{'═' * 65}{X}")

    if not diffs:
        print(f"\n  {G}No differences found between the two posts.{X}")
        print(
            f"  All compared columns, postmeta keys, and taxonomies are identical.\n"
            f"  If Elementor Theme Builder still does not apply to the agent post,\n"
            f"  the cause is not in stored data — it is in PHP hooks or runtime\n"
            f"  conditions evaluated at request time (e.g. template_include filter,\n"
            f"  is_single(), has_category(), or a conditional check inside a plugin)."
        )
        return

    # Group by category
    grouped: dict[str, list[tuple[Diff, str]]] = {
        "ELEMENTOR": [],
        "WORDPRESS":  [],
        "NEUTRAL":    [],
        "UNKNOWN":    [],
    }

    for d in diffs:
        cat, explanation = classify(d)
        grouped[cat].append((d, explanation))

    for cat in ("ELEMENTOR", "WORDPRESS", "NEUTRAL", "UNKNOWN"):
        items = grouped[cat]
        if not items:
            continue
        color = CATEGORY_COLORS[cat]
        label = CATEGORY_LABELS[cat]
        print(f"\n  {color}{B}[{cat}]{X}  {DIM}{label}{X}")
        print(f"  {'─' * 60}")

        for d, explanation in items:
            # Header line
            kind_tag = {
                "diff":        f"{R}✗ differs{X}",
                "only_manual": f"{Y}△ only in manual{X}",
                "only_agent":  f"{R}▽ only in agent{X}",
            }[d.kind]

            print(f"\n  {color}•{X}  {B}{d.key}{X}  ({d.section})  {kind_tag}")
            if d.kind == "diff":
                print(f"       manual: {trunc(d.manual_val, 55)}")
                print(f"       agent:  {trunc(d.agent_val,  55)}")
            elif d.kind == "only_manual":
                print(f"       value:  {trunc(d.manual_val, 55)}")
            elif d.kind == "only_agent":
                print(f"       value:  {trunc(d.agent_val,  55)}")

            print(f"       {DIM}{explanation}{X}")

    # Summary
    el = len(grouped["ELEMENTOR"])
    total = len(diffs)
    print(f"\n{'─' * 65}")
    print(f"  Total differences : {total}")
    print(f"  → Elementor-relevant : {R if el else G}{el}{X}")
    print(f"  → Normal WordPress   : {len(grouped['WORDPRESS'])}")
    print(f"  → Neutral/expected   : {len(grouped['NEUTRAL'])}")
    print(f"  → Unclassified       : {len(grouped['UNKNOWN'])}")
    print(
        f"\n  {DIM}Classification is informational only. Review the ELEMENTOR group\n"
        f"  before deciding whether to modify the agent. A difference in that\n"
        f"  group is worth investigating; it does not automatically mean it\n"
        f"  is the cause.{X}\n"
    )


# ── entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compare two WordPress posts (wp_posts, wp_postmeta, taxonomies).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 wp_compare.py 42 87 --wpconfig /var/www/html/wp-config.php\n"
            "  python3 wp_compare.py 42 87 --host localhost --user u --passwd p --db mydb\n"
        ),
    )
    parser.add_argument("manual_id", type=int, help="Post ID of the manually-created post")
    parser.add_argument("agent_id",  type=int, help="Post ID of the agent-created post")
    parser.add_argument(
        "--wpconfig",
        default="/var/www/html/wp-config.php",
        help="Path to wp-config.php (default: /var/www/html/wp-config.php)",
    )
    parser.add_argument("--host",   default=None)
    parser.add_argument("--user",   default=None)
    parser.add_argument("--passwd", default=None)
    parser.add_argument("--db",     default=None)
    parser.add_argument("--prefix", default=None, help="Table prefix (default: read from wp-config)")
    args = parser.parse_args()

    wpc = Path(args.wpconfig)
    if wpc.exists():
        creds = parse_wpconfig(wpc)
        print(f"  Credentials loaded from {wpc}")
    else:
        creds = {"host": "localhost", "user": None, "passwd": None, "db": None, "prefix": "wp_"}
        if args.wpconfig != "/var/www/html/wp-config.php":
            print(f"  {Y}Warning: wp-config.php not found at {wpc}{X}")

    for k in ("host", "user", "passwd", "db", "prefix"):
        v = getattr(args, k)
        if v is not None:
            creds[k] = v

    if not creds.get("user") or not creds.get("db"):
        sys.exit(
            f"{R}ERROR:{X} DB credentials missing.\n"
            "Pass --wpconfig /path/to/wp-config.php  or  --host/--user/--passwd/--db"
        )

    pfx = creds.get("prefix", "wp_")

    print(f"\n{B}WordPress Post Comparator{X}")
    print(f"  Manual post : #{args.manual_id}")
    print(f"  Agent post  : #{args.agent_id}")
    print(f"  Database    : {creds['user']}@{creds['host']}/{creds['db']}  prefix={pfx}")

    conn = connect(creds)
    cur = conn.cursor()

    try:
        compare_posts(cur, pfx, args.manual_id, args.agent_id)
        compare_postmeta(cur, pfx, args.manual_id, args.agent_id)
        compare_taxonomies(cur, pfx, args.manual_id, args.agent_id)
        analysis()
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
