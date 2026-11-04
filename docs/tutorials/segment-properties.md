# Names and tags for a segmentation: writing a rules module

From the per-body records in DVID to a segmentation layer whose Segments tab lists each
body's name and a set of tags you can click through: family, class, column, side, and flags
such as `fragment`. The tags come from a **rules module**, a short Python file you write for
your dataset. neu-mark has no built-in rules, because what the parts of a name mean (which
token is a column, which a cell type, which marks a fragment) is a fact about the dataset.

The example is cut down from the rules used for a Megaphragma segmentation, whose bodies are
named like `Tm2_A2(L)`, `PEN_E2_truncated`, `SNP(L)_fragment` and `LN_C5(L)_NCL`. The paths
are placeholders.

## 1. Look at the names

`neu_mark.explore` takes a DataFrame of records and returns DataFrames, for a notebook:

```python
from neu_mark import explore as ex, notebook as nb

bodies = nb.body_annotations("dvid://dvid.example.org/abc123/labels_annotations",
                             "bodies.csv")         # from `neu-mark select-bodies`

ex.instances(bodies, "instance")                       # distinct names -> body counts
ex.tokens(bodies, "instance", drop=r"\((L|R)\)")       # the suffix vocabulary
ex.near(ex.tokens(bodies, "instance"), "fragment")     # misspellings of a flag
```

## 2. Write the module

```python
# wasp_rules.py
import re
from neu_mark import rule
from neu_mark.explore import split_tokens
from neu_mark.rules import RuleSet

LABEL = "instance"          # the record field that names each body

# family and class by the name's start; first match wins
CLASSES = [("OL", "Tm", r"Tm\d*"), ("OL", "Mi", r"Mi\d*"), ("OL", "LC", r"LC\d*"),
           ("CX", "PEN", r"PEN[ab]?"), ("CX", "EP", r"EP")]
CLASSES = [(f, c, re.compile(f"(?:{p})(?=$|[_(])")) for f, c, p in CLASSES]


def _class(name):
    return next(((f, c) for f, c, p in CLASSES if p.match(name or "")), (None, None))


def _tokens(r):
    return split_tokens(r["instance"])


@rule(drop=True)
def noise(r):
    """not real data"""
    return next((t for t in ("irrelevant", "block") if t in _tokens(r)), None)


@rule
def family(r):
    """the brain region the cell type belongs to"""
    return _class(r["instance"])[0]


@rule(name="class")
def cell_class(r):
    """the cell's class, carrying its family so the list sorts by family"""
    f, c = _class(r["instance"])
    return f"{f}/{c}" if c else None


@rule(multi=True)
def col(r):
    """optic lobe column, read only for optic lobe cells"""
    if _class(r["instance"])[0] != "OL":
        return []
    return [t for t in _tokens(r)[1:] if re.fullmatch(r"[A-E]\d", t)]


@rule
def side(r):
    """hemisphere"""
    m = re.search(r"\((L|R)\)", r["instance"] or "")
    return m.group(1) if m else None


@rule(prefix="")
def fragment(r):
    """a fragment, not a completely traced cell"""
    return "fragment" if "fragment" in _tokens(r) else None


@rule(tag=False)
def parsed_type(r):
    """the cell type in the name, for the table only"""
    return re.sub(r"\(.*", "", r["instance"] or "").split("_")[0] or None


# Declaration order is the order the viewer lists the facets in; flags come last.
RULES = RuleSet([noise, family, cell_class, col, side, fragment, parsed_type], label=LABEL)
```

What each piece does:

- **`LABEL`** names the field that labels the segments. A rule that declares no `needs`
  reads it, and `test`, `explain` and `remainder` fill it.
- **A rule** returns a value, a list (with `multi=True`) or `None` for "did not fire". Its
  name is the tag prefix (`family:OL`); `prefix=""` makes a bare flag; the docstring is the
  description the viewer shows for each tag.
- **`drop=True`** leaves the body out of the document; the reason is counted in the report.
- **`tag=False`** keeps a value out of the tags but writes it to the `--table` output. A
  cell type usually has hundreds of values, which is a long list to click through; the label
  already starts with it, and the viewer's search box matches label prefixes as you type.
- **Reading a column only within its family** (`col` above) is what keeps `E2` an
  ellipsoid-body column on `PEN_E2` from also being an optic lobe column.
- **Values carrying their family** (`class:OL/Tm`) sort by family in the viewer's list.

## 3. Try it in a notebook

```python
from neu_mark import rules

RS = rules.from_module("wasp_rules.py")
RS.explain("PEN_E2_truncated")      # {'noise': None, 'family': 'CX', 'class': 'CX/PEN', ...}
RS["col"].test("Tm2_A2(L)")         # ['A2']
RS.coverage(bodies)                 # how many bodies each rule fired on
RS.unparsed(bodies, "class")        # the names it missed, by bodies
```

## 4. Build, check, publish

```bash
neu-mark segment-properties --src dvid://dvid.example.org/abc123/labels_annotations \
    --dst s3://my-bucket/seg --meshed-bodies --rules wasp_rules.py \
    --counts dvid://dvid.example.org/abc123/synapses_labelsz \
    --dry-run --table work/segprops.parquet
```

The dry run writes nothing but the table and prints each rule's coverage. A facet that fires
on 2% of bodies looks the same in the viewer as one the data does not have, so read the
numbers before publishing. Rerun without `--dry-run` to write the document into the volume
and link it.

`--drop RULE` excludes the bodies a rule fires on instead of tagging them (`--drop glia`,
with your module's own `glia` rule), and `--label FIELD` labels by another field.

## In the viewer

In the segmentation layer's **Segments** tab, the tags are listed facet by facet in the
module's order, flags last. The search box takes:

- `Tm2` — names starting with `Tm2`, as you type;
- `/^T4[a-d]` — a regular expression over the names (start it with `^`);
- `#class:OL/Tm` and `-#fragment` — keep or drop a tag, which must be typed in full;
- several at once: `/^T4 -#fragment -#truncated`.
