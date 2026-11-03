"""
_miniyaml.py - leitor/escritor YAML minimo (sem dependencia externa).

Cobre o subconjunto usado pelos configs do projeto:
  - comentarios (#) e linhas vazias
  - mapeamentos em bloco  `chave: valor`
  - blocos aninhados por indentacao (dict ou lista)
  - listas em bloco       `- item`
  - listas/dicts inline    `[a, b]`  e  `{a: 1, b: 2}`
  - escalares tipados: int, float, bool, null, string (com ou sem aspas)

NAO cobre YAML avancado (ancoras, multi-doc, blocos literais |/>). Suficiente
para experiment.yaml / run.yaml / aoi.yaml deste repo. No container ha PyYAML;
aqui no host usamos este fallback para o runner nao depender de pip install.
"""
from __future__ import annotations


# --------------------------------------------------------------------------- load
def _scalar(tok: str):
    t = tok.strip()
    if t == "" or t in ("~", "null", "Null", "NULL"):
        return None
    if (t[0] == t[-1]) and t[0] in ("'", '"') and len(t) >= 2:
        return t[1:-1]
    low = t.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t


def _flow(tok: str):
    """Parse inline [..] ou {..}."""
    t = tok.strip()
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        if not inner:
            return []
        return [_flow(p) for p in _split_top(inner)]
    if t.startswith("{") and t.endswith("}"):
        inner = t[1:-1].strip()
        d = {}
        if inner:
            for p in _split_top(inner):
                k, _, v = p.partition(":")
                d[k.strip()] = _flow(v)
        return d
    return _scalar(t)


def _split_top(s: str):
    """Split por virgula no nivel 0 (respeita [] e {})."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def _strip_comment(line: str) -> str:
    out, q = "", None
    for ch in line:
        if q:
            out += ch
            if ch == q:
                q = None
        elif ch in ("'", '"'):
            q = ch
            out += ch
        elif ch == "#":
            break
        else:
            out += ch
    return out.rstrip()


def loads(text: str):
    lines = []
    for raw in text.splitlines():
        s = _strip_comment(raw)
        if s.strip() == "":
            continue
        indent = len(s) - len(s.lstrip(" "))
        lines.append((indent, s.strip()))
    val, _ = _parse_block(lines, 0, 0)
    return val if val is not None else {}


def _parse_block(lines, i, indent):
    # lista?
    if i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
        items = []
        while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
            rest = lines[i][1][2:].strip()
            if rest == "":
                child, i = _parse_block(lines, i + 1, _next_indent(lines, i + 1, indent))
                items.append(child)
            elif ":" in rest and not rest.startswith("["):
                # item de lista que e um mapa inline na mesma linha
                pseudo = [(indent + 2, rest)]
                j = i + 1
                while j < len(lines) and lines[j][0] > indent:
                    pseudo.append((lines[j][0], lines[j][1]))
                    j += 1
                child, _ = _parse_block(pseudo, 0, indent + 2)
                items.append(child)
                i = j
            else:
                items.append(_flow(rest))
                i += 1
        return items, i
    # mapa
    d = {}
    while i < len(lines) and lines[i][0] == indent:
        key, _, rest = lines[i][1].partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest == "":
            ni = _next_indent(lines, i + 1, indent)
            if ni > indent:
                child, i = _parse_block(lines, i + 1, ni)
                d[key] = child
            else:
                d[key] = None
                i += 1
        else:
            d[key] = _flow(rest)
            i += 1
    return d, i


def _next_indent(lines, i, cur):
    if i < len(lines):
        return lines[i][0]
    return cur


def load(path):
    from pathlib import Path
    return loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- dump
def _dump_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    if s == "" or any(c in s for c in ":#{}[],&*!|>'\"%@`") or s != s.strip():
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _dump(obj, indent=0):
    pad = "  " * indent
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict) and v:
                out.append(f"{pad}{k}:")
                out.append(_dump(v, indent + 1))
            elif isinstance(v, list) and v:
                out.append(f"{pad}{k}:")
                out.append(_dump(v, indent + 1))
            else:
                if isinstance(v, list):
                    out.append(f"{pad}{k}: []")
                elif isinstance(v, dict):
                    out.append(f"{pad}{k}: {{}}")
                else:
                    out.append(f"{pad}{k}: {_dump_scalar(v)}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and item:
                body = _dump(item, indent + 1)
                first, *rest = body.split("\n")
                out.append(f"{pad}- {first.strip()}")
                out.extend(rest)
            else:
                out.append(f"{pad}- {_dump_scalar(item)}")
    else:
        out.append(f"{pad}{_dump_scalar(obj)}")
    return "\n".join(out)


def dumps(obj) -> str:
    return _dump(obj, 0) + "\n"


def dump(obj, path):
    from pathlib import Path
    Path(path).write_text(dumps(obj), encoding="utf-8")


if __name__ == "__main__":
    sample = """
network: manet
nums: [1, 2, 3]
profiles:
  home: {batch: 2, workers: 2}
  lab:  {batch: 4, workers: 8}
predictions:
  - aoi: a
    f1: 0.8
  - aoi: b
    f1: 0.55
"""
    d = loads(sample)
    print(d)
    print("----")
    print(dumps(d))
