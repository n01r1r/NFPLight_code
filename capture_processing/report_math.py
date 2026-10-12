"""Offline MathML for the report's fixed mathematical notation (display only)."""

import html
import re


def _tokens(expression: str) -> str:
    parts = []
    for token in re.findall(r"[A-Za-zΑ-ω]+(?:_[A-Za-z0-9θ]+)?|\d+(?:\.\d+)?|[^\s]", expression):
        if "_" in token:
            base, subscript = token.split("_", 1)
            parts.append(f"<msub><mi>{html.escape(base)}</mi><mi>{html.escape(subscript)}</mi></msub>")
        elif token.replace(".", "", 1).isdigit():
            parts.append(f"<mn>{token}</mn>")
        else:
            tag = "mi" if token.isalpha() else "mo"
            parts.append(f"<{tag}>{html.escape(token)}</{tag}>")
    return "<mrow>" + "".join(parts) + "</mrow>"


def _frac(numerator: str, denominator: str) -> str:
    return f"<mfrac>{numerator}{denominator}</mfrac>"


def _power(base: str, exponent: str) -> str:
    return f"<msup>{_tokens(base)}{_tokens(exponent)}</msup>"


def _math(body: str, source: str = "") -> str:
    return ('<math xmlns="http://www.w3.org/1998/Math/MathML" '
            f'aria-label="{html.escape(source, quote=True)}">{body}</math>')


_EPS = _power("10", "−5")
_NORMALIZE = _tokens("Normalize(z)=") + _frac(
    _tokens("z−min(z)"), _tokens("max(max(z)−min(z),") + _EPS + _tokens(")"))
_LOG = _tokens("L(x)=") + _frac(_tokens("ln(x+0.01)−ln(0.01)"),
                                _tokens("ln(1.01)−ln(0.01)"))
_RHO = _tokens("ρ=") + _frac(
    "<msqrt>" + _power("(x−127.5)", "2") + _tokens("+") + _power("(y−127.5)", "2") + "</msqrt>",
    _tokens("128"))
_C_RAW = _tokens("c_raw=") + _frac(_tokens("cos θ_F−cos θ_N"), _tokens("cos θ_N"))
_C_NORM = _tokens("c=") + _frac(_tokens("c_raw"), _tokens("max(c_raw)"))
_DEN = _tokens("max(c,") + _EPS + _tokens(")")
def _patch_mean(side: str) -> str:
    return ('<mrow><msub><mo>mean</mo><mrow><mi>Ω</mi><mo>,</mo><mtext>RGB</mtext></mrow></msub>'
            + _tokens(f"(I_{side})") + '</mrow>')


_GAIN = _tokens("K_L=") + _frac(_patch_mean("N"),
    _tokens("max(") + _patch_mean("F") + _tokens(",") + _EPS + _tokens(")"))
# These are the active author formulas, written explicitly to preserve their
# scientific meaning; this is deliberately not a general TeX parser.
_AUTHOR = {
    "linear input": [_tokens("X=[I_N,I_F]"), _tokens("I_N,I_F ∈ [0,1]")],
    "far input": [_tokens("I_F"), _tokens("I_F ∈ [0,1]")],
    "center-patch gain": [_GAIN],
    "angle ratio": [_tokens("K_θ=") + _frac(_tokens("cos θ_N"), _tokens("cos θ_F")),
                    _tokens("θ_N=atan(") + _frac(_tokens("ρ"), _tokens("4")) + _tokens("); θ_F=atan(") + _frac(_tokens("ρ"), _tokens("12")) + _tokens(")"), _RHO],
    "gained far": [_tokens("K_L I_F"), _tokens("clip(K_L I_F,0,1)")],
    "time-scaled far": [_tokens("K=K_L K_θ; clip(K I_F,0,1)")],
    "difference": [_tokens("Δ=I_N−clip(K I_F,0,1)")],
    "absolute difference": [_tokens("|I_N−clip(K I_F,0,1)|")],
    "mean difference": [_tokens("mean_RGB(|I_N−clip(K I_F,0,1)|)")],
    "relation denominator": [_tokens("C_M=cos θ_N |cos θ_N−cos θ_F|"),
                             _C_RAW, _C_NORM, _tokens("c_den=") + _DEN],
    "raw relation": [_tokens("q=") + _frac(_tokens("mean_RGB(|Δ|)"), _DEN)],
    "raw log relation": [_tokens("ln(max(q,") + _EPS + _tokens("))")],
    "normalized relation": [_tokens("R_M=1−Normalize(q)"), _NORMALIZE],
    "normalized log relation": [_tokens("R_log=1−Normalize(ln(max(q,") + _EPS + _tokens(")))")],
    "saturation mask": [_tokens("M_R=1[max_RGB(I_N,I_F,clip(K_L I_F,0,1))≤0.95]")],
    "log operator · near RGB": [_tokens("L(I_N)"), _LOG],
    "log operator · far RGB": [_tokens("L(I_F)")],
    "33-channel input": [_tokens("V=[I_N,I_F,copy(I_N),copy(I_F),clip(K_L I_F,0,1)]"),
                         _tokens("X_net=2[V,L(V),R_M,R_log,M_R]−1"),
                         _tokens("15+15+3=33; X_net ∈ [−1,1]")],
}
_PROSE_EQUATIONS = {
    "K = K_L K_θ": _tokens("K=K_L K_θ"),
    "C_M = cos θ_N |cos θ_N − cos θ_F|": _tokens("C_M=cos θ_N |cos θ_N−cos θ_F|"),
    "R_M = |I_N − K I_F| / C_M": _tokens("R_M=") + _frac(_tokens("|I_N−K I_F|"), _tokens("C_M")),
    "K_θ = cos θ_N / cos θ_F": _AUTHOR["angle ratio"][0],
    "c_raw = (cos θ_F − cos θ_N) / cos θ_N": _C_RAW,
    "c = c_raw / max(c_raw)": _C_NORM,
    "max(c, 10⁻⁵)": _DEN,
    "ρ = √((x−127.5)²+(y−127.5)²)/128": _RHO,
    "I_N - clip(K I_F,0,1)": _tokens("I_N−clip(K I_F,0,1)"),
    "(raw+1)/2": _frac(_tokens("raw+1"), _tokens("2")),
    "(n+1)/2": _frac(_tokens("n+1"), _tokens("2")),
    "x^(1/2.2)": '<msup>' + _tokens("x") + _frac(_tokens("1"), _tokens("2.2")) + '</msup>',
}


def formula_html(title: str, lhs: str, rhs: str, *, author: bool) -> str:
    if author and title in _AUTHOR:
        equations = _AUTHOR[title]
        # Keep the distinction between the paper denominator and code stages.
        prefix = ["논문", "코드", "코드", "코드"] if title == "relation denominator" else [""] * len(equations)
        rendered = "".join(f'<div class="equation">{label + ": " if label else ""}{_math(body, lhs)}</div>'
                           for label, body in zip(prefix, equations))
        if title == "center-patch gain":
            rendered += '<p>Ω: 중앙 20×20 영역 [118:138, 118:138], RGB 전체 평균.</p>'
        if title in {"normalized relation", "normalized log relation"}:
            rendered += '<p>R_M과 R_log는 각각 정규화·반전한 모델 입력이며, R_log는 정규화된 R_M의 로그가 아닙니다.</p>'
        # Non-equation annotations remain prose, without duplicating the formula.
        if "=" not in rhs and title not in {"linear input", "far input", "33-channel input"}:
            rendered += f'<p>{math_prose(rhs)}</p>'
        return rendered
    return f'<div class="equation">{_math(_tokens(lhs), lhs)}</div><p>{math_prose(rhs)}</p>'


def math_prose(text: str) -> str:
    """Replace fixed report symbols in prose, leaving ordinary words as text."""
    pattern = r"mean_RGB\(\|I_N − clip\(K I_F,0,1\)\|\)|\|I_N − clip\(K I_F,0,1\)\||I_N − clip\(K I_F,0,1\)|log\(R_M\)|log\(I_[NF]\)|K_L I_F|K I_F|[IKRCM]_[A-Za-zθ]+|θ_[NFnf]|c_(?:raw|norm|den)"
    pattern = "|".join(re.escape(source) for source in _PROSE_EQUATIONS) + "|" + pattern
    result, offset = [], 0
    for match in re.finditer(pattern, text):
        body = _PROSE_EQUATIONS.get(match.group(), _tokens(match.group()))
        result.extend((html.escape(text[offset:match.start()]), _math(body, match.group())))
        offset = match.end()
    result.append(html.escape(text[offset:]))
    return "".join(result)


def matrix_html(values) -> str:
    if values is None:
        return "—"
    rows = values if values and isinstance(values[0], (list, tuple)) else [values]
    body = '<mo>[</mo><mtable>' + "".join(
        '<mtr>' + "".join(f'<mtd><mn>{float(value):.7g}</mn></mtd>' for value in row) + '</mtr>'
        for row in rows) + '</mtable><mo>]</mo>'
    return _math(body, "numeric matrix")
