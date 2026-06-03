import re
from fractions import Fraction

import altair as alt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="절댓값 두 개 부등식 그래프", layout="centered")
st.title("📐 고1 수학: 절댓값 두 개 부등식 그래프")
st.write(" 학습목표: 절댓값을 포함한 일차부등식을 풀 수 있다!")
st.caption("예시: `|x-1| + |2x+3| <= 5` 또는 `|3x+2| - |x-4| > 1`")

INPUT_EXAMPLE = "|x-1| + |2x+3| <= 5"

def parse_number(term: str) -> Fraction:
    term = term.strip().replace("−", "-")
    if term == "" or term == "+":
        return Fraction(0)
    if term == "-":
        raise ValueError("잘못된 숫자 형식입니다.")
    try:
        return Fraction(term)
    except Exception:
        raise ValueError(f"숫자를 해석할 수 없습니다: {term}")


def parse_linear(expr: str) -> tuple[Fraction, Fraction]:
    expr = expr.replace(" ", "").replace("*", "").replace("−", "-")
    if expr == "":
        raise ValueError("식이 비어 있습니다.")
    if expr[0] not in "+-":
        expr = "+" + expr

    match = re.search(r"([+-](?:\d+(?:\.\d*)?|\.\d+)?)(?:x)", expr)
    if match:
        a_str = match.group(1)
        if a_str in "+-":
            a = Fraction(1 if a_str == "+" else -1)
        else:
            a = parse_number(a_str)
        rest = expr[: match.start()] + expr[match.end() :]
        b = parse_number(rest) if rest else Fraction(0)
    else:
        a = Fraction(0)
        b = parse_number(expr)
    return a, b


def sign(value: Fraction) -> int:
    return 1 if value >= 0 else -1


def compare(value: Fraction, op: str) -> bool:
    if op == "<":
        return value < 0
    if op == "<=":
        return value <= 0
    if op == ">":
        return value > 0
    if op == ">=":
        return value >= 0
    raise ValueError("알 수 없는 부등식입니다.")


def choose_midpoint(left: Fraction | None, right: Fraction | None) -> Fraction:
    if left is None and right is None:
        return Fraction(0)
    if left is None:
        return right - 1
    if right is None:
        return left + 1
    return (left + right) / 2


def intersect_intervals(interval: tuple[Fraction | None, Fraction | None, bool, bool],
                        segment: tuple[Fraction | None, Fraction | None, bool, bool]) -> tuple[Fraction | None, Fraction | None, bool, bool] | None:
    left, right, left_closed, right_closed = interval
    sleft, sright, sleft_closed, sright_closed = segment

    lower = sleft if sleft is not None and (left is None or sleft > left or (sleft == left and sleft_closed and left_closed)) else left
    upper = sright if sright is not None and (right is None or sright < right or (sright == right and sright_closed and right_closed)) else right

    left_closed_result = (sleft == lower and sleft_closed) or (left == lower and left_closed)
    right_closed_result = (sright == upper and sright_closed) or (right == upper and right_closed)

    if lower is not None and upper is not None and lower > upper:
        return None
    if lower == upper and not (left_closed_result and right_closed_result):
        return None
    return lower, upper, left_closed_result, right_closed_result


def solve_linear_segment(c: Fraction, d: Fraction, op: str,
                         left: Fraction | None, right: Fraction | None) -> tuple[Fraction | None, Fraction | None, bool, bool] | None:
    if c == 0:
        if compare(d, op):
            left_closed = left is None or compare(d if left is None else d, op)
            right_closed = right is None or compare(d if right is None else d, op)
            return left, right, left_closed, right_closed
        return None

    bound = -d / c
    if c > 0:
        if op == "<":
            candidate = (None, bound, False, False)
        elif op == "<=":
            candidate = (None, bound, False, True)
        elif op == ">":
            candidate = (bound, None, False, False)
        else:
            candidate = (bound, None, True, False)
    else:
        if op == "<":
            candidate = (bound, None, False, False)
        elif op == "<=":
            candidate = (bound, None, True, False)
        elif op == ">":
            candidate = (None, bound, False, False)
        else:
            candidate = (None, bound, False, True)

    return intersect_intervals(candidate, (left, right, True, True))


def solve_two_abs_inequality(a1: Fraction, b1: Fraction, a2: Fraction, b2: Fraction,
                             sign_op: int, op: str, rhs_a: Fraction, rhs_b: Fraction) -> list[tuple[Fraction | None, Fraction | None, bool, bool]]:
    boundaries = []
    if a1 != 0:
        boundaries.append(-b1 / a1)
    if a2 != 0:
        boundaries.append(-b2 / a2)
    boundaries = sorted(set(boundaries))
    points = [None] + boundaries + [None]
    intervals: list[tuple[Fraction | None, Fraction | None, bool, bool]] = []

    for i in range(len(points) - 1):
        left = points[i]
        right = points[i + 1]
        x_test = choose_midpoint(left, right)
        s1 = sign(a1 * x_test + b1)
        s2 = sign(a2 * x_test + b2)
        c = s1 * a1 + sign_op * s2 * a2 - rhs_a
        d = s1 * b1 + sign_op * s2 * b2 - rhs_b
        solved = solve_linear_segment(c, d, op, left, right)
        if solved:
            intervals.append(solved)

    merged: list[tuple[Fraction | None, Fraction | None, bool, bool]] = []
    for interval in intervals:
        if not merged:
            merged.append(interval)
            continue
        prev = merged[-1]
        if prev[1] == interval[0] and (prev[3] or interval[2]):
            merged[-1] = (prev[0], interval[1], prev[2], interval[3] or prev[3])
        elif prev[1] is None or interval[0] is None or (prev[1] >= interval[0] and (prev[3] or interval[2])):
            lower = prev[0]
            lower_closed = prev[2]
            upper = max(prev[1], interval[1]) if prev[1] is not None and interval[1] is not None else None
            upper_closed = prev[3] or interval[3]
            merged[-1] = (lower, upper, lower_closed, upper_closed)
        else:
            merged.append(interval)

    return merged


def format_fraction(value: Fraction | None) -> str:
    if value is None:
        return "∞"
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_solution_inequality(interval: tuple[Fraction | None, Fraction | None, bool, bool]) -> str:
    left, right, left_closed, right_closed = interval
    if left is None and right is None:
        return r"\mathbb{R}"
    if left is None:
        op = r"\le" if right_closed else r"<"
        return rf"x {op} {format_fraction(right)}"
    if right is None:
        op = r"\le" if left_closed else r"<"
        return rf"{format_fraction(left)} {op} x"
    if left == right:
        return rf"x = {format_fraction(left)}"
    left_op = r"\le" if left_closed else r"<"
    right_op = r"\le" if right_closed else r"<"
    return rf"{format_fraction(left)} {left_op} x {right_op} {format_fraction(right)}"


def build_solution_latex(intervals: list[tuple[Fraction | None, Fraction | None, bool, bool]]) -> str:
    if not intervals:
        return r"\text{해가 없습니다.}"
    if len(intervals) == 1 and intervals[0][0] is None and intervals[0][1] is None:
        return r"\mathbb{R}"
    return r" \text{ 또는 } ".join(format_solution_inequality(interval) for interval in intervals)


def build_numberline_chart(intervals: list[tuple[Fraction | None, Fraction | None, bool, bool]]) -> None:
    finite_values = [float(v) for interval in intervals for v in (interval[0], interval[1]) if v is not None]
    if finite_values:
        x_min, x_max = min(finite_values) - 2, max(finite_values) + 2
    else:
        x_min, x_max = -10.0, 10.0

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(-1, 1)

    ax.axhline(y=0, color="black", linewidth=1.5)
    ax.plot([x_min - 0.5, x_max + 0.5], [0, 0], "k-", linewidth=1)

    tick_positions = []
    for interval in intervals:
        if interval[0] is not None:
            tick_positions.append(float(interval[0]))
        if interval[1] is not None:
            tick_positions.append(float(interval[1]))

    for pos in sorted(set(tick_positions)):
        ax.plot([pos, pos], [-0.08, 0.08], "k-", linewidth=1)
        ax.text(pos, -0.3, str(format_fraction(Fraction(pos).limit_denominator())), ha="center", fontsize=10)

    for left, right, left_closed, right_closed in intervals:
        start = x_min - 0.5 if left is None else float(left)
        end = x_max + 0.5 if right is None else float(right)

        ax.barh(0, end - start, left=start, height=0.2, color="#2ca02c", alpha=0.6, edgecolor="none")

        if left is not None:
            marker = "o" if left_closed else "o"
            color = "darkgreen" if left_closed else "white"
            ax.plot(float(left), 0, marker=marker, markersize=12, color=color, markeredgecolor="darkgreen", markeredgewidth=2, zorder=5)

        if right is not None:
            marker = "o" if right_closed else "o"
            color = "darkgreen" if right_closed else "white"
            ax.plot(float(right), 0, marker=marker, markersize=12, color=color, markeredgecolor="darkgreen", markeredgewidth=2, zorder=5)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_aspect("auto")

    st.pyplot(fig, use_container_width=True)


def build_graph_data(a1: Fraction, b1: Fraction, a2: Fraction, b2: Fraction, sign_op: int, rhs_a: Fraction, rhs_b: Fraction):
    roots = []
    if a1 != 0:
        roots.append(float(-b1 / a1))
    if a2 != 0:
        roots.append(float(-b2 / a2))
    if roots:
        x_min, x_max = min(roots) - 5, max(roots) + 5
    else:
        x_min, x_max = -10.0, 10.0
    x = np.linspace(x_min, x_max, 500)
    lhs = np.abs(a1 * x + b1) + sign_op * np.abs(a2 * x + b2)
    rhs_line = rhs_a * x + rhs_b
    return pd.DataFrame({"x": x, "LHS": lhs, "RHS": rhs_line})


def plot_inequality(df: pd.DataFrame, op: str) -> alt.Chart:
    base = alt.Chart(df).encode(x=alt.X("x", title="x"))
    line_lhs = base.mark_line(color="#1f77b4", strokeWidth=3).encode(y=alt.Y("LHS", title="절댓값 식"))
    line_rhs = base.mark_line(color="#d62728", strokeWidth=2, opacity=0.8).encode(y=alt.Y("RHS", title="우변 식"))
    if op in ["<", "<="]:
        area = base.mark_area(opacity=0.25, color="#2ca02c").encode(
            y=alt.Y("LHS", title=""),
            y2="RHS"
        ).transform_filter(alt.datum.LHS <= alt.datum.RHS)
    else:
        area = base.mark_area(opacity=0.25, color="#ff7f0e").encode(
            y=alt.Y("RHS", title=""),
            y2="LHS"
        ).transform_filter(alt.datum.LHS >= alt.datum.RHS)
    return alt.layer(area, line_lhs, line_rhs).properties(height=400, width=720)


st.subheader("절댓값을 포함한 일차부등식을 입력하세요")
col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])
with col1:
    expr1 = st.text_input("첫 번째 절댓값 식", value="x-1")
with col2:
    sign_op_text = st.selectbox("연산", ["+", "-"], index=0)
with col3:
    expr2 = st.text_input("두 번째 절댓값 식", value="2x+3")
with col4:
    op = st.selectbox("부등호", ["<", "<=", ">", ">="], index=1)
with col5:
    rhs = st.text_input("우변 식", value="5")

if st.button("그래프 그리기"):
    try:
        a1, b1 = parse_linear(expr1)
        a2, b2 = parse_linear(expr2)
        rhs_a, rhs_b = parse_linear(rhs)
        sign_op = 1 if sign_op_text == "+" else -1

        intervals = solve_two_abs_inequality(a1, b1, a2, b2, sign_op, op, rhs_a, rhs_b)
        solution_latex = build_solution_latex(intervals)

        st.subheader("입력한 부등식")
        latex_expr = rf"\left|{expr1}\right| {sign_op_text} \left|{expr2}\right| {op} {rhs}"
        st.latex(latex_expr)

        st.subheader("그래프")
        df = build_graph_data(a1, b1, a2, b2, sign_op, rhs_a, rhs_b)
        st.altair_chart(plot_inequality(df, op), use_container_width=True)

        st.subheader("수직선으로 표현한 해")
        build_numberline_chart(intervals)

        st.subheader("부등식의 해")
        st.latex(solution_latex)
    except ValueError as error:
        st.error(f"입력 형식 오류: {error}")
        st.info("예: x-1, 2x+3, 5, 또는 2x-4")
