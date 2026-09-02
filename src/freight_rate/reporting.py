"""Small formatting helpers for the generated reports.

``DataFrame.to_markdown`` needs ``tabulate``, which is one more thing a reviewer
would have to install to reproduce a table. The renderer below covers what this
project needs in a few lines and keeps the dependency list at what the model
actually requires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _format(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        magnitude = abs(float(value))
        if magnitude >= 1000:
            return f"{float(value):,.1f}"
        if magnitude >= 1 or magnitude == 0:
            return f"{float(value):.3f}"
        return f"{float(value):.4f}"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def markdown_table(frame: pd.DataFrame, index_label: str | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table.

    Numeric columns are right-aligned so magnitudes line up when scanning.
    """
    display = frame.reset_index()
    if index_label and display.columns[0] in ("index", frame.index.name or "index"):
        display = display.rename(columns={display.columns[0]: index_label})

    headers = [str(c) for c in display.columns]
    rows = [[_format(v) for v in record] for record in display.itertuples(index=False)]
    numeric = [
        bool(pd.api.types.is_numeric_dtype(display[c]) and not pd.api.types.is_bool_dtype(display[c]))
        for c in display.columns
    ]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        padded = [
            cells[i].rjust(widths[i]) if numeric[i] else cells[i].ljust(widths[i])
            for i in range(len(cells))
        ]
        return "| " + " | ".join(padded) + " |"

    rule = "|" + "|".join(
        ("-" * (widths[i] + 1) + ":") if numeric[i] else (":" + "-" * (widths[i] + 1))
        for i in range(len(headers))
    ) + "|"

    return "\n".join([line(headers), rule, *(line(r) for r in rows)])


def markdown_section(title: str, body: str, level: int = 2) -> str:
    return f"{'#' * level} {title}\n\n{body}\n"
