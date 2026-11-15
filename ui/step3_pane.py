"""
Left Pane content (Phase 4A): Step 3 (Gr and Pr(p) - victim station) and Step 4 (Lb(p)).

Public entry points
-------------------
``step3_form(state)``
    Refreshable form for Step 3. Three ways to determine the victim station's Gr and Pr(p):
      1. Auto-lookup from the ITU tables (RR Appendix 7 Tables 7/8/9, ``core.tables_data``).
      2. Calculate from station specs: the Gr calculator (Eq. 84-97, shared with Step 1's Gt
         calculator) and the Pr(p) calculator (Rec. ITU-R S.1430 Eq. 3,
         ``core.Pr_calculations``).
      3. Fully automatic from the receive frequency of an imported ITU .mdb station (Table 8).

``step4_form(state, on_cleared=None)``
    Lb(p) = Pt + Gt + Gr - Pr(p), plus the "Clear all" button.

State used
----------
    * committed results  -> ``state.wizard_results`` (gr, pr_p, final_te, final_b, gr_pr_source,
                            prd_result, lb_p)
    * band selection     -> ``state.step3`` (selected_band, selected_freq, band_freq_map)
    * lookup scratch     -> ``state.victim_lookup`` (rows, freq, gr_calculated, pr_calculated)
    * ITU import         -> ``state.itu_import`` (freqs, victim_freq_ghz)
    * remembered fields  -> ``state.get_draft / put_draft`` ("s3_*" and "prd_*" keys)

Nothing in ``core/`` is modified; it is only imported.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from nicegui import ui

from core import Pr_calculations, tables_data
from ui.left_pane import (
    _draft,
    _drafted_number,
    _f,
    _fmt,
    _readonly_field,
    _section_title,
    _status,
    open_gain_calculator,
)
from ui.state import AppState

# ===========================================================================
# Constants
# ===========================================================================
METHODS = {
    "lookup": "Auto-lookup from ITU tables (Tables 7/8/9)",
    "calc": "Calculate from station specs",
    "itu": "Use frequencies from imported ITU MDB file",
}
TABLE_CHOICES = {
    "7": "Table 7 (transmitting ES)",
    "8": "Table 8 (receiving ES)",
    "9": "Table 9 (bidirectional ES)",
}
ANY = "Any"


# ===========================================================================
# Table helpers (pure functions - no UI)
# ===========================================================================
def _num(value: Any) -> Optional[float]:
    """Parse a workbook cell ("4x10^3", "49 (Note 2)", "-", NaN, ...) into a float or None."""
    return tables_data.parse_engineering_number(value)


def _row_values(table: str, row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Gr, Pr(p), Te and B of one table row (works on raw rows and on parsed rows).

    Tables 7/8 carry the earth-station gain in "Gx (dBi)". Table 9 has both "Gm" (on-axis) and
    "Gr" (horizon): the horizon gain is the one that belongs in Lb(p), so Gr is preferred there.
    The on-axis Gm is never used as a fallback.
    """
    gx, gr_h = _num(row.get("Gx (dBi)")), _num(row.get("Gr (dBi)"))
    if table == "9":
        gr = gr_h if gr_h is not None else gx
    else:
        gr = gx if gx is not None else gr_h
    return {
        "gr": gr,
        "pr": _num(row.get("Pr(p) (dBW)")),
        "te": _num(row.get("Te (K)")),
        "b": _num(row.get("B (Hz)")),
    }


def _band_center_ghz(band: Optional[str]) -> Optional[float]:
    """Centre (GHz) of the first range in a "Frequency Band" string, or None if unparseable."""
    ranges = tables_data.parse_frequency_ranges_ghz(band) if band else []
    return (ranges[0][0] + ranges[0][1]) / 2.0 if ranges else None


def _distinct(rows: List[Dict[str, Any]], key: str) -> List[str]:
    seen: List[str] = []
    for r in rows:
        v = r.get(key)
        if v is not None and str(v).strip() and str(v) not in seen:
            seen.append(str(v))
    return seen


def _filter_rows(rows: List[Dict[str, Any]], orbit: str, modulation: str) -> List[Dict[str, Any]]:
    out = rows
    if orbit != ANY:
        out = [r for r in out if str(r.get("Orbit")) == orbit]
    if modulation != ANY:
        out = [r for r in out if str(r.get("Modulation")) == modulation]
    return out


def _cell(value: Optional[float], fmt: str = "{:.2f}") -> str:
    return "—" if value is None else fmt.format(value)


def _remember_band(state: AppState, band: Optional[str], freq: Optional[float]) -> None:
    """Save the accepted band in ``state.step3`` (feeds frequency / p auto-fill in Step 5)."""
    s3 = state.step3
    s3.selected_band = band
    if freq is not None:
        s3.selected_freq = freq
        if band:
            s3.band_freq_map[band] = freq
    state.victim_lookup.freq = freq


def _save_victim(state: AppState, gr: float, pr: float, te: Optional[float],
                 b: Optional[float], source: str) -> None:
    """Commit Gr / Pr(p) (+ Te, B for the I/N steps) to ``state.wizard_results``."""
    wr = state.wizard_results
    wr.gr, wr.pr_p = gr, pr
    wr.final_te, wr.final_b = te, b
    wr.gr_pr_source = source
    wr.lb_p = None  # derived: Step 4 recomputes it


# ===========================================================================
# Method 3 - fully automatic from the imported ITU .mdb station
# ===========================================================================
def _itu_receive_freqs(state: AppState) -> List[Dict[str, Any]]:
    freqs = state.itu_import.freqs or {}
    return list(freqs.get("receive") or [])


def _itu_sig(state: AppState) -> Optional[str]:
    rx = _itu_receive_freqs(state)
    if not rx:
        return None
    return f"{(state.itu_import.freqs or {}).get('label')}|{rx[0].get('freq_center_ghz')}"


# def _auto_itu(state: AppState) -> bool:
#     """Pick the first receive (R) frequency of the imported station, query Table 8 for it and
#     save the best row. Notifies the outcome. Returns True if Gr / Pr(p) were saved."""
#     rx = _itu_receive_freqs(state)
#     if not rx:
#         ui.notify("No receive (emi_rcp = R) frequency in the imported ITU file — use one of the "
#                   "other methods.", type="warning", multi_line=True)
#         return False

#     state.put_draft("s3_itu_sig", _itu_sig(state))  # set first: a failure must not retry on every render
#     freq = float(rx[0]["freq_center_ghz"])
#     try:
#         rows = tables_data.get_victim_parameters("8", freq)
#     except Exception as ex:  # noqa: BLE001 - e.g. workbook missing
#         ui.notify(f"Could not query Table 8: {ex}", type="negative", multi_line=True)
#         return False

#     state.victim_lookup.rows = rows
#     usable = [r for r in rows
#               if _row_values("8", r)["gr"] is not None and _row_values("8", r)["pr"] is not None]
#     if not usable:
#         ui.notify(f"No Table 8 row at {freq:.4f} GHz has both Gr and Pr(p). Use the Auto-lookup "
#                   "method to pick a band manually, or calculate them.",
#                   type="warning", multi_line=True)
#         return False

#     # This tool only supports GSO, so prefer a GSO row; otherwise the first usable one.
#     best = min(usable, key=lambda r: 0 if str(r.get("Orbit")).strip().lower() == "gso" else 1)
#     v = _row_values("8", best)
#     band = best.get("Frequency Band")
#     _save_victim(state, v["gr"], v["pr"], v["te"], v["b"],
#                  f"(ITU Table {best.get('Table') or '8'}, {band}, receive frequency "
#                  f"{freq:.4f} GHz from the imported ITU file)")
#     _remember_band(state, band, freq)
#     state.itu_import.victim_freq_ghz = freq

#     ui.notify(f"Victim parameters set from Table {best.get('Table') or '8'} at {freq:.4f} GHz: "
#               f"Gr = {v['gr']:.4f} dBi, Pr(p) = {v['pr']:.4f} dBW.", type="positive", multi_line=True)
#     if len(usable) > 1:
#         ui.notify(f"{len(usable)} Table 8 rows match {freq:.4f} GHz; the best one (GSO first) was "
#                   "used. Use “Auto-lookup” to choose a different row.", type="info",
#                   multi_line=True, timeout=8000)
#     return True



def _auto_itu(state: AppState) -> bool:
    """Pick the first receive (R) frequency of the imported station, query ITU tables for it and
    save the best row. Tries Table 8 first, then falls back to Table 7 and 9.
    Returns True if Gr / Pr(p) were saved."""
    rx = _itu_receive_freqs(state)
    if not rx:
        ui.notify("No receive (emi_rcp = R) frequency in the imported ITU file — use one of the "
                  "other methods.", type="warning", multi_line=True)
        return False
    
    state.put_draft("s3_itu_sig", _itu_sig(state))
    freq = float(rx[0]["freq_center_ghz"])
    
    # Fallback search: Try Table 8 (Receiving ES), then Table 7 (Transmitting ES), then Table 9.
    # 14.78 GHz is an uplink frequency, so it won't be in Table 8 (downlink), but will be in Table 7.
    tables_to_try = [
        ("8", "Receiving Earth Station (Table 8)"), 
        ("7", "Transmitting Earth Station (Table 7)"),
        ("9", "Bidirectional ES (Table 9)")
    ]
    
    rows = []
    used_table_alias = None
    used_table_label = None
    
    for alias, label in tables_to_try:
        try:
            rows = tables_data.get_victim_parameters(alias, freq)
            if rows:
                used_table_alias = alias
                used_table_label = label
                break
        except Exception:
            continue
            
    if not rows:
        ui.notify(f"No ITU table row found at {freq:.4f} GHz in Tables 7, 8, or 9. Use the "
                  "Auto-lookup method to pick a band manually, or calculate them.",
                  type="warning", multi_line=True)
        return False
        
    usable = [r for r in rows
              if _row_values(used_table_alias, r)["gr"] is not None and _row_values(used_table_alias, r)["pr"] is not None]
              
    if not usable:
        ui.notify(f"Found rows for {freq:.4f} GHz in {used_table_label}, but none have both Gr and Pr(p). "
                  "Use the Auto-lookup method to pick a band manually, or calculate them.",
                  type="warning", multi_line=True)
        return False
        
    # Prefer GSO rows
    best = min(usable, key=lambda r: 0 if str(r.get("Orbit")).strip().lower() == "gso" else 1)
    v = _row_values(used_table_alias, best)
    band = best.get("Frequency Band")
    
    _save_victim(state, v["gr"], v["pr"], v["te"], v["b"],
                 f"(ITU {used_table_label}, {band}, frequency {freq:.4f} GHz from the imported ITU file)")
    _remember_band(state, band, freq)
    state.itu_import.victim_freq_ghz = freq
    
    ui.notify(f"✅ Victim parameters set from {used_table_label} at {freq:.4f} GHz: "
              f"Gr = {v['gr']:.4f} dBi, Pr(p) = {v['pr']:.4f} dBW.", type="positive", multi_line=True)
    return True


def _itu_panel(state: AppState) -> None:
    rx = _itu_receive_freqs(state)
    if not rx:
        ui.label("No ITU .mdb station with a receive frequency has been imported. Import one in the "
                 "section above, or use another method.").classes("text-warning")
        return

    # Automatic: run once per imported station / receive frequency (not on every re-render).
    sig = _itu_sig(state)
    if sig and state.get_draft("s3_itu_sig") != sig:
        _auto_itu(state)

    label = (state.itu_import.freqs or {}).get("label", "imported station")
    ui.label(f"Station: {label}").classes("text-caption text-grey-7")
    ui.label(f"Receive frequency used: {float(rx[0]['freq_center_ghz']):.4f} GHz"
             + (f" (first of {len(rx)})" if len(rx) > 1 else "")
             + " → Table 8 lookup.").classes("text-caption text-grey-7")

    def reapply() -> None:
        _auto_itu(state)
        step3_form.refresh()

    ui.button("Re-apply from ITU file", icon="refresh", on_click=reapply).props("outline no-caps")


# ===========================================================================
# Method 1 - auto-lookup from the ITU tables
# ===========================================================================
@ui.refreshable
def _lookup_results(state: AppState) -> None:
    """Orbit / modulation filters + the matching-rows table for the selected table and band."""
    table = str(_draft(state, "s3_table", "8"))
    band = state.get_draft("s3_band")
    if not band:
        ui.label("Select a frequency band to see the matching rows.").classes("text-caption text-grey-7")
        return
    try:
        band_rows = tables_data.get_rows_by_band(table, band)
    except Exception as ex:  # noqa: BLE001
        ui.label(f"Could not read Table {table}: {ex}").classes("text-negative")
        return

    orbits, mods = _distinct(band_rows, "Orbit"), _distinct(band_rows, "Modulation")
    orbit = _draft(state, "s3_orbit", ANY)
    mod = _draft(state, "s3_mod", ANY)
    orbit = orbit if orbit in orbits else ANY
    mod = mod if mod in mods else ANY

    def _set_orbit(e) -> None:
        state.put_draft("s3_orbit", e.value)
        _lookup_results.refresh()

    def _set_mod(e) -> None:
        state.put_draft("s3_mod", e.value)
        _lookup_results.refresh()

    with ui.row().classes("w-full no-wrap q-gutter-sm"):
        ui.select([ANY] + orbits, value=orbit, label="Orbit", on_change=_set_orbit) \
            .props("dense outlined").classes("col")
        ui.select([ANY] + mods, value=mod, label="Modulation", on_change=_set_mod) \
            .props("dense outlined").classes("col")

    rows = _filter_rows(band_rows, orbit, mod)
    state.victim_lookup.rows = rows
    state.victim_lookup.freq = _band_center_ghz(band)
    if not rows:
        ui.label("No rows match these filters.").classes("text-warning")
        return

    ui_rows = []
    for i, r in enumerate(rows):
        v = _row_values(table, r)
        ui_rows.append({
            "id": i,
            "table": str(r.get("Table") or table),
            "service": str(r.get("Space Service Designation") or "—"),
            "mod": str(r.get("Modulation") or "—"),
            "orbit": str(r.get("Orbit") or "—"),
            "gr": _cell(v["gr"]),
            "pr": _cell(v["pr"]),
            "te": _cell(v["te"], "{:g}"),
            "b": _cell(v["b"], "{:g}"),
        })
    columns = [
        {"name": "table", "label": "Table", "field": "table", "align": "left"},
        {"name": "service", "label": "Service", "field": "service", "align": "left"},
        {"name": "mod", "label": "Mod.", "field": "mod", "align": "left"},
        {"name": "orbit", "label": "Orbit", "field": "orbit", "align": "left"},
        {"name": "gr", "label": "Gr (dBi)", "field": "gr", "align": "right"},
        {"name": "pr", "label": "Pr(p) (dBW)", "field": "pr", "align": "right"},
        {"name": "te", "label": "Te (K)", "field": "te", "align": "right"},
        {"name": "b", "label": "B (Hz)", "field": "b", "align": "right"},
    ]

    chosen: Dict[str, Optional[int]] = {"id": None}

    def _on_select(e) -> None:
        sel = e.selection
        chosen["id"] = sel[0]["id"] if sel else None

    ui.label(f"{len(rows)} matching row(s). Tick one, then click “Use this row”.") \
        .classes("text-caption text-grey-7")
    ui.table(columns=columns, rows=ui_rows, row_key="id", selection="single",
             on_select=_on_select).props("dense flat bordered").classes("w-full")

    def use_row() -> None:
        idx = chosen["id"]
        if idx is None and len(rows) == 1:
            idx = 0  # only one candidate: no need to tick it
        if idx is None:
            ui.notify("Tick a row in the table first.", type="negative")
            return
        row = rows[idx]
        v = _row_values(table, row)
        if v["gr"] is None or v["pr"] is None:
            missing = " and ".join(n for n, k in (("Gr", "gr"), ("Pr(p)", "pr")) if v[k] is None)
            ui.notify(f"This row has no {missing} in the source table — choose another row, or use "
                      "“Calculate from station specs”.", type="negative", multi_line=True)
            return
        freq = _band_center_ghz(band)
        _save_victim(state, v["gr"], v["pr"], v["te"], v["b"],
                     f"(ITU Table {row.get('Table') or table}, {band})")
        _remember_band(state, band, freq)
        ui.notify(f"Gr = {v['gr']:.4f} dBi, Pr(p) = {v['pr']:.4f} dBW saved.", type="positive")
        step3_form.refresh()

    ui.button("Use this row", icon="check", on_click=use_row).props("unelevated no-caps color=primary")


def _lookup_panel(state: AppState) -> None:
    table = str(_draft(state, "s3_table", "8"))
    if table not in TABLE_CHOICES:
        table = "8"

    def _set_table(e) -> None:
        state.put_draft("s3_table", e.value)
        for k in ("s3_band", "s3_orbit", "s3_mod"):  # the old band belongs to the old table
            state.put_draft(k, None)
        step3_form.refresh()

    ui.radio(TABLE_CHOICES, value=table, on_change=_set_table).props("dense")

    try:
        bands = tables_data.get_all_distinct_bands(table)
    except Exception as ex:  # noqa: BLE001 - e.g. the workbook is missing
        ui.label(f"Could not load Table {table}: {ex}").classes("text-negative").style("white-space: pre-wrap")
        return
    band = state.get_draft("s3_band")
    band = band if band in bands else None

    def _set_band(e) -> None:
        state.put_draft("s3_band", e.value)
        state.put_draft("s3_orbit", None)  # filters belong to the previous band's rows
        state.put_draft("s3_mod", None)
        _lookup_results.refresh()

    ui.select(bands, value=band, label="Frequency band", with_input=True, on_change=_set_band) \
        .props("dense outlined").classes("w-full")
    _lookup_results(state)


# ===========================================================================
# Method 2 - calculate from station specs
# ===========================================================================
def open_pr_calculator(state: AppState,
                       on_accept: Callable[[float, float, float], None]) -> None:
    """Pr(p) calculator dialog (Rec. ITU-R S.1430 Annex 1 §2.3, Eq. 3).
    ``on_accept(pr_p, te, b)`` runs after the dialog has closed."""
    freq0 = (_f(state.itu_import.victim_freq_ghz) or _f(state.victim_lookup.freq)
             or _f(state.get_draft("vd_freq_hint2")) or _f(state.step3.selected_freq))
    te0 = Pr_calculations.lookup_te(freq0) if freq0 > 0 else 150.0
    ms0 = Pr_calculations.lookup_ms_default(freq0) if freq0 > 0 else 4.0

    result: Dict[str, Optional[Dict[str, float]]] = {"r": None}

    with ui.dialog() as dialog, ui.card().style(
            "width: 560px; max-width: 95vw; max-height: 92vh; overflow-y: auto;"):
        ui.label("Determine Pr(p) — victim station threshold interference level").classes("text-h6")
        ui.label("Rec. ITU-R S.1430, Annex 1 §2.3, Eq. 3:  Pr(p) = 10·log₁₀(k·Te·B) + NL + "
                 "10·log₁₀(10^(Ms/10) − 1) − W").classes("text-caption text-grey-7")

        _section_title("Frequency-based defaults")
        ui.label("Te and Ms have recommended values by frequency band when the station's actual "
                 "characteristics are unknown. They are pre-filled and stay editable.") \
            .classes("text-caption text-grey-7")
        freq_in = _drafted_number(state, "Victim operating frequency (GHz)", "prd_freq", freq0) \
            .classes("w-full")

        _section_title("Eq. 3 inputs")
        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            te_in = _drafted_number(state, "Te — receiving system noise temp. (K)", "prd_Te", te0) \
                .classes("col")
            b_in = _drafted_number(state, "B — reference bandwidth (Hz)", "prd_B", 12500.0, fmt="%.1f") \
                .classes("col")
        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            nl_in = _drafted_number(state, "NL — link noise contribution (dB)", "prd_NL",
                                    Pr_calculations.NL_DEFAULT).classes("col")
            ms_in = _drafted_number(state, "Ms — link performance margin (dB)", "prd_Ms", ms0) \
                .classes("col")
            w_in = _drafted_number(state, "W — equivalence factor (dB)", "prd_W", 0.0).classes("col")

        thermal_lbl = ui.label().classes("text-caption text-grey-7")
        pr_field = _readonly_field("Pr(p) =", "—")

        def _invalidate(_e=None) -> None:
            """An input changed after the last calculation: that result is stale."""
            result["r"] = None
            pr_field.set_value("—")
            thermal_lbl.set_text("")

        def fill_defaults() -> None:
            f = _f(freq_in.value)
            if f <= 0:
                ui.notify("Enter the victim operating frequency first.", type="negative")
                return
            te_in.value = Pr_calculations.lookup_te(f)       # also saved as drafts
            ms_in.value = Pr_calculations.lookup_ms_default(f)
            ui.notify(f"Te = {te_in.value:g} K and Ms = {ms_in.value:g} dB filled in for {f:g} GHz.",
                      type="info")

        def calc() -> None:
            try:
                r = Pr_calculations.pr_p(_f(te_in.value), _f(b_in.value),
                                         _f(nl_in.value, Pr_calculations.NL_DEFAULT),
                                         _f(ms_in.value), _f(w_in.value))
            except Exception as ex:  # noqa: BLE001
                _invalidate()
                ui.notify(f"Error: {ex}", type="negative", multi_line=True)
                return
            result["r"] = r
            pr_field.set_value(_fmt(r["Pr_p"], "dBW"))
            thermal_lbl.set_text(f"10·log₁₀(k·Te·B) = {r['thermal_noise_dBW']:.4f} dBW;  "
                                 f"10·log₁₀(10^(Ms/10) − 1) = {r['margin_term_dB']:.4f} dB")
            ui.notify(f"Pr(p) = {r['Pr_p']:.4f} dBW calculated.", type="positive")

        for el in (te_in, b_in, nl_in, ms_in, w_in):
            el.on_value_change(_invalidate)

        with ui.row().classes("w-full q-gutter-sm"):
            ui.button("Fill Te and Ms from frequency", on_click=fill_defaults).props("outline no-caps")
            ui.button("Calculate Pr(p)", on_click=calc).props("outline no-caps")

        ui.separator()

        def use_value() -> None:
            r = result["r"]
            if r is None:
                ui.notify("Calculate Pr(p) first.", type="negative")
                return
            te, b = _f(te_in.value), _f(b_in.value)
            dialog.close()
            on_accept(r["Pr_p"], te, b)  # after close: it refreshes the form that owns this dialog

        with ui.row().classes("w-full justify-end q-gutter-sm"):
            ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
            ui.button("Use this value as Pr(p)", on_click=use_value).props("unelevated no-caps color=primary")

    dialog.open()


def _calc_panel(state: AppState) -> None:
    vl, wr = state.victim_lookup, state.wizard_results

    def accept_gr(gain: float, _geometry: Dict[str, Any]) -> None:
        vl.gr_calculated = gain
        ui.notify(f"Gr = {gain:.4f} dBi calculated (Eq. 84-97).", type="positive")
        step3_form.refresh()

    def accept_pr(pr: float, te: float, b: float) -> None:
        vl.pr_calculated = pr
        wr.prd_result = pr
        state.put_draft("vd_te_calc", te)  # saved with Pr(p) when both are accepted
        state.put_draft("vd_b_calc", b)
        ui.notify(f"Pr(p) = {pr:.4f} dBW calculated (S.1430 Eq. 3).", type="positive")
        step3_form.refresh()

    ui.label("Work out Gr and Pr(p) from the victim station's specifications, then accept both.") \
        .classes("text-caption text-grey-7")
    with ui.row().classes("w-full q-gutter-sm"):
        ui.button("Gr calculator…", icon="calculate",
                  on_click=lambda: open_gain_calculator(state, "gw_Gr", accept_gr)) \
            .props("outline no-caps color=primary")
        ui.button("Pr(p) calculator…", icon="calculate",
                  on_click=lambda: open_pr_calculator(state, accept_pr)) \
            .props("outline no-caps color=primary")

    _readonly_field("Gr (calculated) =", _fmt(vl.gr_calculated, "dBi"))
    _readonly_field("Pr(p) (calculated) =", _fmt(vl.pr_calculated, "dBW"))

    def use_calculated() -> None:
        missing = [n for n, v in (("Gr", vl.gr_calculated), ("Pr(p)", vl.pr_calculated)) if v is None]
        if missing:
            ui.notify(f"Calculate {' and '.join(missing)} first.", type="negative")
            return
        te, b = state.get_draft("vd_te_calc"), state.get_draft("vd_b_calc")
        _save_victim(state, vl.gr_calculated, vl.pr_calculated, te, b,
                     "(calculated: Gr via Eq. 84-97, Pr(p) via S.1430 Eq. 3)")
        ui.notify(f"Gr = {wr.gr:.4f} dBi, Pr(p) = {wr.pr_p:.4f} dBW saved.", type="positive")
        step3_form.refresh()

    ui.button("Use these values as Gr and Pr(p)", icon="check", on_click=use_calculated) \
        .props("unelevated no-caps color=primary")


# ===========================================================================
# Step 3 form
# ===========================================================================
@ui.refreshable
def step3_form(state: AppState) -> None:
    wr = state.wizard_results
    ui.label("RR Appendix 7 Tables 7/8/9; Rec. ITU-R S.1430 Eq. 3; AP7 Annex 3 Eq. 84-97") \
        .classes("text-caption text-grey-7")

    # With an imported ITU station that has a receive frequency, the automatic method is the default.
    method = _draft(state, "s3_method", "itu" if _itu_receive_freqs(state) else "lookup")
    if method not in METHODS:
        method = "lookup"

    def _set_method(e) -> None:
        state.put_draft("s3_method", e.value)
        if e.value == "itu" and _itu_receive_freqs(state):
            _auto_itu(state)  # picking the automatic method applies it straight away
        step3_form.refresh()

    ui.radio(METHODS, value=method, on_change=_set_method).props("dense")
    ui.separator()

    if method == "lookup":
        _lookup_panel(state)
    elif method == "calc":
        _calc_panel(state)
    else:
        _itu_panel(state)

    ui.separator()
    if wr.gr is not None and wr.pr_p is not None:
        _status(True, f"Gr = {wr.gr:.4f} dBi, Pr(p) = {wr.pr_p:.4f} dBW {wr.gr_pr_source or ''}".rstrip())
        if wr.final_te is not None or wr.final_b is not None:
            ui.label(f"Te = {_cell(wr.final_te, '{:g}')} K, B = {_cell(wr.final_b, '{:g}')} Hz "
                     "(used by the I/N steps).").classes("text-caption text-grey-7")
        if state.step3.selected_band:
            ui.label(f"Band: {state.step3.selected_band}").classes("text-caption text-grey-7")
    else:
        _status(False, "Not yet determined.")


# ===========================================================================
# Step 4 - Lb(p)
# ===========================================================================
def step4_form(state: AppState, on_cleared: Optional[Callable[[], None]] = None) -> None:
    """Lb(p) = Pt + Gt + Gr - Pr(p), plus "Clear all".

    ``on_cleared`` is called after ``state.clear_all()`` so the caller can re-render. Without it
    the page is reloaded, which rebuilds every pane from the (now blank) AppState.
    """
    wr = state.wizard_results
    ui.label("Lb(p) = Pt + Gt + Gr − Pr(p)").classes("text-subtitle1")
    ui.label("Minimum required basic transmission loss, from the terms of Steps 1-3.") \
        .classes("text-caption text-grey-7")

    terms = [  # (name, unit, value, source, step, sign)
        ("Pt", "dBW", wr.pt, wr.pt_source, 2, "+"),
        ("Gt", "dBi", wr.gt, wr.gt_source, 1, "+"),
        ("Gr", "dBi", wr.gr, wr.gr_pr_source, 3, "+"),
        ("Pr(p)", "dBW", wr.pr_p, wr.gr_pr_source, 3, "−"),
    ]
    missing = [f"{name} (Step {step})" for name, _u, v, _s, step, _sg in terms if v is None]

    if missing:
        wr.lb_p = None  # derived value: don't leave a stale one behind
        ui.label("Lb(p) can't be calculated yet. Still missing: " + ", ".join(missing) + ".") \
            .classes("text-warning")
    else:
        lb_p = wr.pt + wr.gt + wr.gr - wr.pr_p
        wr.lb_p = lb_p
        with ui.card().classes("w-full"):
            ui.label("Lb(p)").classes("text-caption text-grey-7")
            ui.label(f"{lb_p:.2f} dB").classes("text-h4 text-primary")
            ui.separator()
            for name, unit, value, source, _step, sign in terms:
                with ui.row().classes("w-full no-wrap justify-between items-baseline"):
                    ui.label(f"{sign} {name} {source or ''}".rstrip()).classes("text-body2")
                    ui.label(f"{value:.4f} {unit}").classes("text-body2")
            ui.label(f"= {wr.pt:.4f} + {wr.gt:.4f} + {wr.gr:.4f} − ({wr.pr_p:.4f}) = {lb_p:.4f} dB") \
                .classes("text-caption text-grey-7")

    ui.separator()

    def _do_clear(dialog: ui.dialog) -> None:
        dialog.close()
        state.clear_all()
        if on_cleared is not None:
            on_cleared()
        else:
            ui.navigate.reload()

    with ui.dialog() as confirm, ui.card():
        ui.label("Clear every input and result and start over from Step 1?")
        with ui.row().classes("w-full justify-end q-gutter-sm"):
            ui.button("Cancel", on_click=confirm.close).props("flat no-caps")
            ui.button("Clear all", on_click=lambda: _do_clear(confirm)) \
                .props("unelevated no-caps color=negative")

    ui.button("Clear all", icon="delete_sweep", on_click=confirm.open).props("outline no-caps color=negative")