"""
Step tracker for the Left Pane.

A vertical list of buttons, one per wizard step. Clicking one sets
``state.current_step``, re-renders the tracker so the active step is highlighted,
and calls ``on_change(step_number)`` so the caller can refresh whatever depends on
the current step (e.g. the input form below the tracker).

To add Steps 6-9 later, just append to ``STEPS`` - nothing else needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from nicegui import ui

from ui.state import AppState


@dataclass(frozen=True)
class Step:
    number: int
    title: str


# Titles follow the Streamlit app's headers (Steps 1-5).
STEPS = (
    Step(1, "Gt — interferer antenna gain"),
    Step(2, "Pt — transmit power"),
    Step(3, "Gr and Pr(p) — victim station"),
    Step(4, "Result — Lb(p)"),
    Step(5, "Coordination distance"),
)


def step_title(number: int) -> str:
    return next((s.title for s in STEPS if s.number == number), f"Step {number}")


def set_step(state: AppState, number: int) -> int:
    """Set ``state.current_step``, clamped to the valid range. Returns the value set."""
    state.current_step = max(STEPS[0].number, min(STEPS[-1].number, int(number)))
    return state.current_step


def build_step_tracker(
    state: AppState,
    on_change: Optional[Callable[[int], None]] = None,
) -> Callable[[], None]:
    """Render the tracker in the current NiceGUI context.

    Returns the tracker's ``refresh`` function, so code that changes
    ``state.current_step`` elsewhere (e.g. a future Next button) can call it to
    re-highlight the active step.
    """

    @ui.refreshable
    def tracker() -> None:
        with ui.column().classes("w-full gap-1"):
            for step in STEPS:
                active = step.number == state.current_step
                ui.button(
                    f"{step.number}.  {step.title}",
                    on_click=lambda n=step.number: _select(n),
                ).props(
                    # active step: solid; others: flat. no-caps keeps "Gt"/"Pr(p)" readable.
                    f"align=left no-caps {'unelevated color=primary' if active else 'flat color=grey-8'}"
                ).classes("w-full")

    def _select(number: int) -> None:
        set_step(state, number)
        tracker.refresh()
        if on_change is not None:
            on_change(state.current_step)

    tracker()
    return tracker.refresh