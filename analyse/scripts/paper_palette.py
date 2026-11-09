"""The one colour system for "Imperfectly Imperfect: Evaluating Graded Cognitive Diversity in LLM Persona Simulations".

Single source of truth for every script that colours or marks a run. Hue per MODEL FAMILY (four hues, validated
with the dataviz validator, all pairs, light surface: ALL PASS, no warning); ONE MARKER SHAPE PER RUN (shape alone
identifies a run, hue groups the family); the human reference is black and drawn as a bar, never a dot;
screen-only ladder providers (GPT-5.6, Kimi K3, DeepSeek V4-Flash, Claude Fable 5) are neutral grey, drawn as
hollow circles. Lightness steps of one hue never separate runs. Response-category colours and the diverging
(RdBu_r) and sequential (YlGnBu) maps are not run colours and live with their figures.
"""
from __future__ import annotations

FAMILY_HEX = {
    "sonnet5": "#2a78d6",    # Claude Sonnet 5 (blue)
    "gemini": "#eb6834",     # Gemini 3.5 Flash (orange)
    "dsv4pro": "#c04a8c",    # DeepSeek V4-Pro (magenta)
    "sonnet46": "#4a3aa7",   # Claude Sonnet 4.6, all four conditions (violet)
}
HUMAN_INK = "#333333"
# The human reference is a bar, never a filled point: "|" where the value axis is horizontal,
# "_" where it is vertical; drawn beneath the run marks. One size and edge width shared by every figure.
HUMAN_MARKER_ALONG_X = "|"
HUMAN_MARKER_ALONG_Y = "_"
HUMAN_MARKER_SIZE = 14
HUMAN_MARKER_EDGE = 2.4
NO_BATTERY_GREY = "#898781"   # screen-only ladder cells, no full battery (hollow circle, named on the axis)

RUN_FAMILY = {
    "CR_MAIN_SYNTH_NUM_SONNET5_S42": "sonnet5",
    "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42": "gemini",
    "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42": "dsv4pro",
    "CR_BASELINE_HUMAN_NUMERIC_CS46_S42": "sonnet46",
    "CR_BASELINE_HUMAN_LEXICAL_CS46_S42": "sonnet46",
    "CR_MAIN_SYNTH_NUMERIC_CS46_S42": "sonnet46",
    "CR_MAIN_SYNTH_LEXICAL_CS46_S42": "sonnet46",
}

RUN_COLORS = {"human": HUMAN_INK, **{label: FAMILY_HEX[fam] for label, fam in RUN_FAMILY.items()}}

# One shape per run. Claude Sonnet 4.6: H-Num circle, H-Lex square, S-Num triangle up, S-Lex diamond;
# Claude Sonnet 5 filled plus, Gemini 3.5 Flash filled X, DeepSeek V4-Pro triangle down.
SONNET46_MARKERS = {
    "CR_BASELINE_HUMAN_NUMERIC_CS46_S42": "o",
    "CR_BASELINE_HUMAN_LEXICAL_CS46_S42": "s",
    "CR_MAIN_SYNTH_NUMERIC_CS46_S42": "^",
    "CR_MAIN_SYNTH_LEXICAL_CS46_S42": "D",
}
RUN_MARKERS = {
    "CR_MAIN_SYNTH_NUM_SONNET5_S42": "P",
    "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42": "X",
    "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42": "v",
    **SONNET46_MARKERS,
}

# Configuration-ladder cells: battery families take the family hue; everything else is neutral grey.
LADDER_FAMILY_COLORS = {**{k: v for k, v in FAMILY_HEX.items() if k != "sonnet46"}, "other": NO_BATTERY_GREY}
