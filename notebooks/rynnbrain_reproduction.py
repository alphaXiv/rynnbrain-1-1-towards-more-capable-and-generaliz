# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.10",
#     "matplotlib>=3.8",
#     "numpy>=1.26",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # RynnBrain 1.1: run-backed reproduction

    **Verdict: partially reproduced.** This notebook packages the terminal
    measurements from 316 successful Kubernetes runs. It reproduces strong
    adherence to the released embodied interfaces and partial visual-spatial
    conditioning, while exact camera-gauge controls show that the native-3D
    focal response is not a coherent projective computation.

    The values below are embedded directly from successful aggregate run logs;
    the notebook performs no model inference and needs no credentials.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell
def _():
    protocol_rows = [
        {"Task": "Contact pose (native prompt)", "RynnBrain 122B": "8/8", "Qwen 122B": "0/8"},
        {"Task": "Official localization", "RynnBrain 122B": "7/8", "Qwen 122B": "3/8"},
        {"Task": "Native 3D (zero-shot)", "RynnBrain 122B": "7/8", "Qwen 122B": "2/8"},
    ]
    contact_data = {
        "labels": ["Baseline", "0.8× inset", "1.25× crop", "+10% x", "90°", "180°", "Gray", "Invert", "Blur", "White", "Black"],
        "Position error (pixels)": {
            "RynnBrain": [56.52, 36.28, 73.75, 70.74, 70.70, 56.21, 61.29, 58.40, 47.38, 119.53, 110.77],
            "Qwen": [64.68, 53.78, 77.21, 64.06, 46.31, 60.52, 64.11, 57.48, 66.39, 153.39, 163.52],
        },
        "Angle error (degrees)": {
            "RynnBrain": [7.74, 14.68, 11.38, 8.80, 28.74, 21.33, 17.25, 15.33, 24.81, 39.96, 49.24],
            "Qwen": [24.04, 45.01, 20.56, 34.84, 36.19, 47.16, 27.54, 39.81, 47.81, 44.16, 49.30],
        },
    }
    gauge_data = {
        "scale": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        "Chair — RynnBrain": [1.14, 1.24, 1.41, 1.51, 1.69, 1.93, 2.21, 2.44],
        "Chair — Qwen": [1.17, 1.16, 1.16, 1.16, 1.18, 1.34, 1.31, 1.16],
        "Sofa — RynnBrain": [3.10, 3.32, 3.64, 4.01, 4.41, 5.31, 6.04, 6.94],
        "Sofa — Qwen": [3.36, 3.36, 3.36, 3.36, 3.36, 3.36, 3.36, 3.43],
    }
    campaign_data = {"successful": 316, "cancelled": 15, "failed": 6, "wall_hours": 11.8785, "max_gpus": 16}
    return campaign_data, contact_data, gauge_data, protocol_rows


@app.cell
def _(mo, protocol_rows):
    mo.vstack(
        [
            mo.md("## Headline: protocol adherence"),
            mo.ui.table(protocol_rows, pagination=False),
            mo.md(
                "RynnBrain's clearest reproduced advantage is reliable use of the "
                "released task protocols. These rates alone do not establish grounding; "
                "the ablations below test that boundary."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    robustness_metric = mo.ui.dropdown(
        options=["Position error (pixels)", "Angle error (degrees)"],
        value="Position error (pixels)",
        label="Contact robustness metric",
    )
    mo.vstack([mo.md("## Robustness explorer"), robustness_metric])
    return (robustness_metric,)


@app.cell
def _(contact_data, np, plt, robustness_metric):
    _metric = robustness_metric.value
    _x = np.arange(len(contact_data["labels"]))
    _fig, _ax = plt.subplots(figsize=(10.5, 4.5))
    _ax.plot(_x, contact_data[_metric]["RynnBrain"], "o-", color="#2463a6", linewidth=2.3, label="RynnBrain 122B")
    _ax.plot(_x, contact_data[_metric]["Qwen"], "s--", color="#e76f51", linewidth=2.0, label="Qwen 122B")
    _ax.set_xticks(_x, contact_data["labels"], rotation=28, ha="right")
    _ax.set_ylabel(f"{_metric} — lower is better")
    _ax.set_title("Contact-pose robustness against analytic transformed references")
    _ax.grid(axis="y", alpha=0.25)
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(gauge_data, mo, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for _ax, _scene in zip(_axes, ["Chair", "Sofa"]):
        _ax.plot(gauge_data["scale"], gauge_data[f"{_scene} — RynnBrain"], "o-", color="#2463a6", linewidth=2.3, label="RynnBrain 122B")
        _ax.plot(gauge_data["scale"], gauge_data[f"{_scene} — Qwen"], "s--", color="#e76f51", linewidth=2.0, label="Qwen 122B")
        _ax.set_title(f"Fixed {_scene.lower()} image")
        _ax.set_xlabel("Raw scale λ in λK")
        _ax.set_ylabel("Predicted depth (m)")
        _ax.grid(alpha=0.25)
    _axes[0].legend(frameon=False)
    _fig.suptitle("Exact gauge diagnostic: λK and K are the same camera", y=1.03, weight="bold")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md("## Diagnostic: ordered focal response is not projective geometry"),
            _fig,
            mo.md(
                "A coherent projective parser would be invariant to uniform matrix scale. "
                "RynnBrain instead moves depth by 1.30 m on the chair and 3.84 m on the sofa. "
                "Qwen is flatter here, but separate denominator and white-frame controls show "
                "sparse token-triggered templates rather than reliable normalization."
            ),
        ]
    )
    return


@app.cell
def _(campaign_data, mo):
    _success_rate = 100 * campaign_data["successful"] / (
        campaign_data["successful"] + campaign_data["cancelled"] + campaign_data["failed"]
    )
    _throughput = campaign_data["successful"] / campaign_data["wall_hours"]
    mo.md(
        f"""
        ## Campaign evidence

        - **{campaign_data['successful']}** successful Kubernetes runs with terminal summaries
        - **{campaign_data['wall_hours']:.4f} hours** observed campaign wall time
        - **{campaign_data['max_gpus']} GPUs** maximum concurrent allocation
        - **{_success_rate:.1f}%** terminal success rate and **{_throughput:.2f}** successful runs per observed wall-hour

        Cancelled and failed jobs are excluded from every scientific comparison.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Assessment

    The reproduction is **partial**. Successful runs support strong protocol
    adherence, deterministic fixed-input behavior, and genuine but imperfect
    visual-spatial/temporal conditioning on the released examples. They do not
    establish benchmark accuracy because the release lacks task ground truth.
    Exact projective controls further show that RynnBrain's stable focal-depth
    association is a learned magnitude heuristic, not coherent camera arithmetic.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
