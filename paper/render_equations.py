"""Render stable equation images for the DOCX report."""

from pathlib import Path
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent / "equations"
OUT.mkdir(parents=True, exist_ok=True)

EQUATIONS = {
    "eq01_task": r"$\hat{\mathbf{y}}_{t+1:t+H}=f_{\theta}(\mathbf{X}_{t-L+1:t},\,\mathbf{Z}_{<t+1})$",
    "eq02_text": r"$\mathbf{z}_t=\frac{\sum_{j\in\mathcal{J}_t}\exp(-\lambda\Delta t_j)q_j\mathbf{e}_j}{\sum_{j\in\mathcal{J}_t}\exp(-\lambda\Delta t_j)q_j+\varepsilon}$",
    "eq03_ridge": r"$\hat{\boldsymbol{\beta}}=\arg\min_{\boldsymbol{\beta}}\;\|\mathbf{Y}-\mathbf{X}\boldsymbol{\beta}\|_2^2+\alpha\|\boldsymbol{\beta}\|_2^2$",
    "eq04_fusion": r"$\hat{\mathbf{y}}^{(s)}=\hat{\mathbf{y}}^{(n)}+\mathbf{W}_s\mathbf{z}_t+\mathbf{W}_f(\boldsymbol{\phi}_t\odot\mathbf{U}\mathbf{z}_t)$",
    "eq05_select": r"$\hat{\mathbf{y}}=\hat{\mathbf{y}}^{(s)}\;\mathrm{if}\;C_1\wedge C_2\wedge C_3\wedge C_4;\quad\hat{\mathbf{y}}^{(n)}\;\mathrm{otherwise}$",
    "eq06_mse": r"$\mathrm{MSE}=\frac{1}{NH}\sum_{i=1}^{N}\sum_{h=1}^{H}(y_{i,h}-\hat{y}_{i,h})^2$",
}

for name, equation in EQUATIONS.items():
    fig = plt.figure(figsize=(8.0, 0.65), dpi=300)
    fig.patch.set_alpha(0)
    fig.text(0.5, 0.5, equation, ha="center", va="center", fontsize=18, color="black")
    fig.savefig(OUT / f"{name}.png", transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

print(f"rendered {len(EQUATIONS)} equations")
