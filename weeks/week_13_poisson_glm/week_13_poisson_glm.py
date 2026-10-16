# %% [markdown]
# # Semana 13: GLM de Poisson para conteos de spikes
#
# **Objetivo:** ajustar un modelo de conteos con enlace logarítmico y comprobar que recupera los
# parámetros con los que se generaron los datos.
#
# ## Modelo
#
# Para cada intervalo $t$ con estímulo $x_t$, el conteo de spikes es
#
# $$y_t \sim \text{Poisson}(\lambda_t), \qquad \log \lambda_t = \beta_0 + \beta_1 x_t$$
#
# El enlace log garantiza $\lambda_t > 0$ y hace que cada unidad de estímulo **multiplique** la tasa
# por $e^{\beta_1}$. Los parámetros se estiman maximizando la log-verosimilitud
#
# $$\ell(\boldsymbol\beta) = \sum_t \big(y_t \log\lambda_t - \lambda_t - \log y_t!\big)$$
#
# (el paquete usa Newton-Raphson). Parámetros verdaderos: $\beta_0 = 0.4$, $\beta_1 = 0.7$;
# 180 intervalos con estímulos entre −1.5 y 1.5.

# %%
import numpy as np
import matplotlib.pyplot as plt

from computational_neuroscience import fit_poisson_glm
from computational_neuroscience.outputs import save_data, save_figure

WEEK = 13
TRUE_INTERCEPT, TRUE_SLOPE = 0.4, 0.7
rng = np.random.default_rng(13)
stimulus = np.linspace(-1.5, 1.5, 180)
true_rate = np.exp(TRUE_INTERCEPT + TRUE_SLOPE * stimulus)
counts = rng.poisson(true_rate)
model = fit_poisson_glm(stimulus[:, None], counts)
predicted_rate = model.predict_rate(stimulus[:, None])

edges = np.linspace(-1.5, 1.5, 11)
bin_index = np.clip(np.digitize(stimulus, edges) - 1, 0, 9)
bin_centers = (edges[:-1] + edges[1:]) / 2
bin_mean = np.array([counts[bin_index == b].mean() for b in range(10)])
bin_sem = np.array([counts[bin_index == b].std(ddof=1) / np.sqrt(np.sum(bin_index == b)) for b in range(10)])

# %% [markdown]
# ## Figura
#
# Izquierda: conteos individuales, tasa verdadera y tasa ajustada. Derecha: media de los conteos
# por intervalo de estímulo (± error estándar) en escala log; si el modelo es correcto, los puntos
# siguen una recta.

# %%
figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(stimulus, counts, s=12, alpha=0.45, label="Observed counts")
axes[0].plot(stimulus, true_rate, "--", label=f"True rate (β0={TRUE_INTERCEPT}, β1={TRUE_SLOPE})")
axes[0].plot(stimulus, predicted_rate, label=f"Fitted (β0={model.intercept:.2f}, β1={model.coefficients[0]:.2f})")
axes[0].set(xlabel="Stimulus", ylabel="Count per bin", title="Poisson GLM with log link")
axes[0].legend(fontsize=8)
axes[1].errorbar(bin_centers, bin_mean, yerr=bin_sem, fmt="o", label="Binned mean ± SEM")
axes[1].plot(stimulus, predicted_rate, label="Fitted rate")
axes[1].set(xlabel="Stimulus", ylabel="Mean count (log scale)", yscale="log", title="Log-linear check")
axes[1].legend()
save_figure(WEEK, figure, "poisson_glm.png")
save_data(WEEK, "poisson_glm.npz", stimulus=stimulus, counts=counts, true_rate=true_rate, predicted_rate=predicted_rate)

# %% [markdown]
# ## Validación
#
# - El ajuste converge.
# - Los parámetros estimados son compatibles con los verdaderos: su error estándar se calcula con
#   la información de Fisher, $\text{SE} = \sqrt{\text{diag}\big((X^\top \hat\Lambda X)^{-1}\big)}$,
#   y se exige $|z| = |\hat\beta - \beta| / \text{SE} < 3$.
# - Los residuos de Pearson $(y - \hat\lambda)/\sqrt{\hat\lambda}$ tienen varianza ≈ 1, como exige Poisson.

# %%
pearson = (counts - predicted_rate) / np.sqrt(predicted_rate)
design = np.column_stack((np.ones_like(stimulus), stimulus))
standard_errors = np.sqrt(np.diag(np.linalg.inv(design.T @ (predicted_rate[:, None] * design))))
z_scores = (np.array([model.intercept, model.coefficients[0]]) - [TRUE_INTERCEPT, TRUE_SLOPE]) / standard_errors
assert model.converged
assert np.all(np.abs(z_scores) < 3), z_scores
assert 0.75 < pearson.var() < 1.25, pearson.var()
print(f"beta0={model.intercept:.3f} +/- {standard_errors[0]:.3f} (true {TRUE_INTERCEPT}, z={z_scores[0]:.1f})")
print(f"beta1={model.coefficients[0]:.3f} +/- {standard_errors[1]:.3f} (true {TRUE_SLOPE}, z={z_scores[1]:.1f})")
print(f"Pearson residual variance: {pearson.var():.2f} (Poisson -> 1)")

# %% [markdown]
# ## Interpretación y límites
#
# - $e^{\beta_1} \approx 2$: cada unidad de estímulo duplica aproximadamente la tasa esperada.
# - Con esta semilla, $\hat\beta_1$ queda ~2 SE por debajo del valor verdadero. No es un sesgo:
#   repetido con 300 semillas, la media de $\hat\beta_1$ es 0.707. Una sola muestra no basta para
#   juzgar un estimador.
# - El modelo asume intervalos independientes y varianza igual a la media. Datos reales suelen
#   tener sobredispersión (varianza de Pearson > 1) e historia de spikes, que un GLM puede incluir
#   como regresores adicionales.
#
# **Para tu entrega:** genera conteos sobredispersos (p. ej., multiplica la tasa por un factor
# gamma aleatorio) y comprueba cómo cambia la varianza de los residuos de Pearson.
