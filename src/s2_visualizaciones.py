"""Visualizaciones de S2: solo presentación, sin alterar cálculos ni CSV."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def graficar_bootstrap(variables, bootstrap_medias, distribuciones_medias, NOMBRES):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, variable in zip(axes, variables):
        r = bootstrap_medias.loc[variable]
        ax.hist(distribuciones_medias[variable], bins=35, color="gray", alpha=0.75)
        ax.axvline(r.media, color="black", label="Media S1")
        for campo, color, estilo, etiqueta in [("ic_t", "blue", ":", "IC clásico"),
                ("percentil", "orange", "--", "IC percentil"), ("bca", "green", "-.", "IC BCa")]:
            ax.axvline(r[campo + "_inf"], color=color, ls=estilo, label=etiqueta)
            ax.axvline(r[campo + "_sup"], color=color, ls=estilo)
        ax.set(title=NOMBRES[variable], xlabel="Media remuestreada", ylabel="Frecuencia")
        ax.legend(fontsize=7)
    plt.show()
    plt.close(fig)


def graficar_correlaciones(correlaciones, distribuciones_rho, AZUL, VERDE, NARANJA):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, (nombre, r) in zip(axes, correlaciones.iterrows()):
        ax.hist(distribuciones_rho[nombre], bins=35, color=AZUL, alpha=0.8)
        ax.axvline(r.rho_s1, color="black", label="Spearman S1")
        ax.axvline(r.ic_inf, color=VERDE, ls="--", label="IC95% bootstrap")
        ax.axvline(r.ic_sup, color=VERDE, ls="--")
        # Si el cero queda muy lejos, se indica en texto para no comprimir el histograma.
        if r.ic_inf - r.ancho_ic <= 0 <= r.ic_sup + r.ancho_ic:
            ax.axvline(0, color=NARANJA, ls=":", label="Cero")
        else:
            ax.text(0.03, 0.9, "0 fuera del intervalo", transform=ax.transAxes, fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        ax.set(title=nombre, xlabel="Spearman remuestreado", ylabel="Frecuencia")
        ax.legend(fontsize=7, loc="upper right")
    plt.show()
    plt.close(fig)


def graficar_ic_correlaciones(correlaciones, AZUL, NARANJA):
    # Comparación de correlaciones e IC bootstrap al 95 %
    # Se reutilizan los resultados de 10.000 remuestras.

    tabla_corr = correlaciones.reset_index().copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    y = np.arange(len(tabla_corr))

    rho = tabla_corr["rho_s1"].to_numpy()
    ic_inf = tabla_corr["ic_inf"].to_numpy()
    ic_sup = tabla_corr["ic_sup"].to_numpy()

    errores = np.vstack([
        rho - ic_inf,
        ic_sup - rho
    ])

    ax.errorbar(
        rho, y,
        xerr=errores,
        fmt="o",
        color=AZUL,
        ecolor=AZUL,
        capsize=5,
        markersize=7,
        linewidth=2,
        label="Spearman S1 e IC bootstrap 95 %"
    )

    ax.axvline(
        0,
        color=NARANJA,
        linestyle="--",
        linewidth=1.5,
        label="Correlación nula"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(tabla_corr["relacion"])
    ax.invert_yaxis()

    ax.set_xlabel("Coeficiente de correlación de Spearman")
    ax.set_title("Estabilidad de correlaciones de S1")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    plt.show()

def graficar_montecarlo_convergencia(mc_lognormal, delta_s1, ic_delta_s1, M, AZUL, NARANJA):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    axes[0].hist(mc_lognormal, bins=40, color=AZUL, alpha=0.8)
    axes[0].axvline(delta_s1, color="black", label="Diferencia S1")
    for i, limite in enumerate(ic_delta_s1):
        axes[0].axvline(limite, color=NARANJA, ls="--", label="IC95% S1" if i == 0 else None)
    axes[0].set(title="Diferencias en estudios simulados", xlabel="Fuera de plazo − a tiempo (g)", ylabel="Frecuencia")
    axes[0].legend(fontsize=8)
    eje = np.arange(1, M+1)
    acumulada = np.cumsum(mc_lognormal) / eje
    axes[1].plot(eje[99:], acumulada[99:], color=AZUL, label="Media acumulada")
    axes[1].axhline(delta_s1, color="black", ls="--", label="Media teórica = diferencia S1")
    axes[1].set(title="Convergencia de la media", xlabel="Iteraciones", ylabel="Diferencia media (g)")
    axes[1].legend(fontsize=8)
    plt.show()
    plt.close(fig)


def graficar_cuantiles(diagnostico_distribuciones, AZUL, NARANJA):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    for ax, grupo in zip(axes, ["A tiempo", "Fuera de plazo"]):
        sub = diagnostico_distribuciones.loc[diagnostico_distribuciones.grupo == grupo]
        ax.plot(sub.cuantil_empirico_g, sub.cuantil_empirico_g, "k--", label="Ajuste perfecto")
        ax.plot(sub.cuantil_empirico_g, sub.cuantil_lognormal_g, "o-", color=AZUL, label="Lognormal")
        ax.plot(sub.cuantil_empirico_g, sub.cuantil_gamma_g, "s-", color=NARANJA, label="Gamma")
        ax.set(title=grupo, xlabel="Cuantiles observados (g)", ylabel="Cuantiles teóricos (g)")
        ax.legend(fontsize=8)
    plt.show(); plt.close(fig)


def graficar_cobertura(cobertura_welch_mc, AZUL, NARANJA):
    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    x = np.arange(2)
    y = cobertura_welch_mc.cobertura_ic_welch_95_pct.to_numpy()
    e = 1.96 * cobertura_welch_mc.error_montecarlo_cobertura_pp.to_numpy()
    ax.errorbar(x, y, yerr=e, fmt="o", capsize=6, color=AZUL,
                label="Cobertura estimada ± 1,96 EE Monte Carlo")
    ax.axhline(95, ls="--", color=NARANJA, label="Cobertura nominal 95 %")
    ax.set(xticks=x, xticklabels=["Lognormal", "Gamma"],
           ylabel="Cobertura empírica (%)", title="Validación condicional de los IC de Welch")
    ax.set_ylim(92, 98)
    ax.legend(loc="lower left", fontsize=8)
    plt.show(); plt.close(fig)


def graficar_extremos(sensibilidad_extremos, AZUL, NARANJA):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    for ax, (_, r) in zip(axes, sensibilidad_extremos.iterrows()):
        for y, valor, inf, sup, color in [
            (1, r.base, r.ic_base_inf, r.ic_base_sup, AZUL),
            (0, r.escenario, r.ic_escenario_inf, r.ic_escenario_sup, NARANJA)]:
            ax.errorbar(valor, y, xerr=[[valor-inf], [sup-valor]], fmt="o", color=color, capsize=5)
        ax.set(yticks=[0, 1], yticklabels=["Sin extremos", "Todos los datos"],
               title=r.estimador.replace(" (", "\n("), xlabel="Estimación e IC95%", ylim=(-0.5, 1.5))
    plt.show()
    plt.close(fig)


def graficar_modelos(mc_lognormal, mc_gamma, dist_delta_boot, delta_s1, AZUL, NARANJA, VERDE):
    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    for valores, nombre, color in [(mc_lognormal, "Lognormal", AZUL),
            (mc_gamma, "Gamma", NARANJA), (dist_delta_boot, "Bootstrap", VERDE)]:
        ax.hist(valores, bins=40, density=True, histtype="step", lw=1.5, label=nombre, color=color)
    ax.axvline(delta_s1, color="black", ls="--", label="Diferencia S1")
    ax.set(title="Sensibilidad de la diferencia de peso", xlabel="Fuera de plazo − a tiempo (g)", ylabel="Densidad")
    ax.legend(fontsize=8)
    plt.show()
    plt.close(fig)



def graficar_escenarios(sensibilidad_entradas, AZUL):
    fig, ax = plt.subplots(figsize=(9, 3.3), constrained_layout=True)
    for i, (nombre, r) in enumerate(sensibilidad_entradas.iterrows()):
        ax.errorbar(r.media, i, xerr=[[r.media-r.p2_5], [r.p97_5-r.media]], fmt="o", capsize=5, color=AZUL)
    ax.set(yticks=range(len(sensibilidad_entradas)), yticklabels=sensibilidad_entradas.index,
           xlabel="Diferencia de peso (g), media e intervalo de variación del 95 %",
           title="Cambiar la diferencia de entrada desplaza el resultado")
    plt.show()
    plt.close(fig)


def graficar_recortadas(delta_s1, ic_delta_s1, delta_recortada, ic_recortada, AZUL, NARANJA):
    fig, ax = plt.subplots(figsize=(9, 3.3), constrained_layout=True)
    for y, est, lo, hi, nom, color in [
        (1, delta_s1, *ic_delta_s1, "Media clásica · Welch S1", AZUL),
        (0, delta_recortada, *ic_recortada, "Media recortada 10 % · bootstrap", NARANJA),
    ]:
        ax.errorbar(est, y, xerr=[[est-lo], [hi-est]], fmt="o", capsize=5, color=color, label=nom)
    ax.axvline(0, color="black", linestyle="--")
    ax.set(yticks=[0, 1], yticklabels=["Recortada", "Welch S1"],
           xlabel="Diferencia fuera de plazo - a tiempo (g)",
           title="Sensibilidad de la diferencia de peso a extremos")
    ax.legend(loc="best", fontsize=8)
    plt.show(); plt.close(fig)
