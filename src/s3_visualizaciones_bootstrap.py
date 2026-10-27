# Grupo 7 / S3 — Implementación original de celda 28.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

# ============================================================
# 5.1. Visualización de distribuciones bootstrap
# ============================================================

for nombre, info in resultados_boot.items():

    replicas = info["replicas"]
    columnas = info["columnas"]

    # Excluir el intercepto de los gráficos
    posiciones = [
        j for j, variable in enumerate(columnas)
        if variable != "const"
    ]

    n_graficos = len(posiciones)

    fig, axs = plt.subplots(
        n_graficos,
        1,
        figsize=(9, 2.5 * n_graficos)
    )

    axs = np.atleast_1d(axs)

    # Coeficientes originales
    coef_original = modelos_boot[nombre].params.to_numpy()

    for ax, j in zip(axs, posiciones):

        ax.hist(
            replicas[:, j],
            bins=30,
            alpha=0.75,
            edgecolor="white"
        )

        # Valor nulo
        ax.axvline(
            0,
            color="black",
            linestyle="--",
            label="Valor nulo (0)"
        )

        # Coeficiente original
        ax.axvline(
            coef_original[j],
            color="red",
            linestyle="-",
            label="Coeficiente original"
        )

        ax.set_title(columnas[j])
        ax.set_xlabel("Coeficiente estimado")
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Distribuciones bootstrap — {nombre}",
        fontsize=13,
        y=1.01
    )

    plt.tight_layout()
    plt.show()
