"""Pruebas automáticas sin necesidad del dataset descargado."""
import unittest
import numpy as np
import pandas as pd
from src.validacion import validar_esquema
from src.transformacion import generar_faltantes_mcar


class TestPreparacion(unittest.TestCase):
    def setUp(self):
        n = 250
        self.df = pd.DataFrame({
            "ID": np.arange(1, n + 1),
            "Warehouse_block": ["A"] * n,
            "Mode_of_Shipment": ["Ship"] * n,
            "Customer_care_calls": [3] * n,
            "Customer_rating": [4] * n,
            "Cost_of_the_Product": np.arange(n) + 100,
            "Prior_purchases": [2] * n,
            "Product_importance": ["low"] * n,
            "Gender": ["F"] * n,
            "Discount_offered": [5] * n,
            "Weight_in_gms": np.arange(n) + 700,
            "Reached.on.Time_Y.N": np.arange(n) % 2,
        })

    def test_esquema(self):
        validar_esquema(self.df)

    def test_mcar_no_modifica_original_y_guarda_verdad(self):
        antes = self.df.copy(deep=True)
        resultado, auditoria = generar_faltantes_mcar(self.df)
        pd.testing.assert_frame_equal(self.df, antes)
        self.assertEqual(int(resultado["Weight_in_gms"].isna().sum()), 25)
        self.assertEqual(int(resultado["Cost_of_the_Product"].isna().sum()), 25)
        self.assertEqual(len(auditoria), 50)
        self.assertFalse(resultado["Reached.on.Time_Y.N"].isna().any())
        for fila in auditoria.itertuples(index=False):
            verdadero = self.df.loc[self.df["ID"] == fila.ID, fila.variable].iloc[0]
            self.assertEqual(verdadero, fila.valor_original)

    def test_reproducibilidad(self):
        df1, aud1 = generar_faltantes_mcar(self.df, semilla=42)
        df2, aud2 = generar_faltantes_mcar(self.df, semilla=42)
        pd.testing.assert_frame_equal(df1, df2)
        pd.testing.assert_frame_equal(aud1, aud2)


if __name__ == "__main__":
    unittest.main()
