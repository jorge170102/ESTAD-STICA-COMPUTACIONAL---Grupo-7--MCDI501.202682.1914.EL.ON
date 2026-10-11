import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from src.inferencia import mostrar_estimaciones, prueba_peso_por_entrega, prueba_modo_por_entrega

class TestInferencia(unittest.TestCase):
    def setUp(self):
        r=np.random.default_rng(42);n=600
        self.df=pd.DataFrame({
            "Weight_in_gms": np.r_[r.normal(4500,350,300),r.normal(4000,350,300)],
            "Cost_of_the_Product":r.normal(210,40,n),
            "Discount_offered":r.integers(1,65,n),
            "Reached.on.Time_Y.N":np.repeat([0,1],300),
            "Mode_of_Shipment":np.tile(["Ship","Road","Flight"],200)
        })
        self.df.loc[:59,"Weight_in_gms"]=np.nan
        self.df.loc[:59,"Cost_of_the_Product"]=np.nan

    def test_ic_con_faltantes(self):
        a=mostrar_estimaciones(self.df)
        self.assertEqual(a.loc["Weight_in_gms","N válido"],540)
        self.assertEqual(a.loc["Cost_of_the_Product","N válido"],540)
        self.assertEqual(a.loc["Discount_offered","N válido"],600)
        self.assertTrue((a["IC inferior"]<a["Media estimada"]).all())
        self.assertTrue((a["Media estimada"]<a["IC superior"]).all())

    def test_welch_signo_y_tamano_efecto(self):
        a=prueba_peso_por_entrega(self.df, mostrar_graficos=False)
        self.assertLess(a.loc[0,"Diferencia g (fuera - a tiempo)"],0)
        self.assertLess(a.loc[0,"p"],.05)
        self.assertTrue(np.isfinite(a.loc[0,"d Cohen"]))

    def test_chi_cuadrado(self):
        a=prueba_modo_por_entrega(self.df, mostrar_graficos=False)
        self.assertGreaterEqual(a.loc[0,"Esperado mínimo"],5)
        self.assertTrue(0<=a.loc[0,"p"]<=1)

if __name__=='__main__':unittest.main()
