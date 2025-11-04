# Suite de Evaluaciones Económicas en Salud (Streamlit)

Aplicación Streamlit con módulos independientes para:
- COI (Costo de la Enfermedad)
- BIA (Impacto Presupuestario)
- ROI (Retorno sobre la Inversión)
- CC (Comparación de Costos)
- CMA (Minimización de Costos)
- CCA (Costo‑Consecuencia)
- CEA (Costo‑Efectividad)
- CUA (Costo‑Utilidad)
- CBA (Costo‑Beneficio)

## Uso local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy en Streamlit Community Cloud
1. Crea un repositorio en GitHub y sube `app.py` y `requirements.txt` (y este README).
2. Entra a https://share.streamlit.io, conecta tu cuenta GitHub y selecciona el repo.
3. Establece `app.py` como *Main file*.
4. Deploy.

## Notas
- Los módulos 7, 8 y 9 son independientes (no comparten estado).
- Se incluyen descargas CSV y gráficas básicas en cada módulo.
