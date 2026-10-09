# JaaZ · versión Streamlit

La misma herramienta de genética cuantitativa de JaaZ 1.4, preparada para publicarse en
**Streamlit Community Cloud** (o en cualquier servidor con Streamlit). Usa exactamente el mismo
motor de cálculo que la versión de escritorio (carpeta `motor/`, validado con los ejemplos de Mrode,
los valores del manual de ENDOG y REML = ANOVA en datos balanceados).

Incluye: hoja de datos editable, estadística descriptiva, heredabilidad (5 métodos), repetibilidad
(4), correlación genética (4, con tablero y repetibilidad de cada rasgo), visor de pedigrí,
calidad y verificación del pedigrí, parentesco, apareamientos, BLUP, endogamia y Ne, fundadores y
ancestros, intervalo generacional, subpoblaciones y estructura de rebaños; resultados exportables a
Excel.

## Publicar en Streamlit Community Cloud

1. Cree un repositorio en GitHub y suba **el contenido de esta carpeta** como raíz del repositorio
   (deben quedar `app.py`, `requirements.txt` y la carpeta `.streamlit/` arriba de todo).
   No suba la carpeta `.venv` (ya está en `.gitignore`).
2. Entre a <https://share.streamlit.io>, inicie sesión con GitHub y pulse **Create app**.
3. Elija el repositorio, la rama (`main`) y como *Main file path* escriba `app.py`.
4. En **Advanced settings** elija **Python 3.12 o superior** (numpy 2.5 lo requiere).
5. Pulse **Deploy**. La primera vez tarda unos minutos en instalar las dependencias.

El tema oscuro «obsidiana», el logotipo y el menú por secciones se cargan solos
(`.streamlit/config.toml`, `recursos/`).

**Privacidad.** Cada visitante trabaja en su propia sesión: los archivos que sube y sus
resultados viven en la memoria de esa sesión, no se guardan en disco ni se comparten, y se pierden
al cerrar la pestaña. Para conservarlos, descargue el libro o los resultados en Excel.

## Usarla en su computadora

```cmd
iniciar_streamlit.bat
```

La primera vez crea `.venv` e instala las dependencias (requiere internet y Python 3.12 o
superior); luego abre la app en el navegador (http://localhost:8501).

## Mantener el motor al día

El motor es una copia de la carpeta `pedigree_ultra/` de la versión de escritorio, para que esta
carpeta se pueda publicar sola. Si cambia el motor, actualice la copia (y los ejemplos) con:

```cmd
python sincronizar_motor.py
```

## Pruebas

```cmd
.venv\Scripts\python -m unittest discover -s pruebas -v
```

Abren cada página con `streamlit.testing`, eligen cada método de los 13 análisis, pulsan
«Calcular» y verifican que el resultado aparezca sin errores.

## Estructura

```
app.py                  punto de entrada: configuración, logotipo y menú por secciones
requirements.txt        dependencias con versiones fijas
.streamlit/config.toml  tema «obsidiana»
jaaz_st/                interfaz
  definiciones.py       análisis, métodos, columnas y opciones (como en la versión de escritorio)
  analisis.py           formulario genérico, validación, cálculo con progreso
  vista.py              resultados: indicadores, tablero, tablas, gráficos, fórmulas, Excel
  graficos.py           gráficos con Altair (paleta validada para daltonismo)
  paginas.py            Inicio, Hoja de datos, Visor de pedigrí, Resultados y Ayuda
  estado.py             datos de la sesión, lectura de archivos, detección de columnas
  estilo.py             estilo visual
motor/                  motor de cálculo (copia de pedigree_ultra/)
ejemplos/               datos de ejemplo
recursos/               logotipo
pruebas/                pruebas de la app
sincronizar_motor.py    actualiza motor/ y ejemplos/ desde la versión de escritorio
iniciar_streamlit.bat   lanzador local para Windows
```

### Diferencias con la versión de escritorio

- El visor de pedigrí muestra el árbol de un animal (ancestros y descendientes) en lugar del
  diagrama navegable de toda la población.
- La hoja de datos permite editar, agregar y borrar filas; no tiene columna calculada, ordenar ni
  buscar y reemplazar (prepare esos cambios en Excel).
- No hay atajos de teclado ni buscador Ctrl+K: la navegación es el menú lateral de Streamlit.
