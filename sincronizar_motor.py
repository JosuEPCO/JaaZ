"""Copia el motor de cálculo de JaaZ (la versión de escritorio, carpeta superior) a esta carpeta.

La versión de Streamlit usa exactamente el mismo motor validado (pruebas de Mrode, ENDOG, REML =
ANOVA). Se copia en lugar de importarlo desde la carpeta superior para que esta carpeta se pueda
publicar sola en Streamlit Community Cloud. Ejecute este script cada vez que cambie el motor:

    python sincronizar_motor.py
"""
import os
import shutil

AQUI = os.path.dirname(os.path.abspath(__file__))
ORIGEN = os.path.join(os.path.dirname(AQUI), "pedigree_ultra")


def copiar(origen, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    shutil.copy2(origen, destino)
    print("  ", os.path.relpath(destino, AQUI))


def main():
    if not os.path.isdir(ORIGEN):
        raise SystemExit(f"No se encontró el motor en {ORIGEN}")
    motor = os.path.join(AQUI, "motor")
    print("Motor de cálculo:")
    copiar(os.path.join(ORIGEN, "archivos.py"), os.path.join(motor, "archivos.py"))
    gen = os.path.join(ORIGEN, "genetica")
    for nombre in sorted(os.listdir(gen)):
        if nombre.endswith(".py"):
            copiar(os.path.join(gen, nombre), os.path.join(motor, "genetica", nombre))
    with open(os.path.join(motor, "__init__.py"), "w", encoding="utf-8") as f:
        f.write('"""Motor de cálculo de JaaZ (copia de pedigree_ultra/; se actualiza con sincronizar_motor.py)."""\n')
    print("Datos de ejemplo:")
    ej = os.path.join(os.path.dirname(AQUI), "ejemplos")
    for nombre in sorted(os.listdir(ej)):
        if os.path.isfile(os.path.join(ej, nombre)):
            copiar(os.path.join(ej, nombre), os.path.join(AQUI, "ejemplos", nombre))
    print("Listo.")


if __name__ == "__main__":
    main()
