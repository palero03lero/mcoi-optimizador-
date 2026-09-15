import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

st.set_page_config(page_title="Optimizador MCOI", layout="wide")

st.title("Optimizador Interactivo de Programación Lineal")
st.markdown("""
Esta aplicación permite modelar y resolver problemas de Programación Lineal Continua.
*   **Añade o quita** variables y restricciones desde el panel lateral.
*   **Visualización:** Si el modelo tiene exactamente 2 variables, se generará el método gráfico automáticamente.
*   **Resultados:** Se evalúa la solución óptima y el estado (saturada/inactiva) de las restricciones.
""")

# --- CONFIGURACIÓN DINÁMICA (Panel Lateral) ---
st.sidebar.header("Configuración del Modelo")
tipo_opt = st.sidebar.selectbox("Objetivo", ["Maximizar", "Minimizar"])
num_vars = st.sidebar.number_input("Número de Variables de Decisión", min_value=2, value=2, step=1)
num_cons = st.sidebar.number_input("Número de Restricciones", min_value=1, value=3, step=1)

# --- ENTRADA DE DATOS: FUNCIÓN OBJETIVO ---
st.header("1. Función Objetivo")
cols_obj = st.columns(num_vars)
C = []
for i in range(num_vars):
    val = cols_obj[i].number_input(f"Coeficiente de X{i+1}", value=1.0, step=0.1, key=f"c_{i}")
    C.append(val)

# --- ENTRADA DE DATOS: RESTRICCIONES ---
st.header("2. Restricciones")
st.write("Introduce los coeficientes tecnológicos, el sentido de la desigualdad y el término independiente (capacidad).")
A = []
B = []
signos = []

for i in range(num_cons):
    cols_cons = st.columns(num_vars + 2)
    fila_A = []
    for j in range(num_vars):
        val_a = cols_cons[j].number_input(f"X{j+1} (R{i+1})", value=1.0, step=0.1, key=f"a_{i}_{j}")
        fila_A.append(val_a)
    
    signo = cols_cons[num_vars].selectbox("Signo", ["<=", ">=", "="], key=f"sig_{i}")
    val_b = cols_cons[num_vars+1].number_input(f"Término Indep. (B{i+1})", value=10.0, step=1.0, key=f"b_{i}")
    
    A.append(fila_A)
    signos.append(signo)
    B.append(val_b)
# NUEVO CÓDIGO: Mostrar la ecuación en tiempo real
    ecuacion_str = f"**{fila_A[0]}** X1 "
    for j in range(1, num_vars):
        signo_var = "+" if fila_A[j] >= 0 else "-"
        ecuacion_str += f"{signo_var} **{abs(fila_A[j])}** X{j+1} "
    ecuacion_str += f"{signo} **{val_b}**"
    st.markdown(f"> *Ecuación interpretada:* {ecuacion_str}")
# --- LÓGICA DE RESOLUCIÓN (MÉTODO SIMPLEX / PUNTO INTERIOR) ---
if st.button("Resolver Modelo", type="primary"):
    
    # Preparar matrices para SciPy (linprog minimiza por defecto)
    C_opt = np.array(C) if tipo_opt == "Minimizar" else -np.array(C)
    
    A_ub, B_ub, A_eq, B_eq = [], [], [], []
    
    for i in range(num_cons):
        if signos[i] == "<=":
            A_ub.append(A[i])
            B_ub.append(B[i])
        elif signos[i] == ">=":
            # Invertir signo para adaptar a <=
            A_ub.append([-x for x in A[i]])
            B_ub.append(-B[i])
        else:
            A_eq.append(A[i])
            B_eq.append(B[i])
            
    # Limites (Restricciones de no negatividad Xi >= 0)
    limites = [(0, None) for _ in range(num_vars)]
    
    # Ejecutar algoritmo HiGHS (Dual Simplex / Interior Point)
    res = linprog(c=C_opt, 
                  A_ub=A_ub if A_ub else None, b_ub=B_ub if B_ub else None,
                  A_eq=A_eq if A_eq else None, b_eq=B_eq if B_eq else None,
                  bounds=limites, method="highs")
    
    st.markdown("---")
    st.header("3. Resultados y Análisis")
    
    if res.success:
        Z_opt = res.fun if tipo_opt == "Minimizar" else -res.fun
        st.success(f"**Solución Óptima encontrada:** Z = {Z_opt:.4f}")
        
        # Mostrar valores de las variables
        cols_res = st.columns(num_vars)
        for i in range(num_vars):
            cols_res[i].info(f"**X{i+1}** = {res.x[i]:.4f}")
            
        # Análisis de saturación de restricciones
        st.subheader("Estado de las Restricciones")
        for i in range(num_cons):
            # Calcular recurso consumido
            consumo = sum(A[i][j] * res.x[j] for j in range(num_vars))
            holgura = abs(B[i] - consumo)
            
            # Si la holgura es prácticamente cero, la restricción está activa/saturada
            if np.isclose(holgura, 0, atol=1e-5):
                estado = "🔴 Activa (Saturada)"
            else:
                estado = "🟢 Inactiva (con Holgura/Exceso)"
                
            st.write(f"- **Restricción {i+1}:** {estado} | Consumido: {consumo:.2f} / Límite: {B[i]:.2f} (Holgura: {holgura:.2f})")
            
        # --- VISUALIZACIÓN GRÁFICA (Solo para 2 variables) ---
     # --- VISUALIZACIÓN GRÁFICA (Solo para 2 variables) ---
        if num_vars == 2:
            st.subheader("Método Gráfico (2D)")
            
            # NUEVA LÓGICA DE ESCALA: Calculamos los cortes reales con los ejes X1 y X2
            max_x1 = max(res.x[0] * 1.5, 5) # Al menos un poco más allá del óptimo
            max_x2 = max(res.x[1] * 1.5, 5)
            
            for i in range(num_cons):
                # Solo usamos restricciones de tipo <= para acotar la vista principal
                if signos[i] == "<=":
                    if A[i][0] > 0: max_x1 = max(max_x1, (B[i] / A[i][0]) * 1.1)
                    if A[i][1] > 0: max_x2 = max(max_x2, (B[i] / A[i][1]) * 1.1)
            
            # Crear malla de puntos adaptada a los nuevos ejes
            d1 = np.linspace(0, max_x1, 400)
            d2 = np.linspace(0, max_x2, 400)
            X1, X2 = np.meshgrid(d1, d2)
            
            # Calcular región factible
            region_factible = np.ones_like(X1, dtype=bool)
            for i in range(num_cons):
                evaluacion = A[i][0] * X1 + A[i][1] * X2
                if signos[i] == "<=":
                    region_factible &= (evaluacion <= B[i])
                elif signos[i] == ">=":
                    region_factible &= (evaluacion >= B[i])
                else:
                    region_factible &= (np.abs(evaluacion - B[i]) < (max_x1+max_x2)*0.005)
                    
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Pintar región factible (aspect='auto' evita que se deforme si X1 y X2 tienen escalas distintas)
            ax.imshow(region_factible.astype(int), 
                      extent=(0, max_x1, 0, max_x2), origin='lower', cmap='Greens', alpha=0.3, aspect='auto')
            
           # Dibujar líneas de las restricciones y añadir etiqueta visual (sin decimales .0)
            for i in range(num_cons):
                texto_r = f"R{i+1}"
                
                if A[i][1] != 0:
                    # Ecuación completa para restricciones con X2 usando :g
                    texto_ecuacion = f"{texto_r}: {A[i][0]:g}X1 + {A[i][1]:g}X2 {signos[i]} {B[i]:g}"
                    y_linea = (B[i] - A[i][0] * d1) / A[i][1]
                    linea, = ax.plot(d1, y_linea, linewidth=2, label=texto_ecuacion)
                    
                    # Calcular posición para colocar el texto sobre la línea
                    x_texto = max_x1 * 0.15 * (i + 1)
                    if A[i][0] > 0 and x_texto > (B[i] / A[i][0]): 
                        x_texto = (B[i] / A[i][0]) * 0.5 
                    
                    y_texto = (B[i] - A[i][0] * x_texto) / A[i][1]
                    
                    # Imprimir el texto con la ecuación sobre la recta
                    if 0 <= y_texto <= max_x2:
                        ax.text(x_texto, y_texto, f" {texto_ecuacion} ", color=linea.get_color(), 
                                fontweight='bold', fontsize=9, 
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
                else:
                    # Ecuación completa para restricciones verticales (solo X1) usando :g
                    texto_ecuacion_vertical = f"{texto_r}: {A[i][0]:g}X1 {signos[i]} {B[i]:g}"
                    x_linea = B[i] / A[i][0]
                    linea = ax.axvline(x=x_linea, linewidth=2, label=texto_ecuacion_vertical)
                    ax.text(x_linea, max_x2 * 0.7, f" {texto_ecuacion_vertical} ", color=linea.get_color(), 
                            fontweight='bold', fontsize=9, 
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            
           # --- NUEVO: Calcular e identificar los vértices factibles (Vi) ---
            # 1. Recopilar todas las ecuaciones de las rectas (Restricciones + Ejes X e Y)
            lineas = [(A[i][0], A[i][1], B[i]) for i in range(num_cons)]
            lineas.extend([(1, 0, 0), (0, 1, 0)]) # Restricciones de no negatividad (X1=0, X2=0)
            
            vertices = []
            # 2. Calcular intersecciones evaluando todos los pares posibles de rectas
            for i in range(len(lineas)):
                for j in range(i + 1, len(lineas)):
                    a1, b1, c1 = lineas[i]
                    a2, b2, c2 = lineas[j]
                    det = a1 * b2 - a2 * b1
                    if abs(det) > 1e-8: # Si el determinante no es 0, las rectas se cruzan
                        x_int = (c1 * b2 - c2 * b1) / det
                        y_int = (a1 * c2 - a2 * c1) / det
                        
                        # 3. Comprobar si el punto de corte cumple TODAS las restricciones (es factible)
                        if x_int >= -1e-5 and y_int >= -1e-5:
                            es_factible = True
                            for k in range(num_cons):
                                val = A[k][0] * x_int + A[k][1] * y_int
                                if signos[k] == "<=" and val > B[k] + 1e-5: es_factible = False
                                elif signos[k] == ">=" and val < B[k] - 1e-5: es_factible = False
                                elif signos[k] == "=" and abs(val - B[k]) > 1e-5: es_factible = False
                            
                            # Si es factible y no lo hemos guardado ya, lo añadimos a la lista
                            if es_factible:
                                if not any(abs(v[0]-x_int)<1e-5 and abs(v[1]-y_int)<1e-5 for v in vertices):
                                    x_limpio = 0.0 if abs(x_int) < 1e-8 else x_int
                                    y_limpio = 0.0 if abs(y_int) < 1e-8 else y_int
                                    vertices.append((x_limpio, y_limpio))
            
            # Dibujar los vértices encontrados (Puntos negros)
            for idx, v in enumerate(vertices):
                ax.plot(v[0], v[1], 'ko', markersize=7) # 'ko' significa punto (o) negro (k)
                ax.text(v[0] + max_x1*0.02, v[1] + max_x2*0.02, f"V{idx+1}", color='black', 
                        fontweight='bold', fontsize=10, zorder=5,
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

            # Dibujar el punto óptimo (Punto rojo, dibujado después para que quede por encima de los negros)
            ax.plot(res.x[0], res.x[1], 'ro', markersize=11, zorder=6, label=f'Óptimo ({res.x[0]:g}, {res.x[1]:g})')
            
            # Dibujar curvas de nivel de la función objetivo
            Z_grid = C[0]*X1 + C[1]*X2
            curvas = ax.contour(X1, X2, Z_grid, levels=20, cmap='coolwarm', alpha=0.5)
            ax.clabel(curvas, inline=True, fontsize=8)
            
            ax.set_xlim(0, max_x1)
            ax.set_ylim(0, max_x2)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_title('Región Factible, Vértices y Solución Óptima')
            
            # Leyenda mejorada: colocada debajo del gráfico
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Usar use_container_width para que ocupe bien el espacio en Streamlit
            st.pyplot(fig, use_container_width=True)


    # --- NUEVO: Mostrar tabla con el valor numérico en cada vértice ---
            st.subheader("Análisis de Vértices y Valor Numérico")
            st.markdown("Valor numérico de la función objetivo (Z) evaluado en cada vértice factible:")
            
            datos_tabla = []
            for idx, v in enumerate(vertices):
                # Calcular el valor Z para este vértice concreto
                z_val = C[0]*v[0] + C[1]*v[1]
                
                # Comprobar si este vértice es la solución óptima
                es_optimo = "⭐ ÓPTIMO" if abs(z_val - Z_opt) < 1e-4 else ""
                
                # Guardar los datos en la tabla
                datos_tabla.append({
                    "Vértice": f"V{idx+1}",
                    "X1": round(v[0], 4),
                    "X2": round(v[1], 4),
                    "Valor Función Objetivo (Z)": round(z_val, 4),
                    "Estado": es_optimo
                })
            
            # Dibujar la tabla en la interfaz web
            st.table(datos_tabla)
            
    else:
        st.error("❌ El modelo es INFACTIBLE (no hay intersección en las restricciones) o NO ACOTADO (crece hasta el infinito).")
        st.write(f"Mensaje del solver: {res.message}")
