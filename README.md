# BinPackingGA_Thania.Jose

**Problema de Bin Packing resuelto mediante una heurística de Algoritmo Genético**

---

## Instrucciones para configurar y ejecutar el proyecto

### 1. **Pre-requisitos**
   - Asegúrate de tener instalado Python 3.8 o superior. Puedes descargarlo desde [python.org](https://www.python.org/).

### 2. **Instalación de dependencias**
   - Clona este repositorio o descarga los archivos del proyecto.
   - Abre una terminal en la carpeta del proyecto y ejecuta el siguiente comando para instalar las bibliotecas requeridas:

     ```bash
     pip install -r requirements.txt
     ```

   - Si no tienes un archivo `requirements.txt`, instala las bibliotecas directamente con:

     ```bash
     pip install pygame numpy matplotlib
     ```

### 3. **Configuración inicial**
   - Abre el archivo `proyecto.py` en tu editor de texto o IDE favorito.
   - Modifica los parámetros iniciales según tus datos de entrada:
     - Ajusta la lista `ITEMS` con los pesos de los objetos que deseas procesar.
     - Cambia el valor de `BIN_CAPACITY` para definir la capacidad máxima de los contenedores.

### 4. **Ejecución del programa**
   - Una vez configurado, guarda los cambios y ejecuta el programa desde la terminal con el comando:

     ```bash
     python proyecto.py
     ```

### 5. **Visualización de resultados**
   - Durante la ejecución, el programa mostrará:
     - Una visualización gráfica de los contenedores y los ítems distribuidos.
     - Información sobre la generación actual del algoritmo.
   - Al finalizar, se graficará la evolución del fitness a lo largo de las generaciones.

---

## Notas adicionales

- **Interacción en la visualización gráfica:**
  - Usa las teclas de flecha para desplazarte (arriba, abajo, izquierda, derecha) si los contenedores exceden el tamaño de la pantalla.

- **Parámetros del algoritmo genético:**
  - Puedes ajustar estos valores en el código para experimentar con el rendimiento:
    - `population_size`: Número de soluciones iniciales en la población.
    - `generations`: Número máximo de generaciones a ejecutar.
    - `truncation_percentage`: Porcentaje de truncamiento para la selección.
    - `elite_size`: Tamaño de la élite preservada durante la reinserción.

- **Dependencias principales:**
  - `pygame`: Para la visualización gráfica.
  - `numpy`: Para cálculos matemáticos.
  - `matplotlib`: Para graficar la evolución del fitness.

---

Con esta guía, deberías poder configurar, ejecutar y explorar el funcionamiento de este proyecto fácilmente. :D
