<!-- Presentación general del curso

-------------------------------------
Preliminares
Presentación general del curso 
-------------------------------------

Nombre del curso: Diplomado en construcción de Aplicaciones Asistidas por IA



-->
# Building AI-Powered Applications with LLMs

![Banner del curso](assets/images/Banner.png)


<!--
*********           PRELIMINARES  *****************************


Describe de manera precisa y comprensible el propósito general del curso o asignatura, así como sus particularidades, enfatizando su relevancia práctica para el estudiante.



Describe la relevancia del contenido del curso para la formación, esto es: los saberes a explorar, las habilidades a desarrollar, qué metodología se empleará, cuál es el proceso de evaluación y cómo están estructuradas los módulos del curso. 

Para redactar la presentación, ten en cuenta las siguientes recomendaciones:

Establece el objetivo general y los específicos, o los resultados de aprendizaje (según sea el caso).
Inicia con un contexto histórico o geográfico sobre el tema central del curso.
Describe brevemente el tema central, es decir, lo que el estudiante aprenderá.
Añade aspectos que resulten significativos para el estudiante, por qué es necesario conocer este tema, cómo se aplicará en su quehacer profesional o académico, para qué le servirá en el presente y a futuro. Señala también la importancia del curso en el marco del programa.
Relaciona un ejemplo, algunas cifras notables o la aplicación principal de los conceptos para demostrar su relevancia.
Menciona qué habilidades se pueden desarrollar.
Indica las unidades de las que consta el curso y los temas a abordar en cada una.
Finaliza con un párrafo de cierre en el cual motives al estudiante a realizar el curso.

 
** Esta presentación no debe superar las 300 palabras.  



-->

Aunque la inteligencia artificial ha existido como un campo exitoso y prometedor entre los expertos durante varias décadas, la llegada de capacidades computacionales más avanzadas —ofrecidas por las GPU modernas— y las habilidades demostradas con el lanzamiento de ChatGPT fueron una gran sorpresa para muchos.

No está del todo claro cómo este "pequeño" avance en la escala de los modelos pudo desencadenar la gran cantidad de aplicaciones asistidas por IA que estamos viendo explotar cada semana. Lo que sí es claro es que el campo de la ingeniería de software está siendo revolucionado, y que el nuevo paradigma de construcción de software ya no consiste en los tradicionales flujos de ejecución, sino que la nueva ingeniería de sistemas debe integrar a los LLM en dichos flujos.

Hemos diseñado este curso para introducirte a este nuevo paradigma. Este curso está basado en LangChain, el framework más popular de la actualidad para interactuar con las APIs de los principales modelos de lenguaje.

<!--

*******************************Resultados de aprendizaje******************
Establecen las dinámicas de ENSEÑANZA-APRENDIZAJE dentro del curso y encaminan el proceso hacia lo que queremos que los estudiantes sepan, comprendan y sean capaces de hacer al finalizar el curso.


Esta información se extrae de la carta descriptiva, por esa razón es importante consultarla antes de redactar esta parte. Ten presente que la versión en Word contiene el objetivo general y los específicos; mientras que la versión en Excel contiene los resultados de aprendizaje.

-->
<!--
Pregunta orientadora
Es un interrogante que sirve como punto de partida para la exploración del tema central del curso, y está diseñado para dirigir la atención del estudiante, reconociendo de qué manera se apropia de ese saber. A través de esta pregunta, el conocimiento se logra concretar en una respuesta que recoge la esencia del curso, guiando al estudiante hacia el descubrimiento de conceptos importantes o la resolución de problemas dentro de un contexto determinado.



Formula la pregunta hablando al estudiante, de manera directa, concisa y sin ambigüedades. 
Evita utilizar términos confusos o complejos que dificulten su comprensión.
Recuerda que el estudiante dará respuesta a la pregunta orientadora al terminar el curso, por ello, es importante contextualizarla con un dato de interés o mediante un caso específico. 
La respuesta a esta pregunta se afianza o ejercita durante todo el proceso por medio de las evidencias de aprendizaje. Se espera que, al finalizar el curso, la respuesta tenga amplia relación con la actividad final.

**Procura no emplear más de 200 palabras. 
-->


La siguiente imagen es la respuesta de ChatGPT al prompt:

*“Based on what you know about me, draw a picture of what you think my current life looks like.”*

![alt text](image.png)

*Figura 1: Representación visual generada por ChatGPT sobre la vida del autor del diplomado.*

¿Qué tan parecida es esta imagen a tu vida en tu caso? ¿Te sorprende que ChatGPT tenga tanta información sobre tu vida y tus gustos o todo lo contrario? ¿Qué opinas que va a pasar con los datos en un futuro donde todo es asistido por IA? ¿Le daremos a las IA la información sensible de las personas y las empresas?

***¿Cómo podemos asegurarnos de que nuestros datos permanezcan privados, y al mismo tiempo, aprovechar todo el poder de los modelos de lenguaje en los datos privados?***

<!--

Mapa del curso
Es una herramienta visual que proporciona una visión general de la estructura y el contenido del curso.Se presenta en forma de diagrama y muestra los temas del curso divididos en unidades temáticas.


Es importante relacionar el nombre del curso, de sus respectivas unidades y de los saberes o temáticas correspondientes a cada una de ellas. Esta información se obtiene del formato de planeación o de la carta descriptiva. 

Ejemplos:
       

** Tanto el mapa del curso como cualquier gráfico de autoría propia y/o adaptado de otros autores deben entregarse en formato editable. 

-->

## Configuración del sistema antes de comenzar

Antes de empezar a trabajar con los módulos del curso, debes configurar tu sistema para poder ejecutar los ejemplos correctamente.

La forma más sencilla de hacerlo es descargando el archivo de configuración [📄environment.yml](assets/resources/environment.yml)
, el cual creará automáticamente un entorno de Conda llamado `DiplomadoIA_env` con todas las dependencias necesarias para el curso.

### Requisitos previos

- Tener **Anaconda** instalado en tu computador.
- Usar una terminal Bash (en Windows puedes usar Anaconda Prompt, git bash, WSL o similares).

### Instalación

Una vez descargado el archivo de configuración, ejecuta el siguiente comando en tu terminal:

```bash
conda env create -f environment.yml
```
### Activación del entorno

Para activar el entorno en tu terminal, ejecuta:

```bash
conda activate diplomado_IA
```
A partir de aquí, cualquier comando que ejecutes usará las dependencias definidas para el curso.

### Uso del entorno en Visual Studio Code

!!! warning "Para tener en cuenta"
   
    Para ejecutar notebooks `.ipynb` en Visual Studio Code usando este entorno:

    1. Abre **VS Code**.
    2. Abre la carpeta del proyecto o el notebook deseado.
    3. En la parte superior derecha del notebook, haz clic en la selección de kernel.
    4. Elige el kernel correspondiente al entorno `diplomado_IA`.  
    Si no aparece, reinicia VS Code o asegúrate de haber activado el entorno desde la terminal integrada.
    5. Comienza a ejecutar celdas normalmente.

!!! tip
    Puedes asegurarte de que el entorno se registre correctamente como kernel ejecutando en la terminal:
    ```bash
    python -m ipykernel install --user --name diplomado_IA --display-name "Python (diplomado_IA)"
    ```

