# Desarrollo de Aplicaciones con LLM y LangChain

Repositorio del sitio de documentación para el curso *Desarrollo de Aplicaciones con LLM y LangChain*, construido con MkDocs y el tema Material.

El sitio desplegado está disponible en: **https://jcphysica5.github.io/llm-langchain-apps/**

---

## Requisitos previos

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) o [Anaconda](https://www.anaconda.com/products/distribution)
- Terminal o interfaz de línea de comandos

## Instalación y uso local

### 1. Clonar el repositorio

```bash
git clone git@github.com:jcphysica5/Principios_de_IA_repo.git
cd Principios_de_IA_repo
```

### 2. Crear el entorno Conda

```bash
conda env create -f environment.yml
```

### 3. Activar el entorno

```bash
conda activate mkdocs-env
```

### 4. Servir el sitio localmente

```bash
mkdocs serve
```

Abre `http://127.0.0.1:8000` en tu navegador. El sitio se recarga automáticamente al guardar cambios.

### 5. Generar los archivos estáticos (opcional)

```bash
mkdocs build
```

## Despliegue

El despliegue se realiza automáticamente mediante GitHub Actions al hacer push a la rama `main`. El workflow se encuentra en `.github/workflows/deploy.yml`.
