# Integración de GitHub con Railway y Uso de Filtros

Este README explica cómo conectar tu repositorio de GitHub con Railway para desplegar automáticamente tu proyecto, y cómo configurar filtros en GitHub Actions (o en la integración nativa de Railway) para controlar cuándo y cómo se ejecutan los despliegues.

---

## 1. Introducción

Railway es una plataforma de despliegue que permite conectar directamente tu repositorio de GitHub para que, al hacer push a ciertas ramas o cambios específicos, tu aplicación se despliegue automáticamente. Los “filtros de GitHub” (en el contexto de GitHub Actions o de la integración nativa de Railway) son mecanismos que te permiten indicar en qué ramas, rutas o eventos debe activarse ese despliegue.

---

## 2. Pre-requisitos

Antes de comenzar, asegúrate de tener lo siguiente:

1. **Cuenta en Railway**  
   - Si aún no tienes una, regístrate en [railway.app](https://railway.app/) y crea un nuevo proyecto.

2. **Repositorio en GitHub**  
   - Tu código fuente (por ejemplo, una aplicación de Unity o cualquier otro proyecto) debe estar alojado en un repositorio público o privado en GitHub.

3. **Acceso al Railway CLI (opcional)**  
   - Para configuraciones avanzadas o despliegues manuales, instala el CLI de Railway:  
     ```bash
     npm install -g railway
     ```
   - Pero para la mayoría de casos, bastará con la integración directa desde la interfaz web de Railway.

---

## 3. Conectar GitHub con Railway

### 3.1. Crear un Proyecto en Railway

1. Ingresa a tu panel de Railway.  
2. Haz clic en **“New Project”** y selecciona **“Deploy from GitHub”**.  
3. Autoriza a Railway a acceder a tu cuenta de GitHub (si aún no lo has hecho).  
4. Elige el repositorio que quieras desplegar.  

> **Resultado:** Railway creará un pipeline vinculado a ese repositorio y, por defecto, desplegará cada vez que hagas push a la rama principal (`main` o `master`).

### 3.2. Configurar Variables de Entorno y Secrets

1. Dentro del proyecto de Railway, ve a la pestaña **“Variables”**.  
2. Agrega las variables necesarias para tu aplicación (por ejemplo, claves de API, cadenas de conexión a bases de datos, etc.).  
3. Railway guardará estas variables en un entorno seguro y tu aplicación podrá leerlas en tiempo de ejecución.

---

## 4. Entendiendo los Filtros de GitHub

Existen dos enfoques principales para aplicar filtros que controlen cuándo se dispara un despliegue:

1. **Filtros nativos en la integración de Railway**  
2. **Filtros en GitHub Actions** (archivo `.github/workflows/ci.yml` o similar)

### 4.1. Filtros en la Integración Nativa de Railway

- Al conectar GitHub vía la interfaz de Railway, por defecto Railway escucha eventos `push` y `pull_request` en la rama principal.  
- Si quieres desplegar solo cuando se actualice una rama distinta o cuando cambien archivos en ciertas rutas, puedes ir a **“Settings” → “Deployments”** en Railway y ajustar:  
  - **Branch**: selecciona la(s) rama(s) que dispararán el despliegue (por ejemplo, `main`, `develop`, etc.).  
  - **Path Filters** (cuando esté disponible): establece rutas o patrones de archivos. Por ejemplo, `src/**` o `Assets/**` en proyectos de Unity para que Railway solo despliegue si se modifica código relevante.

> **Ejemplo de filtro de ruta (si la interfaz lo permite):**  
> - `src/**` → despliega cuando haya cambios dentro de la carpeta `src/`.  
> - `Assets/Scenes/**` → en un proyecto Unity, despliega solo si cambian escenas.

### 4.2. Filtros en GitHub Actions

Si prefieres un control más granular o ya tienes flujos de trabajo (`workflows`) propios, puedes crear un archivo en `.github/workflows/deploy.yml` con filtros de activación:

```yaml
name: “Desplegar en Railway”

on:
  push:
    branches:
      - main                   # → Despliega solo cuando hagas push a main
      - release/*              # → También despliega en ramas que empiecen con “release/”
    paths:
      - “src/**”               # → Solo si cambia algo en “src/”
      - “README.md”
  pull_request:
    branches:
      - main                   # → Ejecuta validaciones en PRs dirigidos a main

jobs:
  deploy:
    name: “Despliegue a Railway”
    runs-on: ubuntu-latest

    steps:
      - name: “Revisar código”
        uses: actions/checkout@v3

      - name: “Instalar Railway CLI”
        run: npm install -g railway

      - name: “Login en Railway”
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
        run: |
          railway login --ci $RAILWAY_TOKEN

      - name: “Seleccionar Proyecto”
        run: railway link --project ${{ secrets.RAILWAY_PROJECT_ID }}

      - name: “Desplegar”
        run: railway up --detach
