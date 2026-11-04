# Portafolio académico y proyecto de IA

El sitio estático está en `docs/` y no requiere instalación ni compilación. Abre `docs/index.html` para revisarlo. Todos los recursos locales usan rutas relativas, compatibles con `https://ilululf.github.io/IA_2026/`.

## Completar el CV

Busca los textos entre corchetes en `docs/index.html` y reemplaza:

- `[Tu nombre]`, también en el título del navegador y el pie de página.
- `[Tu título profesional o carrera]` y la presentación personal.
- Título o carrera, institución, años y formación complementaria.
- Correo y LinkedIn. Sustituye los bloques de contacto correspondientes por enlaces `mailto:tu@correo.com` y `https://www.linkedin.com/in/tu-perfil/`.
- Habilidades adicionales y nivel de experiencia. La lista actual identifica tecnologías del código, sin afirmar dominio personal.

`docs/assets/profile.svg` es un gráfico original representativo del proyecto. Puedes reemplazarlo por una foto y actualizar su texto alternativo.

## Publicar en GitHub Pages

1. Sube `docs/` y `.github/workflows/pages.yml` a la rama `main` de `ilululf/IA_2026`.
2. En GitHub abre **Settings → Pages → Build and deployment → Source → GitHub Actions**.
3. En **Actions**, ejecuta **Publicar portafolio en GitHub Pages → Run workflow**, o realiza un cambio en `docs/` y súbelo a `main`.
4. Comprueba que la ejecución termina correctamente. La URL prevista es `https://ilululf.github.io/IA_2026/`; no debe darse por publicada hasta que finalice el despliegue.

Si prefieres publicar sin Actions, sube solamente `docs/` y selecciona **Deploy from a branch → main → /docs** en Settings → Pages. Usa un solo método de despliegue.

## Datos y resultados

Las imágenes de ejemplo y la matriz de confusión proceden del repositorio original. Los archivos `test_metrics.json` y `result_summary.json` se conservan como evidencia del experimento `pchip_vgg16_model1_srb_vs_no_srb_stage2_block5_w060_040_lr1em07_v5`.

La accuracy 66,55 %, macro F1 0,581 y ROC AUC 0,671 corresponden únicamente al test de SRB/NO_SRB, con 278 muestras. No son métricas de la cascada completa ni una afirmación de que sea el mejor experimento. El diagrama mantiene la cascada propuesta por la autora o autor; el repositorio incluye también experimentos independientes cuyo orden y población de entrada son diferentes.

Los resultados e imágenes se guardaron como una instantánea del repositorio consultado el 3 de noviembre de 2026, commit `af51f1732c0ef16e2e963987cd5a83e71c0c3e8a`. Los enlaces del sitio permiten consultar la versión actual de los archivos.
