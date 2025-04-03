# Generador de Imágenes con IA - Stable Diffusion 3.5

Este proyecto utiliza el modelo **Stable Diffusion 3.5 Large** para generar imágenes realistas a partir de descripciones en lenguaje natural (*prompts*), optimizado para ejecutarse en GPU con PyTorch.


## 👨‍💻 Autor

**Juan Pablo Corona**  
Universidad Técnica Federico Santa María  
**Inteligencia Artificial Generativa**

Características
Generación de imágenes con Stable Diffusion 3.5.

Soporte para ejecución en GPU (CUDA).

Medición de estadísticas de rendimiento: tiempo de inferencia, memoria usada, tiempo total.

Sugerencias automáticas para mejorar eficiencia y rendimiento.

##  Requisitos

- Python 3.8+
- CUDA Toolkit (si tienes GPU NVIDIA)
- PyTorch
- diffusers
- transformers
- accelerate
- Una cuenta en [Hugging Face](https://huggingface.co)

Ejemplo de uso

texto_descriptivo = "un programador tocando el pasto"
Estadísticas de salida
Al ejecutar el script, obtendrás:

Tiempo total y de inferencia
Memoria usada en GPU y memoria pico
Recomendaciones para optimizar el rendimiento
