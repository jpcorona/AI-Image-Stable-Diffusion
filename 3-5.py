import torch
import time
from diffusers import StableDiffusion3Pipeline

llm = "stabilityai/stable-diffusion-3.5-large"

tiempo_inicio_total = time.time()

# Cargar el pipeline del modelo
pipeline = StableDiffusion3Pipeline.from_pretrained(llm, torch_dtype=torch.bfloat16)
pipeline.enable_attention_slicing()
pipeline = pipeline.to("cuda")

texto_descriptivo = "un programador tocando el pasto"

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

tiempo_inicio_inferencia = time.time()

resultado = pipeline(
    texto_descriptivo,
    num_inference_steps=20,   # Más pasos mejora calidad pero toma más tiempo
    guidance_scale=3.5,       # Controla fidelidad al texto (más alto = más exacto)
    height=512,
    width=512
)

tiempo_fin_inferencia = time.time()

imagenes = resultado.images

for i, imagen in enumerate(imagenes):
    imagen.save(f"imagen_{i}.png")

# Medición de tiempo total
tiempo_fin_total = time.time()

# Estadísticas de rendimiento
tiempo_total = tiempo_fin_total - tiempo_inicio_total
tiempo_inferencia = tiempo_fin_inferencia - tiempo_inicio_inferencia
memoria_usada = torch.cuda.memory_allocated() / 1e6  # en MB
memoria_pico = torch.cuda.max_memory_allocated() / 1e6  # en MB

# Mostrar estadísticas
print("\n--- Estadísticas de Rendimiento ---")
print(f" Tiempo total de ejecución: {tiempo_total:.2f} segundos")
print(f" Tiempo de inferencia: {tiempo_inferencia:.2f} segundos")
print(f" Memoria GPU usada: {memoria_usada:.2f} MB")
print(f" Pico de memoria GPU: {memoria_pico:.2f} MB")

# Sugerencias para mejorar rendimiento:
print("\n--- Sugerencias para optimización ---")
print("✅ Reducir 'guidance_scale' si no se requiere tanta fidelidad al prompt.")
print("✅ Usar imágenes de menor resolución (por ejemplo 384x384).")
print("✅ Activar 'attention slicing' (ya está activado).")
print("✅ Considerar usar precision float16 si tu GPU lo permite.")
print("✅ Si tienes múltiples imágenes que generar, agrúpalas en batch.")

