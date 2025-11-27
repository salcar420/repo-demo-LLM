import time
import ollama
import csv
import psutil  # <--- NUEVO: Para medir memoria RAM y CPU
import os

# --- 1. CONFIGURACIÓN PROFESIONAL ---
modelos = ['llama3', 'mistral', 'gemma:7b']

# --- AHORA LOS PROMPTS TIENEN CATEGORÍA ---
# Esto te permitirá hacer gráficos por "Habilidad" en Excel
prompts_data = [
    {
        "categoria": "Creatividad",
        "prompt": "Escribe un poema corto sobre el silencio en el espacio."
    },
    {
        "categoria": "Razonamiento",
        "prompt": "Tengo 5 camisas secándose al sol y tardan 2 horas. Si pongo 10 camisas, ¿cuánto tardan? Explica tu lógica."
    },
    {
        "categoria": "Coding",
        "prompt": "Escribe una función simple en Python para detectar si una palabra es un palíndromo."
    },
    {
        "categoria": "Resumen",
        "prompt": "Resume el siguiente texto en una frase: 'La inteligencia artificial es una rama de la informática que busca simular la inteligencia humana en máquinas.'"
    }
]

archivo_csv = 'benchmark_pro.csv'

# Función para obtener uso de RAM en MB
def obtener_uso_ram():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convertir a MB

# --- 2. FUNCIÓN PRINCIPAL ---
def correr_benchmark():
    print(f"🚀 Iniciando Benchmark PROFESIONAL de {len(modelos)} modelos...")
    print(f"🧪 Se evaluarán {len(prompts_data)} categorías por modelo.")
    print(f"📂 Guardando en: {archivo_csv}\n")

    with open(archivo_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # --- NUEVAS COLUMNAS: Categoría, RAM y CPU ---
        writer.writerow(['Modelo', 'Categoria', 'Prompt', 'Latencia_TTFT(s)', 'Total_Time(s)', 'Tokens_Seg(t/s)', 'RAM_Usada(MB)', 'Tokens_Total'])
        
        print(f"{'MODELO':<12} | {'CATEGORIA':<12} | {'VELOCIDAD':<10} | {'RAM (MB)':<10} | {'ESTADO'}")
        print("-" * 75)

        for modelo in modelos:
            for item in prompts_data:
                categoria = item['categoria']
                prompt = item['prompt']
                
                try:
                    # Medimos RAM antes de empezar
                    ram_inicio = psutil.virtual_memory().used / (1024 * 1024)
                    
                    start_time = time.time()
                    
                    # --- CONFIGURACIÓN AVANZADA ---
                    # options={'temperature': 0} hace que el modelo sea más lógico y menos "loco"
                    response = ollama.chat(
                        model=modelo, 
                        messages=[{'role': 'user', 'content': prompt}], 
                        stream=True,
                        options={'temperature': 0.1} 
                    )
                    
                    first_token_time = None
                    token_count = 0
                    
                    for chunk in response:
                        if first_token_time is None:
                            first_token_time = time.time()
                        token_count += 1
                    
                    end_time = time.time()
                    
                    # Medimos RAM al final (aproximación del pico de carga del sistema)
                    ram_fin = psutil.virtual_memory().used / (1024 * 1024)
                    ram_delta = ram_fin - ram_inicio # Cuánto subió la RAM
                    
                    # Cálculos
                    ttft = first_token_time - start_time if first_token_time else 0
                    total_time = end_time - start_time
                    tps = token_count / total_time if total_time > 0 else 0
                    
                    # Guardamos datos enriquecidos
                    writer.writerow([
                        modelo, 
                        categoria, 
                        prompt, 
                        f"{ttft:.4f}", 
                        f"{total_time:.4f}", 
                        f"{tps:.2f}", 
                        f"{ram_delta:.2f}", # Nueva métrica
                        token_count
                    ])
                    
                    print(f"{modelo:<12} | {categoria:<12} | {tps:<10.2f} | {ram_delta:<10.2f} | ✅ OK")

                except Exception as e:
                    print(f"{modelo:<12} | {categoria:<12} | ERROR      | 0.00       | ❌ {e}")

    print("\n" + "="*30)
    print("✅ ¡BENCHMARK PRO COMPLETADO!")

if __name__ == "__main__":
    correr_benchmark()