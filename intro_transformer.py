# Importando pipeline de transformers
from transformers import pipeline, AutoTokenizer

def create_simple_lm():
    """_summary_
    Crea un LLM simple usando una version destilada
    de GPT-2 para generación de texto. El modelo
    GTP-2 tiene 124 millones de parámetros. Es adecuado
    para tareas básicas de generación de texto. No requiere
    uso de GPU.
    """
    # Nombre del modelo
    model_name = "distilgpt2"
    
    # Creando el generador de texto
    generator = pipeline(
        "text-generation",
        model=model_name,
        pad_token_id= AutoTokenizer.from_pretrained(model_name).eos_token_id       
    )
    
    return generator

# Creando una instancia del LLM simple
text_generator = create_simple_lm()

# Prompt de ejemplo
prompt = "Once upon a time"

# Generacion del texto
outputs = text_generator(
    prompt, # Texto de entrada
    max_length=50, # Longitud máxima de la secuencia generada
    num_return_sequences=1 # Número de secuencias a generar
    )

# Imprimiendo el resultado
print(outputs[0]['generated_text'])