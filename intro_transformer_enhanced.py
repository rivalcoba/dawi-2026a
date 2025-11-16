# Importando pipeline de transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

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
        pad_token_id= AutoTokenizer.from_pretrained(model_name).eos_token_id)
    
    return generator

def generate_text(generator, prompt, max_length=1000):
    result = generator (
        prompt, 
        max_length=max_length, 
        num_return_sequences=1,
        do_sample=True, 
        temperature=0.7)
    
    return result[0]['generated_text']

def run_llm_demo():
    """_summary_
    Demuestra la funcionalidad basica de un LLM con
    explicaciones
    """
    print("🤖 Cargando el modelo LLM...")
    generator = create_simple_lm()
    
    print("\n✨ Demo de un LLM✨")
    print("Este demo muestra la generacion de texto usando un modelo LLM pequeño basado en GPT-2 destilado.")
    
    # Prompt para demostrar diferentes capacidades
    prompts = [
        "Once upon a time",
        "The quick brown fox jumps over the lazy dog.",
        "Javascript programming is",]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        generated_text = generate_text(generator, prompt)
        input("\nPresiona Enter para ver el resultado...")
        print(f"Generated Text: {generated_text}")

def interactive_demo():
    """_summary_
    Permite al usuario ingresar prompts y ver
    las respuestas generadas por el LLM de forma interactiva.
    """
    print("🤖 Cargando el modelo LLM para demo interactiva...")
    generator = create_simple_lm()
    
    print("\n✨ Demo Interactiva de un LLM✨")
    print("Ingresa un prompt para generar texto. Escribe 'exit' para salir.")
    
    while True:
        prompt = input("\nIngresa tu prompt: ")
        if prompt.lower() == 'exit':
            print("Saliendo de la demo interactiva.")
            break
        
        response = generate_text(generator, prompt)
        print(f"Generated Text: {response}")

def explain_process():
    """_summary_
    Explica el funcionamiento del LLM con un ejemplo.
    """
    print("🤖 Explicando el proceso del LLM...")
    print("1. Se ingresa el texto -> Tokenización -> Numeros")
    print("2. Numeros -> Proceso del Modelo -> Predicciones")
    print("3. Predicciones -> Nuevo Token -> Salida de Texto")
    
    # Ejemplo de Tokenización
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    text = "Hello, how are you?"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    print(f"\nEjemplo de Tokenización:")
    print(f"Texto original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Texto decodificado: {decoded}")

if __name__ == "__main__":
    print("Elige una opción:")
    print("1. Demostración básica")
    print("2. Modo interactivo")
    print("3. Explicar el proceso del LLM")

    choice = input("Ingresa tu opción (1-3): ")

    if choice == "1":
        run_llm_demo()
    elif choice == "2":
        interactive_demo()
    elif choice == "3":
        explain_process()
    else:
        print("Invalid choice!")