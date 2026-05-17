import os
import groq

def rewrite_scene(base_prompt: str, chunk_text: str) -> str:
    """
    Takes a base prompt from style_transfer_prompts (which contains hardcoded scenes)
    and replaces the hardcoded scene with the actual user-selected chunk text,
    then calls Groq to perform the style transfer.
    """
    api_key = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY_2")
    if not api_key:
        return "Error: Neither GROQ_API_KEY_1 nor GROQ_API_KEY_2 found in environment."
        
    client = groq.Groq(api_key=api_key)
    
    # Extract just the style rules from the prompt (everything before "ORIGINAL SCENE")
    if "ORIGINAL SCENE FROM RRR" in base_prompt:
        rules_part = base_prompt.split("ORIGINAL SCENE FROM RRR")[0].strip()
    else:
        rules_part = base_prompt
        
    # Rebuild the prompt with the actual selected chunk text
    full_prompt = (
        f"{rules_part}\n\n"
        f"ORIGINAL SCENE TO REWRITE:\n"
        f"---\n{chunk_text}\n---\n\n"
        f"TASK:\n"
        f"Rewrite this scene COMPLETELY following the style rules above.\n"
        f"- CRITICAL: Keep ALL character names EXACTLY as they appear in the original (e.g. Bheem, Ram, Malli, Jenny, Scott). Do NOT rename any character under any circumstances.\n"
        f"- Keep the exact same plot events and meaning.\n"
        f"- Completely transform the dialogue tone, the scene setting, and the vibe.\n"
        f"- Add proper screenplay formatting (scene headings, action lines, character names).\n"
    )
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a master screenplay writer."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.8,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during LLM generation: {str(e)}"
